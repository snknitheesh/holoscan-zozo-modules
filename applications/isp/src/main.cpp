// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ISP: a synthetic raw camera (holoscan-isp's colour checker scene, moving,
// mosaicked with known black levels, white balance and vignetting) feeds
// holoscan-isp's IspOp, which turns each Bayer frame into RGB on the GPU. A
// checker measures every frame's PSNR against the camera's truth, and with a
// display HoloViz shows the raw mosaic beside the processed image.

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/isp/isp_op.hpp>
#include <holoscan/isp/isp_pipeline.hpp>
#include <holoscan/isp/synthetic_raw.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if ISP_DEMO_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "check_op.hpp"
#include "raw_source_op.hpp"
#include "render_check_op.hpp"
#include "view_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace app = isp_demo;
namespace isp = holoscan::isp;

constexpr std::string_view kAppName = "isp";

enum class DisplayMode { kAuto, kOn, kOff, kOffscreen };

struct Options {
  std::uint32_t count{300};
  std::uint32_t period_ms{33};
  int width{1280};
  int height{720};
  isp::BayerPattern pattern{isp::BayerPattern::kRGGB};
  int bits{12};
  isp::RawFormat format{isp::RawFormat::kUint16};
  isp::DemosaicMethod demosaic{isp::DemosaicMethod::kMalvar};
  isp::WhiteBalanceMode white_balance{isp::WhiteBalanceMode::kManual};
  isp::SceneKind scene{isp::SceneKind::kChart};
  float sharpen{0.0F};
  isp::Backend backend{isp::Backend::kCuda};
  std::optional<double> min_psnr;
  DisplayMode display{DisplayMode::kAuto};
  std::uint32_t timeout_seconds{120};
  bool compile_only{false};
  bool help{false};
};

template <typename T>
T parse_number(std::string_view value, std::string_view option) {
  T result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc() || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(option) + " requires a number, got '" +
                                std::string(value) + "'");
  }
  return result;
}

bool parse_bool(std::string_view value, std::string_view option) {
  if (value != "true" && value != "false") {
    throw std::invalid_argument(std::string(option) + " requires true or false");
  }
  return value == "true";
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string_view option{argv[i]};
    if (option == "--help" || option == "-h") {
      opts.help = true;
      return opts;
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(option));
    }
    const std::string_view value{argv[++i]};
    if (option == "--count") {
      opts.count = parse_number<std::uint32_t>(value, option);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--width") {
      opts.width = parse_number<int>(value, option);
    } else if (option == "--height") {
      opts.height = parse_number<int>(value, option);
    } else if (option == "--pattern") {
      const auto pattern = isp::parse_pattern(value);
      if (!pattern) throw std::invalid_argument("--pattern must be RGGB, BGGR, GRBG or GBRG");
      opts.pattern = *pattern;
    } else if (option == "--bits") {
      opts.bits = parse_number<int>(value, option);
    } else if (option == "--packing") {
      if (value == "none") {
        opts.format = isp::RawFormat::kUint16;
      } else if (value == "raw10") {
        opts.format = isp::RawFormat::kRaw10;
      } else if (value == "raw12") {
        opts.format = isp::RawFormat::kRaw12;
      } else {
        throw std::invalid_argument("--packing must be none, raw10 or raw12");
      }
    } else if (option == "--demosaic") {
      const auto method = isp::parse_demosaic(value);
      if (!method) throw std::invalid_argument("--demosaic must be bilinear or malvar");
      opts.demosaic = *method;
    } else if (option == "--white-balance") {
      const auto mode = isp::parse_white_balance(value);
      if (!mode) {
        throw std::invalid_argument("--white-balance must be manual, gray-world or white-patch");
      }
      opts.white_balance = *mode;
    } else if (option == "--scene") {
      if (value != "chart" && value != "gray") {
        throw std::invalid_argument("--scene must be chart or gray");
      }
      opts.scene = value == "chart" ? isp::SceneKind::kChart : isp::SceneKind::kGray;
    } else if (option == "--sharpen") {
      opts.sharpen = parse_number<float>(value, option);
    } else if (option == "--backend") {
      const auto backend = isp::parse_backend(value);
      if (!backend) throw std::invalid_argument("--backend must be cuda, cpu or auto");
      opts.backend = *backend;
    } else if (option == "--min-psnr") {
      opts.min_psnr = parse_number<double>(value, option);
    } else if (option == "--display") {
      if (value == "auto") {
        opts.display = DisplayMode::kAuto;
      } else if (value == "true") {
        opts.display = DisplayMode::kOn;
      } else if (value == "false") {
        opts.display = DisplayMode::kOff;
      } else if (value == "offscreen") {
        opts.display = DisplayMode::kOffscreen;
      } else {
        throw std::invalid_argument("--display must be auto, true, false or offscreen");
      }
    } else if (option == "--timeout-seconds") {
      opts.timeout_seconds = parse_number<std::uint32_t>(value, option);
    } else if (option == "--compile-only") {
      opts.compile_only = parse_bool(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (opts.format == isp::RawFormat::kRaw10) opts.bits = 10;
  if (opts.format == isp::RawFormat::kRaw12) opts.bits = 12;
  if (opts.format == isp::RawFormat::kUint16 && opts.bits <= 8)
    opts.format = isp::RawFormat::kUint8;
  return opts;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "  --count N                Frames (default: 300)\n"
         "  --period-ms N            Camera period, 1..1000 ms (default: 33)\n"
         "  --width N --height N     Image size, 16x16 to 4096x4096 (default: 1280x720)\n"
         "  --pattern P              Bayer pattern: RGGB, BGGR, GRBG or GBRG (default: RGGB)\n"
         "  --bits N                 Bits per sample, 2..16; 8 or less in uint8 (default: 12)\n"
         "  --packing none|raw10|raw12  MIPI CSI-2 packing (default: none)\n"
         "  --demosaic malvar|bilinear  (default: malvar)\n"
         "  --white-balance manual|gray-world|white-patch  (default: manual, the camera's gains)\n"
         "  --scene chart|gray       The colour chart, or grays only (default: chart)\n"
         "  --sharpen A              Unsharp mask amount, 0..4 (default: 0)\n"
         "  --backend cuda|cpu|auto  Where the ISP runs (default: cuda); cpu needs --display "
         "false\n"
         "  --min-psnr DB            Every frame's PSNR must exceed this (default: 28 for malvar,\n"
         "                           26 for bilinear, 24 for 8 bits or fewer)\n"
         "  --display auto|true|false|offscreen\n"
         "                           Show raw and processed in HoloViz (default: auto, on when a\n"
         "                           display is available); offscreen renders into CUDA memory\n"
         "  --timeout-seconds N      Allowance beyond the expected run time (default: 120)\n"
         "  --compile-only true|false  Build and check the graph without running it\n";
}

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

isp::SyntheticRawConfig camera_of(const Options& opts) {
  isp::SyntheticRawConfig camera;
  camera.width = opts.width;
  camera.height = opts.height;
  camera.pattern = opts.pattern;
  camera.format = opts.format;
  camera.bit_depth = opts.bits;
  // About 6 % of full scale, as sensors have, slightly different per channel.
  const float black = std::round(0.0625F * static_cast<float>((1 << opts.bits) - 1));
  const float step = std::max(1.0F, black / 32.0F);
  camera.black_levels = {black, black + step, black, black + 2.0F * step};
  camera.vignetting = 0.4F;
  camera.kind = opts.scene;
  return camera;
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool cuda = isp::resolve_backend(opts.backend) == isp::Backend::kCuda;
  const bool display = opts.display == DisplayMode::kOn ||
                       (opts.display == DisplayMode::kAuto && display_available &&
                        ISP_DEMO_ENABLE_VISUALIZATION && cuda);
  const bool offscreen = opts.display == DisplayMode::kOffscreen;
  if ((display || offscreen) && !ISP_DEMO_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if ((display || offscreen) && !cuda) {
    throw std::invalid_argument("the display needs the cuda backend; add --display false");
  }
  if (display && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add --display false");
  }

  const isp::SyntheticRawConfig camera = camera_of(opts);
  isp::IspOpConfig isp_config;
  isp_config.backend = opts.backend;
  isp_config.params = isp::synthetic_isp_params(camera);
  isp_config.params.demosaic = opts.demosaic;
  isp_config.params.white_balance = opts.white_balance;
  isp_config.params.sharpen = opts.sharpen;
  if (opts.white_balance != isp::WhiteBalanceMode::kManual) {
    // Start from neutral gains, so that what the estimate finds shows.
    isp_config.params.wb_gains = {1.0F, 1.0F, 1.0F};
  }
  isp_config.packing = opts.format == isp::RawFormat::kRaw10   ? isp::Packing::kMipiRaw10
                       : opts.format == isp::RawFormat::kRaw12 ? isp::Packing::kMipiRaw12
                                                               : isp::Packing::kNone;
  isp_config.stats = true;
  isp_config.max_width = opts.width;
  isp_config.max_height = opts.height;

  holoscan::Graph graph{std::string(kAppName)};
  auto finished = std::make_shared<std::atomic<std::uint32_t>>(0U);
  auto source = graph.op<app::RawSourceOp>(
      "source", app::RawSourceConfig{.camera = camera,
                                     .count = opts.count,
                                     .period = std::chrono::milliseconds(opts.period_ms),
                                     .finished = finished});
  auto processor = graph.op<isp::IspOp>("isp", isp_config);
  auto report = std::make_shared<app::Report>();
  auto check = graph.op<app::CheckOp>("check", app::CheckConfig{.camera = camera,
                                                                .params = isp_config.params,
                                                                .expected_frames = opts.count,
                                                                .report = report,
                                                                .finished = finished});

  // Lossless edges: every frame is processed and checked, in order. The source
  // keeps one frame in flight, so no queue fills.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->raw, processor->raw, lossless);
  graph.add_flow(processor->image, check->image, lossless);
  graph.add_flow(processor->stats, check->stats, lossless);
  graph.add_flow(processor->histogram, check->histogram, lossless);

  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  on_gpu("source", "raw");
  if (cuda) {
    for (const char* port : {"image", "histogram", "stats"}) on_gpu("isp", port);
  }

#if ISP_DEMO_ENABLE_VISUALIZATION
  auto rendered = std::make_shared<app::RenderTally>();
  if (display || offscreen) {
    namespace viz = holoscan::visualization;
    // The raw mosaic on the left, the processed image on the right.
    auto view =
        graph.op<app::ViewOp>("view", app::ViewConfig{.width = opts.width,
                                                      .height = opts.height,
                                                      .format = camera.format,
                                                      .bit_depth = camera.bit_depth,
                                                      .black_level = camera.black_levels[0]});
    viz::LayerStack layers;
    layers.add_image_layer(
        "view", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                    .max_width = static_cast<std::uint32_t>(2 * opts.width),
                                    .max_height = static_cast<std::uint32_t>(opts.height)});
    // On screen, or offscreen into CUDA memory, where RenderCheckOp looks at each frame.
    viz::SinkDesc sink = viz::WindowSinkDesc{.title = "ISP: raw | processed",
                                             .width = static_cast<std::uint32_t>(2 * opts.width),
                                             .height = static_cast<std::uint32_t>(opts.height)};
    if (offscreen) {
      sink = viz::CudaSinkDesc{.width = static_cast<std::uint32_t>(2 * opts.width),
                               .height = static_cast<std::uint32_t>(opts.height)};
    }
    auto visualizer = graph.op<viz::HolovizOp>("holoviz", std::move(layers), std::move(sink));
    // Best effort: the display keeps the newest frame and never holds back the pipeline.
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    graph.add_flow(source->raw, view->raw, latest);
    graph.add_flow(processor->image, view->image, latest);
    graph.add_flow(view->view, visualizer->image_layer_input(0), latest);
    on_gpu("view", "view");
    if (offscreen) {
      auto render_check = graph.op<app::RenderCheckOp>("render_check", rendered);
      graph.add_flow(visualizer->output(), render_check->frame, lossless);
      on_gpu("holoviz", "output");
    }
  }
#endif

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) {
    return 1;
  }
  if (opts.compile_only) {
    std::cout << kAppName << ": plan=OK\n";
    return 0;
  }
  if (!signals) {
    std::cerr << kAppName << ": warning: running without managed SIGINT/SIGTERM handling: "
              << holoscan::runtime_error_message(signals.error())
              << "; an interrupt ends the process without a report\n";
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(opts.timeout_seconds) +
                        std::chrono::milliseconds(2LL * opts.count * opts.period_ms);
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  const bool timed_out = session.is_running();
  if (timed_out) {
    std::cerr << kAppName << ": timed out\n";
    session.request_stop();
  }
  session.wait();
  const holoscan::RunTermination termination = session.termination();
  const bool sdk_reported = termination == holoscan::RunTermination::kFailed &&
                            holoscan::automatic_diagnostic_logging_enabled() &&
                            holoscan::log_level() <= holoscan::LogLevel::ERROR;
  if (!session.diagnostics().empty() && !sdk_reported) {
    std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
  }
  if (signals) {
    if (auto closed = signals->close(); !closed) {
      std::cerr << kAppName << ": warning: closing the signal domain failed: "
                << holoscan::runtime_error_message(closed.error()) << '\n';
    }
  }

  // The bound every frame must beat: most of the error is at edges, where a
  // mosaic cannot know the colours, and 8-bit samples lose the darks too.
  const double min_psnr =
      opts.min_psnr.value_or(opts.bits <= 8                                  ? 24.0
                             : opts.demosaic == isp::DemosaicMethod::kMalvar ? 28.0
                                                                             : 26.0);
  const app::Report& r = *report;
  // An auto white balance must find the camera's gains, where the scene allows it.
  bool gains_ok = true;
  const bool gray_world_scene = opts.white_balance == isp::WhiteBalanceMode::kGrayWorld &&
                                opts.scene == isp::SceneKind::kGray;
  const bool white_patch_scene = opts.white_balance == isp::WhiteBalanceMode::kWhitePatch &&
                                 opts.scene == isp::SceneKind::kChart;
  if (opts.white_balance != isp::WhiteBalanceMode::kManual) {
    for (int k = 0; k < 3; ++k) {
      gains_ok = gains_ok && std::abs(r.wb_gains[static_cast<std::size_t>(k)] /
                                          camera.wb_gains[static_cast<std::size_t>(k)] -
                                      1.0F) < 0.02F;
    }
  }
  const bool estimate_checked =
      opts.white_balance == isp::WhiteBalanceMode::kManual || gray_world_scene || white_patch_scene;
  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
#if ISP_DEMO_ENABLE_VISUALIZATION
  // Offscreen, HoloViz must have shown both views; it may skip frames, never all.
  const bool shown = !offscreen || rendered->showing != 0;
#else
  const bool shown = true;
#endif
  const bool passed = !timed_out && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted &&
                      r.frames == opts.count && r.frame_gaps == 0 && r.bad_histograms == 0 &&
                      r.worst_psnr > min_psnr && (!estimate_checked || gains_ok) && shown;
  std::cout << std::fixed << std::setprecision(2) << kAppName << ": frames=" << r.frames << '/'
            << opts.count << " backend=" << isp::backend_name(isp::resolve_backend(opts.backend))
            << " demosaic=" << isp::demosaic_name(opts.demosaic)
            << " mean_psnr=" << (r.frames == 0 ? 0.0 : r.psnr_sum / r.frames)
            << " worst_psnr=" << r.worst_psnr << " (min " << min_psnr << ")" << std::setprecision(3)
            << " wb_gains=" << r.wb_gains[0] << ',' << r.wb_gains[1] << ',' << r.wb_gains[2]
            << " mean_luma=" << r.mean_luma << " ae_error_ev=" << r.ae_error_ev;
  if (r.frame_gaps != 0 || r.bad_histograms != 0) {
    std::cout << " frame_gaps=" << r.frame_gaps << " bad_histograms=" << r.bad_histograms;
  }
#if ISP_DEMO_ENABLE_VISUALIZATION
  if (offscreen) {
    std::cout << " rendered=" << rendered->frames << " showing=" << rendered->showing;
  }
#endif
  std::cout << '\n'
            << kAppName << ": verdict="
            << (passed    ? "PASS"
                : stopped ? "STOPPED"
                          : "FAIL")
            << '\n';
  return passed ? 0 : 1;
}

void validate(const Options& opts) {
  if (opts.period_ms == 0 || opts.period_ms > 1000 || opts.timeout_seconds == 0) {
    throw std::invalid_argument("--period-ms must be 1..1000 and --timeout-seconds positive");
  }
  if (opts.count == 0) {
    throw std::invalid_argument("--count must be positive");
  }
  if (opts.width > 4096 || opts.height > 4096) {
    throw std::invalid_argument("synthetic frames must be 16x16 to 4096x4096");
  }
  isp::validate_config(isp::IspOpConfig{.params = isp::IspParams{.sharpen = opts.sharpen}});
  static_cast<void>(isp::make_synthetic_scene(camera_of(opts)));
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM while this is the only thread; the SDK
  // refuses a process that already runs others.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  const Options opts = parse_args(argc, argv);
  if (opts.help) {
    usage(argv[0]);
    return 0;
  }
  validate(opts);
  return execute(opts, signals);
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
