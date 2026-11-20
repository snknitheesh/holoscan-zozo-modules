// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Denoise: a synthetic camera sends noisy low-light frames, in which a disc
// moves, and three exposure brackets of a high-dynamic-range scene, all on the
// GPU. holoscan-denoise's DenoiseOp cleans the frames and its ExposureFusionOp
// fuses the brackets; a checker measures the PSNR gain, the trail behind the
// disc and what the fused image keeps of the shadow and the window, against the
// scenes' truth; and HoloViz shows noisy beside denoised, and the brackets
// beside the fused image.

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/denoise/denoise_op.hpp>
#include <holoscan/denoise/exposure_fusion_op.hpp>
#include <holoscan/denoise/synthetic_scene.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if DENOISE_DEMO_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "check_op.hpp"
#include "render_check_op.hpp"
#include "scene_source_op.hpp"
#include "view_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace app = denoise_demo;
namespace dn = holoscan::denoise;

constexpr std::string_view kAppName = "denoise";
// How long a stopped run may take to wind down before the demo gives up on it.
constexpr std::chrono::seconds kStopGrace{30};
// The trail check, per frame: the denoised frame's mean squared error behind
// the disc may be at most this share of the noisy frame's, and its mean error
// there (a ghost of the bright disc raises it) at most this many code values
// from the noisy frame's. What the filters achieve is well inside both (see the
// README); a ghost a few code values deep is not.
constexpr double kMaxTrailRatio = 1.0;
constexpr double kMaxTrailBias = 1.5;

enum class DisplayMode { kAuto, kOn, kOff, kOffscreen };

struct Options {
  std::uint32_t count{300};
  std::uint32_t period_ms{33};
  int width{1280};
  int height{720};
  dn::Backend backend{dn::Backend::kCuda};
  std::optional<dn::SpatialFilter> spatial;  // Default: bilateral, none for vpi and auto.
  bool temporal{true};
  float strength{0.5F};
  float read_noise{6.0F};
  float shot_gain{1.0F};
  float speed{6.0F};
  int cut_frame{-1};
  double min_gain{5.0};
  DisplayMode display{DisplayMode::kAuto};
  std::uint32_t timeout_seconds{120};
  bool compile_only{false};
  bool help{false};
};

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

bool parse_bool(std::string_view value, std::string_view option) {
  if (value == "true") return true;
  if (value == "false") return false;
  throw std::invalid_argument(std::string(option) + " requires true or false");
}

template <typename T>
void parse_number(std::string_view value, std::string_view option, T& result) {
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc() || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(option) + " requires a number, got '" +
                                std::string(value) + "'");
  }
}

Options parse_args(int argc, char** argv) {
  Options opts;
  using Parse = std::function<void(std::string_view value, std::string_view option)>;
  const auto number = [](auto& target) -> Parse {
    return [&target](std::string_view value, std::string_view option) {
      parse_number(value, option, target);
    };
  };
  const std::pair<std::string_view, Parse> kOptions[] = {
      {"--count", number(opts.count)},
      {"--period-ms", number(opts.period_ms)},
      {"--width", number(opts.width)},
      {"--height", number(opts.height)},
      {"--backend",
       [&](std::string_view value, std::string_view option) {
         const auto backend = dn::parse_backend(value);
         if (!backend)
           throw std::invalid_argument(std::string(option) + " must be cuda, vpi or auto");
         opts.backend = *backend;
       }},
      {"--spatial",
       [&](std::string_view value, std::string_view option) {
         opts.spatial = dn::parse_spatial(value);
         if (!opts.spatial) {
           throw std::invalid_argument(std::string(option) + " must be none, bilateral or nlm");
         }
       }},
      {"--temporal", [&](std::string_view value,
                         std::string_view option) { opts.temporal = parse_bool(value, option); }},
      {"--strength", number(opts.strength)},
      {"--read-noise", number(opts.read_noise)},
      {"--shot-gain", number(opts.shot_gain)},
      {"--speed", number(opts.speed)},
      {"--cut-frame", number(opts.cut_frame)},
      {"--min-gain", number(opts.min_gain)},
      {"--display",
       [&](std::string_view value, std::string_view option) {
         if (value == "auto") {
           opts.display = DisplayMode::kAuto;
         } else if (value == "offscreen") {
           opts.display = DisplayMode::kOffscreen;
         } else if (value == "true" || value == "false") {
           opts.display = parse_bool(value, option) ? DisplayMode::kOn : DisplayMode::kOff;
         } else {
           throw std::invalid_argument("--display must be auto, true, false or offscreen");
         }
       }},
      {"--timeout-seconds", number(opts.timeout_seconds)},
      {"--compile-only",
       [&](std::string_view value, std::string_view option) {
         opts.compile_only = parse_bool(value, option);
       }},
  };
  for (int i = 1; i < argc; ++i) {
    const std::string_view option{argv[i]};
    if (option == "--help" || option == "-h") {
      opts.help = true;
      return opts;
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(option));
    }
    const auto* entry = std::find_if(std::begin(kOptions), std::end(kOptions),
                                     [option](const auto& known) { return known.first == option; });
    if (entry == std::end(kOptions)) {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
    entry->second(argv[++i], option);
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "Input (rendered scenes with ground truth):\n"
         "  --count N                 Frames, 1..1000000 (default: 300)\n"
         "  --period-ms N             Frame period, 1..1000 ms (default: 33)\n"
         "  --width N --height N      Frame size, even, 64x64 to 3840x2160 (default: 1280x720)\n"
         "  --read-noise SIGMA        Gaussian read noise, code values (default: 6)\n"
         "  --shot-gain G             Shot noise: Poisson of value / G, times G; 0 none (default: "
         "1)\n"
         "  --speed PX                The disc's pixels per frame (default: 6)\n"
         "  --cut-frame N             Switch to a second scene at frame N (default: none)\n"
         "Denoising:\n"
         "  --backend cuda|vpi|auto   Where DenoiseOp runs (default: cuda)\n"
         "  --spatial none|bilateral|nlm\n"
         "                            Spatial filter (default: bilateral; none for vpi and auto)\n"
         "  --temporal true|false     Motion-adaptive temporal filter (default: true)\n"
         "  --strength S              0..1 (default: 0.5)\n"
         "  --min-gain DB             PSNR gain every settled frame must reach, 0..100 (default: "
         "5)\n"
         "Output:\n"
         "  --display auto|true|false|offscreen\n"
         "                            Show noisy | denoised and brackets | fused in HoloViz\n"
         "                            (default: auto, when a display is available); offscreen\n"
         "                            renders into CUDA memory and checks what was rendered\n"
         "  --timeout-seconds N       Fail when no frame is checked for N s, 1..3600 (default: "
         "120)\n"
         "  --compile-only true|false Build and compile the graph, then exit (default: false)\n";
}

void validate(Options& opts) {
  if (opts.period_ms == 0 || opts.period_ms > 1000) {
    throw std::invalid_argument("--period-ms must be 1..1000");
  }
  if (opts.timeout_seconds == 0 || opts.timeout_seconds > 3600) {
    throw std::invalid_argument("--timeout-seconds must be 1..3600");
  }
  if (opts.count == 0 || opts.count > 1000000) {
    throw std::invalid_argument("--count must be 1..1000000");
  }
  if (!(opts.min_gain >= 0.0 && opts.min_gain <= 100.0)) {
    throw std::invalid_argument("--min-gain must be 0..100 dB");
  }
  if (opts.cut_frame < -1) {
    throw std::invalid_argument("--cut-frame must be a frame number");
  }
  if (opts.width < 64 || opts.height < 64 || opts.width > 3840 || opts.height > 2160 ||
      opts.width % 2 != 0 || opts.height % 2 != 0) {
    throw std::invalid_argument("frames must be even, 64x64 to 3840x2160");
  }
  if (!(opts.read_noise >= 0.0F && opts.read_noise <= 50.0F) ||
      !(opts.shot_gain >= 0.0F && opts.shot_gain <= 16.0F) ||
      !(opts.speed >= 0.0F && opts.speed <= 64.0F)) {
    throw std::invalid_argument("noise must be 0..50, shot gain 0..16 and speed 0..64");
  }
  if (!opts.spatial) {
    opts.spatial = opts.backend == dn::Backend::kCuda ? dn::SpatialFilter::kBilateral
                                                      : dn::SpatialFilter::kNone;
  }
}

dn::DenoiseOpConfig denoise_config(const Options& opts,
                                   std::shared_ptr<dn::DenoiseStats> stats = nullptr) {
  dn::DenoiseOpConfig config;
  config.stats = std::move(stats);
  config.backend = opts.backend;
  config.params.spatial = *opts.spatial;
  config.params.temporal = opts.temporal;
  config.params.strength = opts.strength;
  config.max_width = opts.width;
  config.max_height = opts.height;
  return config;
}

// The engine DenoiseOp must run on a CI runner (HOLOSCAN_CI_PLATFORM): VPI's
// TNR on the VIC on a Jetson, for the vpi backend and for auto with settings
// it runs (temporal, no spatial filter), when the module has VPI; the cuda
// backend on x86_64 for auto. Nothing to expect elsewhere.
std::optional<std::string> expected_engine(const Options& opts) {
  const char* platform = std::getenv("HOLOSCAN_CI_PLATFORM");
  if (platform == nullptr) return std::nullopt;
  const auto built = dn::available_backends();
  const bool vpi_built = std::find(built.begin(), built.end(), dn::Backend::kVpi) != built.end();
  const bool tnr_settings = opts.temporal && opts.spatial == dn::SpatialFilter::kNone;
  if (opts.backend == dn::Backend::kCuda) return "cuda";
  if (std::string_view(platform) == "jetson" && vpi_built) {
    if (opts.backend == dn::Backend::kVpi || tnr_settings) return "vpi-vic";
    return "cuda";
  }
  if (std::string_view(platform) == "x86_64") {
    return opts.backend == dn::Backend::kVpi ? "vpi-cuda" : "cuda";
  }
  return std::nullopt;
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool display =
      opts.display == DisplayMode::kOn || (opts.display == DisplayMode::kAuto &&
                                           display_available && DENOISE_DEMO_ENABLE_VISUALIZATION);
  const bool offscreen = opts.display == DisplayMode::kOffscreen;
  if ((display || offscreen) && !DENOISE_DEMO_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add --display false");
  }

  const dn::LowLightScene scene{.width = opts.width,
                                .height = opts.height,
                                .format = dn::PixelFormat::kRgb8,
                                .speed = opts.speed,
                                .object_radius = opts.height / 12,
                                .cut_frame = opts.cut_frame};
  const dn::NoiseModel noise{.read_sigma = opts.read_noise, .shot_gain = opts.shot_gain, .seed = 1};
  const dn::HdrScene hdr{.width = opts.width, .height = opts.height, .channels = 3, .drift = 1};

  holoscan::Graph graph{std::string(kAppName)};
  auto finished = std::make_shared<std::atomic<std::uint32_t>>(0U);
  auto source = graph.op<app::SceneSourceOp>(
      "source", app::SceneSourceConfig{.scene = scene,
                                       .noise = noise,
                                       .hdr = hdr,
                                       .count = opts.count,
                                       .period = std::chrono::milliseconds(opts.period_ms),
                                       .finished = finished});
  auto stats = std::make_shared<dn::DenoiseStats>();
  auto denoiser = graph.op<dn::DenoiseOp>("denoise", denoise_config(opts, stats));
  auto fusion = graph.op<dn::ExposureFusionOp>(
      "fusion", dn::ExposureFusionOpConfig{.max_width = opts.width, .max_height = opts.height});
  auto report = std::make_shared<app::Report>();
  auto check = graph.op<app::CheckOp>("check", app::CheckConfig{.scene = scene,
                                                                .hdr = hdr,
                                                                .expected_frames = opts.count,
                                                                .report = report,
                                                                .finished = finished});

  // Lossless edges: every frame is denoised, fused and checked, in order. The
  // whole graph runs on one worker (below), and the source sends a refused
  // output again on its next tick.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->noisy, denoiser->image, lossless);
  graph.add_flow(source->bracket0, fusion->bracket0, lossless);
  graph.add_flow(source->bracket1, fusion->bracket1, lossless);
  graph.add_flow(source->bracket2, fusion->bracket2, lossless);
  graph.add_flow(source->noisy, check->noisy, lossless);
  graph.add_flow(denoiser->denoised, check->denoised, lossless);
  graph.add_flow(fusion->image, check->fused, lossless);

  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  for (const char* port : {"noisy", "bracket0", "bracket1", "bracket2"}) on_gpu("source", port);
  on_gpu("denoise", "denoised");
  on_gpu("fusion", "image");

#if DENOISE_DEMO_ENABLE_VISUALIZATION
  auto rendered = std::make_shared<app::RenderTally>();
  if (display || offscreen) {
    namespace viz = holoscan::visualization;
    auto view =
        graph.op<app::ViewOp>("view", app::ViewConfig{.width = opts.width, .height = opts.height});
    const auto view_width = static_cast<std::uint32_t>(2 * opts.width);
    const auto view_height = static_cast<std::uint32_t>(opts.height + opts.height / 2);
    viz::LayerStack layers;
    layers.add_image_layer("view", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                                       .max_width = view_width,
                                                       .max_height = view_height});
    // On screen, or offscreen into CUDA memory, where RenderCheckOp looks at each frame.
    viz::SinkDesc sink =
        viz::WindowSinkDesc{.title = "Denoise: noisy | denoised, and brackets | fused",
                            .width = view_width,
                            .height = view_height};
    if (offscreen) {
      sink = viz::CudaSinkDesc{.width = view_width, .height = view_height};
    }
    auto visualizer = graph.op<viz::HolovizOp>("holoviz", std::move(layers), std::move(sink));
    // Best effort: the display keeps the newest frame and never holds back the pipeline.
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    graph.add_flow(source->noisy, view->noisy, latest);
    graph.add_flow(denoiser->denoised, view->denoised, latest);
    graph.add_flow(source->bracket0, view->bracket0, latest);
    graph.add_flow(source->bracket1, view->bracket1, latest);
    graph.add_flow(source->bracket2, view->bracket2, latest);
    graph.add_flow(fusion->image, view->fused, latest);
    graph.add_flow(view->view, visualizer->image_layer_input(0), latest);
    on_gpu("view", "view");
    if (offscreen) {
      auto render_check = graph.op<app::RenderCheckOp>("render_check", rendered);
      // The check may skip rendered frames, so it never holds HoloViz back.
      graph.add_flow(visualizer->output(), render_check->frame, latest);
      on_gpu("holoviz", "output");
    }
  }
#endif

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  // A lossless chain stays on one worker: split across workers, a source runs
  // while its consumer's lossless queue is full and its frame is refused (see
  // the messages module's README, "Run a graph on one worker").
  graph.partition(std::string(kAppName)).include_all();
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) {
    return 1;
  }
  const std::uint32_t workers = plan.scheduler_plan().worker_count;
  if (opts.compile_only) {
    std::cout << kAppName << ": plan=OK workers=" << workers
              << " diagnostics=" << plan.diagnostics().size() << '\n';
    return workers == 1 && plan.diagnostics().empty() ? 0 : 1;
  }
  if (!signals) {
    std::cerr << kAppName << ": warning: running without managed SIGINT/SIGTERM handling: "
              << holoscan::runtime_error_message(signals.error())
              << "; an interrupt ends the process without a report\n";
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  // The run times out when no frame has been checked for --timeout-seconds,
  // however long a large, slow run takes in all.
  const auto allowance =
      std::chrono::seconds(opts.timeout_seconds) + std::chrono::milliseconds(opts.period_ms);
  auto last_progress = std::chrono::steady_clock::now();
  std::uint32_t last_finished = 0;
  bool timed_out = false;
  while (!session.wait_for(100ms)) {
    const auto now = std::chrono::steady_clock::now();
    if (const std::uint32_t done = finished->load(); done != last_finished) {
      last_finished = done;
      last_progress = now;
    } else if (now - last_progress > allowance) {
      timed_out = true;
      break;
    }
  }
  if (timed_out) {
    std::cerr << kAppName << ": timed out: no frame checked for " << opts.timeout_seconds << " s\n";
    session.request_stop();
    // Stopping is cooperative; an operator that never returns would hang here.
    if (!session.wait_for(kStopGrace)) {
      std::cerr << kAppName << ": the run did not stop within "
                << std::chrono::duration_cast<std::chrono::seconds>(kStopGrace).count()
                << " s; verdict=FAIL\n";
      std::_Exit(1);
    }
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

  const app::Report& r = *report;
  const double checked = r.checked == 0 ? 1.0 : static_cast<double>(r.checked);
  const double trail = r.trail_samples == 0 ? 0.0 : r.trail_squared / r.trail_samples;
  const double trail_input = r.trail_samples == 0 ? 0.0 : r.trail_input_squared / r.trail_samples;
  // Every settled frame gains; behind the disc, every frame is no noisier than
  // the input and shows no ghost; the fused image clips none of the window and
  // crushes none of the shadow, and keeps their texture.
  const bool denoised_ok = r.checked != 0 && r.worst_gain >= opts.min_gain;
  const bool trail_ok =
      opts.speed == 0.0F || (r.trail_samples != 0 && r.worst_trail_ratio <= kMaxTrailRatio &&
                             r.worst_trail_bias <= kMaxTrailBias);
  const bool fused_ok = r.fused_clipped == 0 && r.fused_crushed == 0 &&
                        r.worst_shadow_texture >= 6.0 && r.worst_window_texture >= 6.0;
  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
#if DENOISE_DEMO_ENABLE_VISUALIZATION
  // Offscreen, HoloViz must have shown the view; it may skip frames, never all.
  const bool shown = !offscreen || rendered->showing != 0;
#else
  const bool shown = true;
#endif
  // What DenoiseOp ran, as it reports it; on a CI runner, what it must be.
  const dn::Backend backend = stats->backend.load();
  const std::string engine = stats->engine.load();
  const std::optional<std::string> expected = expected_engine(opts);
  const bool engine_ok = !expected || engine == *expected;
  const bool passed = workers == 1 && !timed_out && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted &&
                      r.frames == opts.count && r.frame_gaps == 0 && denoised_ok && trail_ok &&
                      fused_ok && shown && engine_ok;
  std::cout << std::fixed << std::setprecision(2) << kAppName << ": frames=" << r.frames << '/'
            << opts.count << " backend=" << (opts.backend == dn::Backend::kAuto ? "auto:" : "")
            << dn::backend_name(backend) << " engine=" << (engine.empty() ? "none" : engine)
            << " spatial=" << dn::spatial_name(*opts.spatial)
            << " temporal=" << (opts.temporal ? "on" : "off") << " strength=" << opts.strength
            << " workers=" << workers << '\n'
            << kAppName << ": noisy_psnr=" << r.noisy_psnr_sum / checked
            << " denoised_psnr=" << r.denoised_psnr_sum / checked
            << " mean_gain=" << (r.denoised_psnr_sum - r.noisy_psnr_sum) / checked
            << " worst_gain=" << r.worst_gain << " (min " << opts.min_gain << ")"
            << " trail_mse=" << trail << " input_mse=" << trail_input
            << " worst_trail_ratio=" << r.worst_trail_ratio << " (max " << kMaxTrailRatio << ")"
            << " worst_trail_bias=" << r.worst_trail_bias << " (max " << kMaxTrailBias << ")"
            << '\n'
            << kAppName << ": fused_clipped=" << r.fused_clipped
            << " fused_crushed=" << r.fused_crushed << " shadow_texture=" << r.worst_shadow_texture
            << " window_texture=" << r.worst_window_texture;
  if (r.frame_gaps != 0) std::cout << " frame_gaps=" << r.frame_gaps;
#if DENOISE_DEMO_ENABLE_VISUALIZATION
  if (offscreen) {
    std::cout << " rendered=" << rendered->frames << " showing=" << rendered->showing;
  }
#endif
  if (expected) {
    std::cout << '\n'
              << kAppName << ": expected_engine=" << *expected << " ("
              << std::getenv("HOLOSCAN_CI_PLATFORM") << ")";
  }
  std::cout << '\n'
            << kAppName << ": verdict="
            << (passed    ? "PASS"
                : stopped ? "STOPPED"
                          : "FAIL")
            << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM while this is the only thread; the SDK
  // refuses a process that already runs others.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  Options opts = parse_args(argc, argv);
  if (opts.help) {
    usage(argv[0]);
    return 0;
  }
  validate(opts);
  // Settings the operators refuse are usage errors too.
  dn::validate_config(denoise_config(opts));
  return execute(opts, signals);
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
