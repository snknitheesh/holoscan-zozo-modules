// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Color Convert: a synthetic camera sends NV12 or YUYV frames of moving
// colour bars over gradients; ColorConvertOp turns each into RGBA8 at full
// size, and ResizeOp letterboxes it into a small RGBA8 image in the same pass.
// HoloViz shows the full frame with the small one inset, in a window or
// rendered offscreen. Headless and offscreen runs check every output, and
// every render, against the module's CPU reference.

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/color_convert/color_convert_op.hpp>
#include <holoscan/color_convert/converter.hpp>
#include <holoscan/color_convert/resize_op.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if COLOR_CONVERT_APP_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "camera_op.hpp"
#include "check_ops.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace app = color_convert_app;
namespace cc = holoscan::color_convert;

constexpr std::string_view kAppName = "color_convert";

enum class Display { kAuto, kWindow, kOffscreen, kNone };

struct Options {
  cc::PixelFormat camera{cc::PixelFormat::kNv12};
  int width{1920};
  int height{1080};
  int resize_width{640};
  int resize_height{640};
  cc::ResizeMode mode{cc::ResizeMode::kBilinear};
  bool letterbox{true};
  cc::Backend backend{cc::Backend::kCuda};
  std::optional<std::uint32_t> count;
  std::uint32_t period_ms{33};
  Display display{Display::kAuto};
  std::filesystem::path image_dir;
  std::uint32_t timeout_seconds{60};
  bool compile_only{false};
  bool help{false};
};

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

bool display_available() {
  return env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
}

bool parse_bool(std::string_view value, std::string_view option) {
  if (value == "true") return true;
  if (value == "false") return false;
  throw std::invalid_argument(std::string(option) + " requires true or false");
}

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
    if (option == "--camera") {
      const auto format = cc::parse_format(value);
      if (format != cc::PixelFormat::kNv12 && format != cc::PixelFormat::kYuyv) {
        throw std::invalid_argument("--camera must be nv12 or yuyv");
      }
      opts.camera = *format;
    } else if (option == "--width") {
      opts.width = parse_number<int>(value, option);
    } else if (option == "--height") {
      opts.height = parse_number<int>(value, option);
    } else if (option == "--resize-width") {
      opts.resize_width = parse_number<int>(value, option);
    } else if (option == "--resize-height") {
      opts.resize_height = parse_number<int>(value, option);
    } else if (option == "--mode") {
      const auto mode = cc::parse_resize_mode(value);
      if (!mode) throw std::invalid_argument("--mode must be nearest, bilinear or area");
      opts.mode = *mode;
    } else if (option == "--letterbox") {
      opts.letterbox = parse_bool(value, option);
    } else if (option == "--backend") {
      const auto backend = cc::parse_backend(value);
      if (!backend) throw std::invalid_argument("--backend must be cuda, vic or auto");
      opts.backend = *backend;
    } else if (option == "--count") {
      opts.count = parse_number<std::uint32_t>(value, option);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--display") {
      if (value == "auto") {
        opts.display = Display::kAuto;
      } else if (value == "window") {
        opts.display = Display::kWindow;
      } else if (value == "offscreen") {
        opts.display = Display::kOffscreen;
      } else if (value == "none") {
        opts.display = Display::kNone;
      } else {
        throw std::invalid_argument("--display must be auto, window, offscreen or none");
      }
    } else if (option == "--image-dir") {
      opts.image_dir = value;
    } else if (option == "--timeout-seconds") {
      opts.timeout_seconds = parse_number<std::uint32_t>(value, option);
    } else if (option == "--compile-only") {
      opts.compile_only = parse_bool(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "Camera:\n"
         "  --camera nv12|yuyv         The synthetic camera's format, BT.709 limited range\n"
         "                             (default: nv12)\n"
         "  --width N --height N       Its frame size, even (default: 1920x1080)\n"
         "  --count N                  Frames, then stop (default: 60; with a window, until it\n"
         "                             is closed)\n"
         "  --period-ms N              Its period, 1..1000 ms (default: 33)\n"
         "Conversion:\n"
         "  --backend cuda|vic|auto    Where both operators run (default: cuda)\n"
         "  --resize-width N --resize-height N\n"
         "                             ResizeOp's output size (default: 640x640)\n"
         "  --mode nearest|bilinear|area\n"
         "                             ResizeOp's sampling (default: bilinear)\n"
         "  --letterbox true|false     Keep the aspect ratio (default: true)\n"
         "Display:\n"
         "  --display auto|window|offscreen|none\n"
         "                             A HoloViz window (live: the newest frame only), HoloViz\n"
         "                             rendering offscreen, every render checked, or no display;\n"
         "                             auto is window with a display, else none\n"
         "  --image-dir DIR            With offscreen, also write each render as PPM there\n"
         "  --timeout-seconds N        Fail when no frame finishes for N seconds (default: 60)\n"
         "  --compile-only true|false  Build and check the graph without running it\n";
}

void validate(Options& opts) {
  if (opts.display == Display::kAuto) {
    opts.display = display_available() ? Display::kWindow : Display::kNone;
  }
#if !COLOR_CONVERT_APP_ENABLE_VISUALIZATION
  if (opts.display != Display::kNone) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
#endif
  if (opts.display == Display::kWindow && !display_available() && !opts.compile_only) {
    throw std::invalid_argument(
        "the window needs a display server; set DISPLAY, or use --display offscreen or none");
  }
  if (opts.display == Display::kWindow) {
    opts.count = opts.count.value_or(0);
  } else {
    opts.count = opts.count.value_or(60);
    if (*opts.count == 0) {
      throw std::invalid_argument("--count must be positive without a window");
    }
  }
  if (opts.period_ms == 0 || opts.period_ms > 1000 || opts.timeout_seconds == 0) {
    throw std::invalid_argument("--period-ms must be 1..1000 and --timeout-seconds positive");
  }
  if (!opts.image_dir.empty()) {
    if (opts.display != Display::kOffscreen) {
      throw std::invalid_argument("--image-dir writes offscreen renders");
    }
    std::error_code error;
    if (!std::filesystem::is_directory(opts.image_dir, error)) {
      throw std::invalid_argument("--image-dir is not a directory: " + opts.image_dir.string());
    }
  }
  app::validate_config(
      app::SyntheticCameraConfig{.format = opts.camera,
                                 .width = opts.width,
                                 .height = opts.height,
                                 .period = std::chrono::milliseconds(opts.period_ms)});
  cc::validate_config(cc::ResizeConfig{.width = opts.resize_width,
                                       .height = opts.resize_height,
                                       .output_format = cc::PixelFormat::kRgba8,
                                       .backend = opts.backend});
  cc::check_available(opts.backend);
}

app::Pipeline pipeline(const Options& opts) {
  return app::Pipeline{
      .camera = opts.camera,
      .width = opts.width,
      .height = opts.height,
      .converted = cc::PixelFormat::kRgba8,
      .resized = cc::PixelFormat::kRgba8,
      .resized_width = opts.resize_width,
      .resized_height = opts.resize_height,
      .resize_params = cc::ConversionParams{.matrix = app::kCameraMatrix,
                                            .range = app::kCameraRange,
                                            .mode = opts.mode,
                                            .letterbox = opts.letterbox,
                                            .fill_rgb = {48, 48, 48}},
  };
}

// The inset's place in a render of the camera's size: the right top corner,
// its side 30% of the width, square in pixels.
struct Inset {
  int x;
  int y;
  int width;
  int height;
};

Inset inset_of(const Options& opts) {
  const int side = opts.width * 3 / 10;
  const int height = side * opts.resize_height / opts.resize_width;
  const int margin = opts.height / 24;
  return Inset{opts.width - side - margin, margin, side, height};
}

// On a CI runner, auto must take the VIC on the Jetson and CUDA elsewhere.
std::optional<cc::Backend> expected_auto() {
  const char* platform = std::getenv("HOLOSCAN_CI_PLATFORM");
  if (platform == nullptr) return std::nullopt;
  return std::string_view(platform) == "jetson" ? cc::Backend::kVic : cc::Backend::kCuda;
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  holoscan::Graph graph{std::string(kAppName)};
  const app::Pipeline pipe = pipeline(opts);
  auto report = std::make_shared<app::Report>();
  const bool live = opts.display == Display::kWindow;
  const bool offscreen = opts.display == Display::kOffscreen;

  // Lossless (no window): one frame in flight, sent once its outputs (and
  // its render) are checked; every frame is converted, checked and rendered.
  // Live (a window): every edge keeps only the newest frame.
  app::SyntheticCameraConfig camera_config{.format = opts.camera,
                                           .width = opts.width,
                                           .height = opts.height,
                                           .count = *opts.count,
                                           .period = std::chrono::milliseconds(opts.period_ms)};
  if (!live) {
    camera_config.done = [report] { return report->frames.load(); };
  }
  auto camera = graph.op<app::SyntheticCameraOp>("camera", std::move(camera_config));
  auto convert = graph.op<cc::ColorConvertOp>(
      "convert", cc::ColorConvertConfig{.output_format = pipe.converted,
                                        .matrix = app::kCameraMatrix,
                                        .range = app::kCameraRange,
                                        .backend = opts.backend,
                                        .max_width = opts.width,
                                        .max_height = opts.height,
                                        .on_backend = [report](cc::Backend backend) {
                                          report->convert_backend = static_cast<int>(backend);
                                        }});
  auto resize = graph.op<cc::ResizeOp>(
      "resize", cc::ResizeConfig{.width = opts.resize_width,
                                 .height = opts.resize_height,
                                 .mode = opts.mode,
                                 .letterbox = opts.letterbox,
                                 .fill_rgb = pipe.resize_params.fill_rgb,
                                 .output_format = pipe.resized,
                                 .matrix = app::kCameraMatrix,
                                 .range = app::kCameraRange,
                                 .backend = opts.backend,
                                 .on_backend = [report](cc::Backend backend) {
                                   report->resize_backend = static_cast<int>(backend);
                                 }});

  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  const holoscan::ConnectionOptions newest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                           .queue_depth = 1U};
  const holoscan::ConnectionOptions& edge = live ? newest : lossless;
  graph.add_flow(camera->image, convert->image, edge);
  graph.add_flow(camera->image, resize->image, edge);
  if (!live) {
    // The cuda backend matches the reference byte for byte; the VIC within
    // its tolerance (the module's VIC parity test).
    const app::Tolerance tolerance{.exact = opts.backend == cc::Backend::kCuda};
    auto check = graph.op<app::FrameCheckOp>(
        "check",
        app::FrameCheckConfig{
            .report = report, .pipeline = pipe, .converted = tolerance, .resized = tolerance});
    graph.add_flow(convert->converted, check->converted, lossless);
    graph.add_flow(resize->resized, check->resized, lossless);
  }

  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  on_gpu("camera", "image");
  on_gpu("convert", "converted");
  on_gpu("resize", "resized");

#if COLOR_CONVERT_APP_ENABLE_VISUALIZATION
  if (opts.display != Display::kNone) {
    namespace viz = holoscan::visualization;
    const Inset inset = inset_of(opts);
    viz::LayerStack layers;
    layers.add_image_layer(
        "frame", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8A8Unorm,
                                     .max_width = static_cast<std::uint32_t>(opts.width),
                                     .max_height = static_cast<std::uint32_t>(opts.height)});
    layers.add_image_layer(
        "inset",
        viz::ImageLayerDesc{
            .format = viz::ImageFormat::kR8G8B8A8Unorm,
            .max_width = static_cast<std::uint32_t>(opts.resize_width),
            .max_height = static_cast<std::uint32_t>(opts.resize_height),
            .view = viz::ViewTransform{.offset_x = static_cast<float>(inset.x) / opts.width,
                                       .offset_y = static_cast<float>(inset.y) / opts.height,
                                       .scale_x = static_cast<float>(inset.width) / opts.width,
                                       .scale_y = static_cast<float>(inset.height) / opts.height}});
    viz::SinkDesc sink =
        viz::WindowSinkDesc{.title = "Color Convert", .width = 1280, .height = 720};
    if (offscreen) {
      sink = viz::CudaSinkDesc{.width = static_cast<std::uint32_t>(opts.width),
                               .height = static_cast<std::uint32_t>(opts.height),
                               .format = viz::ImageFormat::kR8G8B8A8Unorm};
    }
    // Each full frame draws a render, over the newest inset. HoloViz takes the
    // newest of each, so it never holds back the conversions; the render check
    // finds which frame a render shows from the frame itself.
    auto visualizer = graph.op<viz::HolovizOp>("holoviz", std::move(layers), std::move(sink),
                                               viz::FrameTrigger{viz::LayerTrigger{0}},
                                               std::chrono::milliseconds(1000));
    graph.add_flow(convert->converted, visualizer->image_layer_input(0), newest);
    graph.add_flow(resize->resized, visualizer->image_layer_input(1), newest);
    if (offscreen) {
      auto render_check = graph.op<app::RenderCheckOp>(
          "render_check", app::RenderCheckConfig{.report = report,
                                                 .pipeline = pipe,
                                                 .inset_x = inset.x,
                                                 .inset_y = inset.y,
                                                 .inset_width = inset.width,
                                                 .inset_height = inset.height,
                                                 .image_dir = opts.image_dir});
      graph.add_flow(visualizer->output(), render_check->rendered, newest);
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
              << holoscan::runtime_error_message(signals.error()) << '\n';
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  const std::uint32_t frames = *opts.count;
  const auto finished = [&] { return report->frames.load(); };
  std::uint32_t seen = 0;
  auto last_progress = std::chrono::steady_clock::now();
  bool timed_out = false;
  while (session.is_running()) {
    std::this_thread::sleep_for(10ms);
    const auto now = std::chrono::steady_clock::now();
    if (!live && finished() >= frames) {
      // HoloViz renders the last frames a little later: wait until the
      // renders stop coming, for at most two seconds.
      if (offscreen) {
        std::uint32_t renders = report->rendered.load();
        for (int i = 0; i < 20; ++i) {
          std::this_thread::sleep_for(100ms);
          const std::uint32_t now_rendered = report->rendered.load();
          if (now_rendered == renders && renders != 0) break;
          renders = now_rendered;
        }
      }
      session.request_stop();
      break;
    }
    if (live) continue;
    if (const std::uint32_t done = finished(); done != seen) {
      seen = done;
      last_progress = now;
    } else if (now - last_progress >= std::chrono::seconds(opts.timeout_seconds)) {
      timed_out = true;
      std::cerr << kAppName << ": timed out: no frame finished for " << opts.timeout_seconds
                << " s\n";
      session.request_stop();
      break;
    }
  }
  session.wait();
  const holoscan::RunTermination termination = session.termination();
  const bool sdk_reported = termination == holoscan::RunTermination::kFailed &&
                            holoscan::automatic_diagnostic_logging_enabled() &&
                            holoscan::log_level() <= holoscan::LogLevel::ERROR;
  // A run this program stopped once every frame was checked leaves a warning
  // about that request, which is no news.
  const bool all_checked = !live && finished() >= frames;
  if (!session.diagnostics().empty() && !sdk_reported && !all_checked) {
    std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
  }
  if (signals) {
    if (auto closed = signals->close(); !closed) {
      std::cerr << kAppName << ": warning: closing the signal domain failed: "
                << holoscan::runtime_error_message(closed.error()) << '\n';
    }
  }

  const auto ran = [](int backend) {
    return backend < 0 ? std::string_view("none")
                       : cc::backend_name(static_cast<cc::Backend>(backend));
  };
  const int convert_backend = report->convert_backend.load();
  const int resize_backend = report->resize_backend.load();
  std::cout << kAppName << ": camera=" << cc::format_name(opts.camera) << ' ' << opts.width << 'x'
            << opts.height << " backend=" << cc::backend_name(opts.backend) << " (convert ran "
            << ran(convert_backend) << ", resize ran " << ran(resize_backend) << ")\n";
  bool auto_ok = true;
  if (opts.backend == cc::Backend::kAuto) {
    if (const auto expected = expected_auto()) {
      auto_ok = convert_backend == static_cast<int>(*expected) &&
                resize_backend == static_cast<int>(*expected);
      std::cout << kAppName << ": auto=" << ran(convert_backend) << ',' << ran(resize_backend)
                << " expected=" << cc::backend_name(*expected) << " ("
                << std::getenv("HOLOSCAN_CI_PLATFORM") << ")\n";
    }
  }
  if (live) {
    const bool stopped = termination == holoscan::RunTermination::kStopped ||
                         termination == holoscan::RunTermination::kRequestedCompleted;
    std::cout << kAppName << ": verdict=" << (stopped ? "STOPPED" : "FAIL") << '\n';
    return stopped ? 0 : 1;
  }
  bool passed =
      !timed_out && all_checked && auto_ok && termination != holoscan::RunTermination::kFailed;
  {
    const std::lock_guard lock(report->mutex);
    passed = passed && report->failed_frames == 0 && report->failed_renders == 0;
    // Offscreen, HoloViz must have rendered, with the inset at least once.
    passed = passed && (!offscreen || (report->rendered.load() != 0 && report->insets_seen != 0));
    std::cout << std::fixed << std::setprecision(3) << kAppName
              << ": frames=" << report->frames.load() << '/' << frames
              << " converted: mean=" << report->converted.mean << " p99=" << report->converted.p99
              << " max=" << report->converted.max << " differing=" << report->converted.wrong
              << " resized: mean=" << report->resized.mean << " p99=" << report->resized.p99
              << " max=" << report->resized.max << " differing=" << report->resized.wrong;
    if (offscreen) {
      std::cout << " renders=" << report->rendered.load() << " insets=" << report->insets_seen
                << " render: mean=" << report->render.mean << " max=" << report->render.max;
    }
    std::cout << '\n';
    if (!report->first_failure.empty()) {
      std::cout << kAppName << ": first failure: " << report->first_failure << '\n';
    }
  }
  std::cout << kAppName << ": verdict=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  // Before any other thread, so that the SDK can own SIGINT and SIGTERM.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  Options opts;
  try {
    opts = parse_args(argc, argv);
    if (opts.help) {
      usage(argv[0]);
      return 0;
    }
    validate(opts);
  } catch (const std::invalid_argument& error) {
    std::cerr << "error: " << error.what() << "\nSee " << argv[0] << " --help\n";
    return 2;
  }
  try {
    return execute(opts, signals);
  } catch (const std::exception& error) {
    std::cerr << kAppName << ": error: " << error.what() << '\n';
    return 1;
  }
}
