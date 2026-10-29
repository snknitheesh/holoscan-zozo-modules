// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Stereo Camera: holoscan-stereo-camera's StereoCameraOp, from a RealSense, a
// side-by-side V4L2 camera or a recording, into a reporter that checks every
// frame, and optionally a HoloViz window with the left, right and RGB images.
// --synthetic writes a synthetic recording first and plays it, every image
// checked against the synthetic camera: a run that needs no camera and no data.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
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
#if STEREO_CAMERA_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include <holoscan/stereo_camera/stereo_camera_op.hpp>
#include <holoscan/stereo_camera/synthetic.hpp>

#include "frame_report_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace app = stereo_camera_app;
namespace sc = holoscan::stereo_camera;

constexpr std::string_view kAppName = "stereo_camera";

enum class DisplayMode { kAuto, kOn, kOff };

struct Options {
  sc::Backend backend{sc::Backend::kRealSense};
  std::optional<bool> rgb;
  std::optional<bool> depth;
  // Playback.
  std::filesystem::path recording;
  std::filesystem::path synthetic;
  int synthetic_width{320};
  int synthetic_height{240};
  bool loop{false};
  // Live cameras.
  std::string serial;
  std::optional<int> width;
  std::optional<int> height;
  int fps{30};
  int rgb_width{1280};
  int rgb_height{720};
  bool emitter{true};
  std::filesystem::path device{"/dev/video0"};
  std::filesystem::path calibration_file;
  sc::SideBySideFormat format{sc::SideBySideFormat::kYuyv};
  bool color{false};
  // The run.
  std::optional<std::uint32_t> count;
  std::uint32_t period_ms{0};
  bool device_output{false};
  bool print_calibration{false};
  DisplayMode display{DisplayMode::kAuto};
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
  if (value.empty() || error != std::errc() || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(option) + " requires a whole number");
  }
  return result;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "Source:\n"
         "  --source realsense|v4l2-sbs|playback  The backend (default: realsense)\n"
         "  --rgb true|false           Also the RGB image (default: realsense true; playback\n"
         "                             what the recording has)\n"
         "  --depth true|false         Also the depth image (default: realsense false; playback\n"
         "                             what the recording has)\n"
         "  --count N                  Frames, then the run ends (default: 300 live, the\n"
         "                             recording's frames for playback)\n"
         "Playback:\n"
         "  --recording DIR            A recording to play\n"
         "  --synthetic DIR            Write a synthetic recording to DIR and play it, checking\n"
         "                             every image against the synthetic camera; DIR must be\n"
         "                             new, empty or an earlier synthetic recording\n"
         "  --synthetic-size WxH       The synthetic camera's size (default: 320x240)\n"
         "  --loop true|false          Play the recording again until --count (default: false)\n"
         "  --period-ms N              Pace, 1..1000 ms (default: the recorded rate)\n"
         "RealSense:\n"
         "  --serial S                 Camera serial number (default: the first one)\n"
         "  --width W --height H       Infrared size (default: 1280x720)\n"
         "  --fps F                    Frame rate (default: 30)\n"
         "  --rgb-width W --rgb-height H  Color size (default: 1280x720)\n"
         "  --emitter true|false       Infrared projector (default: true)\n"
         "V4L2 side by side:\n"
         "  --device PATH              V4L2 device (default: /dev/video0)\n"
         "  --width W --height H       The whole side-by-side frame (default: 2560x720)\n"
         "  --format yuyv|gray         The camera's pixel format (default: yuyv)\n"
         "  --color true|false         The eyes in RGB rather than gray (default: false)\n"
         "  --calibration FILE         The camera's calibration file (required)\n"
         "Output:\n"
         "  --device-output true|false Images in GPU rather than host memory (default: false)\n"
         "  --display auto|true|false  Show left, right and RGB in HoloViz (default: auto,\n"
         "                             on for a live camera when a display is available)\n"
         "  --print-calibration true|false  Print the calibration file of the camera\n"
         "  --timeout-seconds N        Fail when no frame arrives for N s (default: 60)\n"
         "  --compile-only true|false  Build and check the graph without running it\n";
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string_view option = argv[i];
    if (option == "--help" || option == "-h") {
      opts.help = true;
      return opts;
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("unknown option or missing value: " + std::string(option));
    }
    const std::string_view value = argv[++i];
    if (option == "--source") {
      const auto backend = sc::parse_backend(value);
      if (!backend) throw std::invalid_argument("--source takes realsense, v4l2-sbs or playback");
      opts.backend = *backend;
    } else if (option == "--rgb") {
      opts.rgb = parse_bool(value, option);
    } else if (option == "--depth") {
      opts.depth = parse_bool(value, option);
    } else if (option == "--count") {
      opts.count = parse_number<std::uint32_t>(value, option);
    } else if (option == "--recording") {
      opts.recording = value;
    } else if (option == "--synthetic") {
      opts.synthetic = value;
    } else if (option == "--synthetic-size") {
      const std::size_t x = value.find('x');
      if (x == std::string_view::npos) throw std::invalid_argument("--synthetic-size takes WxH");
      opts.synthetic_width = parse_number<int>(value.substr(0, x), option);
      opts.synthetic_height = parse_number<int>(value.substr(x + 1), option);
    } else if (option == "--loop") {
      opts.loop = parse_bool(value, option);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--serial") {
      opts.serial = value;
    } else if (option == "--width") {
      opts.width = parse_number<int>(value, option);
    } else if (option == "--height") {
      opts.height = parse_number<int>(value, option);
    } else if (option == "--fps") {
      opts.fps = parse_number<int>(value, option);
    } else if (option == "--rgb-width") {
      opts.rgb_width = parse_number<int>(value, option);
    } else if (option == "--rgb-height") {
      opts.rgb_height = parse_number<int>(value, option);
    } else if (option == "--emitter") {
      opts.emitter = parse_bool(value, option);
    } else if (option == "--device") {
      opts.device = value;
    } else if (option == "--calibration") {
      opts.calibration_file = value;
    } else if (option == "--format") {
      const auto format = sc::parse_side_by_side_format(value);
      if (!format) throw std::invalid_argument("--format takes yuyv or gray");
      opts.format = *format;
    } else if (option == "--color") {
      opts.color = parse_bool(value, option);
    } else if (option == "--device-output") {
      opts.device_output = parse_bool(value, option);
    } else if (option == "--print-calibration") {
      opts.print_calibration = parse_bool(value, option);
    } else if (option == "--display") {
      if (value == "auto") {
        opts.display = DisplayMode::kAuto;
      } else {
        opts.display = parse_bool(value, option) ? DisplayMode::kOn : DisplayMode::kOff;
      }
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

// The operator's configuration for `opts`, after validate().
sc::StereoCameraConfig camera_config(const Options& opts) {
  sc::StereoCameraConfig config;
  sc::SourceConfig& source = config.source;
  source.backend = opts.backend;
  source.rgb = opts.rgb.value_or(false);
  source.depth = opts.depth.value_or(false);
  source.realsense = sc::RealSenseOptions{.serial = opts.serial,
                                          .width = opts.width.value_or(1280),
                                          .height = opts.height.value_or(720),
                                          .fps = opts.fps,
                                          .rgb_width = opts.rgb_width,
                                          .rgb_height = opts.rgb_height,
                                          .emitter_enabled = opts.emitter};
  source.v4l2 = sc::V4l2SideBySideOptions{
      .device = opts.device,
      .width = opts.width.value_or(2560),
      .height = opts.height.value_or(720),
      .fps = opts.fps,
      .format = opts.format,
      .output = opts.color ? sc::PixelFormat::kRgb8 : sc::PixelFormat::kMono8,
      .calibration_file = opts.calibration_file};
  source.playback = sc::PlaybackOptions{
      .directory = opts.synthetic.empty() ? opts.recording : opts.synthetic, .loop = opts.loop};
  config.count = opts.count.value_or(0);
  config.period = std::chrono::milliseconds(opts.period_ms);
  config.memory = opts.device_output ? sc::OutputMemory::kDevice : sc::OutputMemory::kHost;
  return config;
}

// Checks the options and fills in what depends on the backend; writes the
// synthetic recording. Throws std::invalid_argument for a usage error.
void validate(Options& opts) {
  if (opts.period_ms > 1000) {
    throw std::invalid_argument("--period-ms must be 0..1000");
  }
  if (opts.count && *opts.count == 0) {
    throw std::invalid_argument("--count must be positive");
  }
  if (opts.backend == sc::Backend::kPlayback) {
    if (opts.recording.empty() == opts.synthetic.empty()) {
      throw std::invalid_argument("--source playback needs --recording DIR or --synthetic DIR");
    }
    if (!opts.synthetic.empty()) {
      // Replace an earlier synthetic recording, but nothing else.
      std::error_code error;
      if (std::filesystem::exists(opts.synthetic, error) &&
          !std::filesystem::is_empty(opts.synthetic, error)) {
        bool synthetic = false;
        try {
          synthetic = sc::read_calibration(opts.synthetic / sc::kCalibrationFile).camera.name ==
                      "synthetic";
        } catch (const std::invalid_argument&) {
        }
        if (!synthetic) {
          throw std::invalid_argument("--synthetic " + opts.synthetic.string() +
                                      ": exists and is not a synthetic recording");
        }
        std::filesystem::remove_all(opts.synthetic, error);
      }
      const std::uint32_t frames = opts.count.value_or(30);
      try {
        sc::write_synthetic_recording(opts.synthetic, opts.synthetic_width, opts.synthetic_height,
                                      opts.loop ? std::min(frames, 32U) : frames,
                                      opts.rgb.value_or(true), opts.depth.value_or(true));
      } catch (const std::runtime_error& failure) {
        throw std::invalid_argument(failure.what());
      }
    }
    const sc::RecordingReader reader(opts.synthetic.empty() ? opts.recording : opts.synthetic);
    opts.rgb = opts.rgb.value_or(reader.has_rgb());
    opts.depth = opts.depth.value_or(reader.has_depth());
    if (!opts.count) {
      opts.count = static_cast<std::uint32_t>(reader.size());
    }
    if (!opts.loop && *opts.count > reader.size()) {
      throw std::invalid_argument("--count exceeds the recording's " +
                                  std::to_string(reader.size()) + " frames; add --loop true");
    }
  } else {
    if (!opts.recording.empty() || !opts.synthetic.empty() || opts.loop) {
      throw std::invalid_argument("--recording, --synthetic and --loop need --source playback");
    }
    if (opts.backend == sc::Backend::kRealSense) {
      opts.rgb = opts.rgb.value_or(true);
      opts.depth = opts.depth.value_or(false);
    }
    opts.count = opts.count.value_or(300);
  }
  sc::validate_config(camera_config(opts));

  if (opts.display == DisplayMode::kAuto) {
    opts.display = STEREO_CAMERA_ENABLE_VISUALIZATION && opts.backend != sc::Backend::kPlayback &&
                           display_available()
                       ? DisplayMode::kOn
                       : DisplayMode::kOff;
  }
  if (opts.display == DisplayMode::kOn) {
    if (!STEREO_CAMERA_ENABLE_VISUALIZATION) {
      throw std::invalid_argument("--display needs a build with visualization enabled");
    }
    if (!display_available() && !opts.compile_only) {
      throw std::invalid_argument(
          "the display needs a display server; set DISPLAY or add --display false");
    }
  }
}

#if STEREO_CAMERA_ENABLE_VISUALIZATION
// Left and right side by side, the RGB image under them: one window, a layer for
// each image, each layer in its own part of the window.
void add_display(holoscan::Graph& graph, const holoscan::OperatorHandle<sc::StereoCameraOp>& camera,
                 const sc::SourceLayout& layout, std::chrono::milliseconds period) {
  namespace viz = holoscan::visualization;
  const auto format = [](const sc::StreamLayout& stream) {
    return sc::channels(stream.format) == 3 ? viz::ImageFormat::kR8G8B8Unorm
                                            : viz::ImageFormat::kR8Unorm;
  };
  const bool rgb = layout.rgb.has_value();
  const auto width = static_cast<std::uint32_t>(layout.left.width);
  const auto height = static_cast<std::uint32_t>(layout.left.height);
  // The view's offset and scale are fractions of the window.
  const float rows = rgb ? 2.0F : 1.0F;
  viz::LayerStack layers;
  layers.add_image_layer("left", viz::ImageLayerDesc{.format = format(layout.left),
                                                     .max_width = width,
                                                     .max_height = height,
                                                     .view = viz::ViewTransform{
                                                         .scale_x = 0.5F, .scale_y = 1.0F / rows}});
  layers.add_image_layer(
      "right", viz::ImageLayerDesc{.format = format(layout.right),
                                   .max_width = width,
                                   .max_height = height,
                                   .view = viz::ViewTransform{
                                       .offset_x = 0.5F, .scale_x = 0.5F, .scale_y = 1.0F / rows}});
  if (rgb) {
    layers.add_image_layer(
        "rgb", viz::ImageLayerDesc{
                   .format = viz::ImageFormat::kR8G8B8Unorm,
                   .max_width = static_cast<std::uint32_t>(layout.rgb->width),
                   .max_height = static_cast<std::uint32_t>(layout.rgb->height),
                   .view = viz::ViewTransform{
                       .offset_x = 0.25F, .offset_y = 0.5F, .scale_x = 0.5F, .scale_y = 0.5F}});
  }
  // A frame is drawn when its last image arrives, so that the others, sent
  // before it, are there; they keep a few frames, so that HoloViz still finds
  // the one that goes with it.
  const std::size_t trigger = rgb ? 2U : 1U;
  const auto max_input_age = std::max(4 * period, std::chrono::milliseconds{250});
  auto visualizer = graph.op<viz::HolovizOp>(
      "holoviz", std::move(layers),
      viz::WindowSinkDesc{.title = "Stereo Camera",
                          .width = 2U * width,
                          .height = static_cast<std::uint32_t>(rows * static_cast<float>(height))},
      viz::FrameTrigger{viz::LayerTrigger{trigger}}, max_input_age);
  const holoscan::ConnectionOptions newest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                           .queue_depth = 1U};
  const holoscan::ConnectionOptions recent{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                           .queue_depth = 4U};
  graph.add_flow(camera->left, visualizer->image_layer_input(0), recent);
  graph.add_flow(camera->right, visualizer->image_layer_input(1), rgb ? recent : newest);
  if (rgb) {
    graph.add_flow(camera->rgb, visualizer->image_layer_input(2), newest);
  }
}
#endif

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

void print_summary(const app::FrameReport& report, std::uint32_t frames, bool print_calibration) {
  std::cout << std::fixed << std::setprecision(3) << kAppName << ": frames=" << report.frames.load()
            << '/' << frames;
  if (report.calibration) {
    const sc::StereoCalibration& calibration = *report.calibration;
    std::cout << " source=" << report.backend << " camera=\"" << calibration.camera.name << "\"";
    if (!calibration.camera.serial.empty()) std::cout << " serial=" << calibration.camera.serial;
    if (!calibration.camera.firmware.empty()) {
      std::cout << " firmware=" << calibration.camera.firmware;
    }
    std::cout << " eyes=" << report.eyes << " baseline_mm=" << 1000.0 * calibration.baseline()
              << " fx=" << calibration.left.fx << " rectified=" << calibration.rectified;
    if (!report.rgb.empty()) std::cout << " rgb=" << report.rgb;
    if (!report.depth.empty()) {
      std::cout << " depth=" << report.depth
                << " depth_scale_mm=" << 1000.0 * calibration.depth->scale << " valid_depth="
                << (report.depth_pixels == 0 ? 0.0
                                             : 100.0 * report.valid_depth / report.depth_pixels)
                << "%";
    }
  }
  std::cout << " wrong_sizes=" << report.wrong_sizes << " not_increasing=" << report.not_increasing
            << " frame_gaps=" << report.frame_gaps
            << " calibration_changes=" << report.calibration_changes;
  if (report.verified != 0 || report.wrong_images != 0) {
    std::cout << " truth: verified=" << report.verified << " wrong_images=" << report.wrong_images;
  }
  std::cout << '\n';
  if (print_calibration && report.calibration) {
    std::cout << sc::format_calibration(*report.calibration);
  }
}

int execute(const Options& opts, SignalDomain& signals) {
  holoscan::Graph graph{std::string(kAppName)};
  const std::uint32_t frames = *opts.count;
  const bool realtime = opts.backend != sc::Backend::kPlayback;
  auto report = std::make_shared<app::FrameReport>();
  auto stats = std::make_shared<sc::StereoCameraStats>();
  sc::StereoCameraConfig config = camera_config(opts);
  config.stats = stats;
  auto camera = graph.op<sc::StereoCameraOp>("camera", config);
  const sc::SourceLayout layout = camera->layout();
  auto reporter = graph.op<app::FrameReportOp>(
      "report", app::FrameReportConfig{.report = report,
                                       .rgb = layout.rgb.has_value(),
                                       .depth = layout.depth.has_value(),
                                       .expected_frames = frames,
                                       .realtime = realtime,
                                       .synthetic = !opts.synthetic.empty()});
  // A live camera keeps only its newest frame on every edge: the reporter takes
  // the newest when it is done, and never holds the camera back. A recording
  // plays losslessly: every frame, in order.
  const holoscan::ConnectionOptions edge =
      realtime ? holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U}
               : holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(camera->left, reporter->left, edge);
  graph.add_flow(camera->right, reporter->right, edge);
  if (layout.rgb) graph.add_flow(camera->rgb, reporter->rgb, edge);
  if (layout.depth) graph.add_flow(camera->depth, reporter->depth, edge);
  graph.add_flow(camera->calibration, reporter->calibration, edge);
#if STEREO_CAMERA_ENABLE_VISUALIZATION
  if (opts.display == DisplayMode::kOn) {
    const auto period = opts.period_ms != 0
                            ? std::chrono::milliseconds(opts.period_ms)
                            : std::chrono::milliseconds(1000 / std::max(opts.fps, 1));
    add_display(graph, camera, layout, period);
  }
#endif
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions compile_options;
  if (opts.device_output) {
    std::vector<std::string> ports{"left", "right"};
    if (layout.rgb) ports.emplace_back("rgb");
    if (layout.depth) ports.emplace_back("depth");
    for (const std::string& port : ports) {
      compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = "camera", .output_port = port, .device = holoscan::DeviceId{0}});
    }
  }
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
    // Profilers, sanitizers and preloaded libraries can start a thread before
    // main(); the SDK then refuses to take over SIGINT and SIGTERM.
    std::cerr << kAppName << ": warning: running without managed SIGINT/SIGTERM handling: "
              << holoscan::runtime_error_message(signals.error())
              << "; an interrupt ends the process without a report\n";
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  // The run times out when no frame arrives for this long.
  const auto patience = std::chrono::seconds(opts.timeout_seconds);
  std::uint32_t seen = 0;
  auto last_progress = std::chrono::steady_clock::now();
  bool timed_out = false;
  while (session.is_running()) {
    std::this_thread::sleep_for(10ms);
    const auto now = std::chrono::steady_clock::now();
    if (const std::uint32_t reported = report->frames.load(); reported != seen) {
      seen = reported;
      last_progress = now;
    } else if (now - last_progress >= patience) {
      timed_out = true;
      break;
    }
  }
  if (timed_out) {
    std::cerr << kAppName << ": timed out: no frame for " << opts.timeout_seconds << " s\n";
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
  // A stop request (SIGINT, SIGTERM, a closed window) ends the run early: not a
  // failure, but not a pass either.
  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
  // In real time, frames the reporter was too busy for are skipped; the last one must arrive.
  const bool all_frames =
      realtime ? report->last_frame_id.load() >= frames : report->frames.load() == frames;
  const bool passed = !timed_out && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted && all_frames &&
                      report->passed();
  print_summary(*report, frames, opts.print_calibration);
  if (realtime) {
    std::cout << kAppName << ": camera frames sent=" << stats->sent.load()
              << " dropped=" << stats->dropped.load() << '\n';
  }
  std::cout << kAppName << ": verdict=" << (passed ? "PASS" : stopped ? "STOPPED" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM while this is the only thread; the SDK
  // refuses a process that already runs others.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  auto opts = parse_args(argc, argv);
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
