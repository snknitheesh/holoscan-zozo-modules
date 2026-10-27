// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// AprilTag Detection: a synthetic scene source fans out to an AprilTagDetectorOp
// per backend, each feeding a reporter that checks it against ground truth; the
// first can also draw on a Holoviz display.

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
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
#if APRILTAG_DETECTION_ENABLE_VISUALIZATION
#include <holoscan/apriltag/apriltag_overlay_op.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include <holoscan/apriltag/apriltag_detector_op.hpp>
#include <holoscan/apriltag/detection.hpp>
#include <holoscan/apriltag/detector_backend.hpp>
#include <holoscan/apriltag/synthetic_scene.hpp>

#include "detection_report_op.hpp"
#include "frame_source_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace at = holoscan::apriltag;

constexpr std::string_view kAppName = "apriltag_detection";

enum class DisplayMode { kAuto, kOn, kOff };

// How long a finite run may take beyond its expected time.
constexpr auto kTimeoutAllowance = std::chrono::seconds{120};

struct Options {
  std::vector<at::Backend> backends{at::Backend::kCuda};
  at::DetectorOptions detector;
  int width{1280};
  int height{720};
  std::uint32_t count{300};
  std::uint32_t period_ms{33};
  // The scene's, for the detectors' poses.
  at::CameraIntrinsics intrinsics;
  double tag_size{};
  bool device_input{false};
  DisplayMode display{DisplayMode::kAuto};
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

std::vector<at::Backend> parse_backends(std::string_view value) {
  if (value == "all") {
    return at::available_backends();
  }
  std::vector<at::Backend> backends;
  while (!value.empty()) {
    const std::size_t comma = value.find(',');
    const std::string_view name = value.substr(0, comma);
    const auto backend = at::parse_backend(name);
    if (!backend) {
      throw std::invalid_argument("unknown backend '" + std::string(name) +
                                  "'; use cpu, cuda, vpi or all");
    }
    if (std::find(backends.begin(), backends.end(), *backend) != backends.end()) {
      throw std::invalid_argument("backend '" + std::string(name) + "' is listed twice");
    }
    backends.push_back(*backend);
    value = comma == std::string_view::npos ? std::string_view{} : value.substr(comma + 1);
  }
  if (backends.empty()) {
    throw std::invalid_argument("--backend needs at least one backend");
  }
  return backends;
}

Options parse_args(int argc, char** argv) {
  Options opts;
  using Parse = std::function<void(std::string_view value, std::string_view option)>;
  const auto number = [](auto& target) -> Parse {
    return [&target](std::string_view value, std::string_view option) {
      parse_number(value, option, target);
    };
  };
  const auto flag = [](bool& target) -> Parse {
    return [&target](std::string_view value, std::string_view option) {
      target = parse_bool(value, option);
    };
  };
  const std::pair<std::string_view, Parse> kOptions[] = {
      {"--backend",
       [&](std::string_view value, std::string_view) { opts.backends = parse_backends(value); }},
      {"--family", [&](std::string_view value, std::string_view) { opts.detector.family = value; }},
      {"--decimate", number(opts.detector.decimate)},
      {"--max-hamming", number(opts.detector.max_hamming)},
      {"--refine-edges", flag(opts.detector.refine_edges)},
      {"--sharpening", number(opts.detector.decode_sharpening)},
      {"--width", number(opts.width)},
      {"--height", number(opts.height)},
      {"--count", number(opts.count)},
      {"--period-ms", number(opts.period_ms)},
      {"--device-input", flag(opts.device_input)},
      {"--display",
       [&](std::string_view value, std::string_view option) {
         opts.display = value == "auto"             ? DisplayMode::kAuto
                        : parse_bool(value, option) ? DisplayMode::kOn
                                                    : DisplayMode::kOff;
       }},
      {"--compile-only", flag(opts.compile_only)},
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
         "Input (rendered tags with ground truth):\n"
         "  --count N                  Frames (default: 300)\n"
         "  --period-ms N              Frame period, 1..1000 ms (default: 33)\n"
         "  --width N --height N       Image size (default: 1280x720)\n"
         "  --device-input true|false  Frames in GPU memory (default: false)\n"
         "Detection:\n"
         "  --backend LIST             Comma-separated: cpu, cuda, vpi, or all compiled\n"
         "                             backends (default: cuda)\n"
         "  --family NAME              AprilTag family (default: tag36h11)\n"
         "  --max-hamming N            Bit errors to correct, 0..2 (default: 1)\n"
         "  --decimate N               Detect quads at 1/N resolution, 1..8 (default: 2)\n"
         "  --refine-edges true|false  Refine edges at full resolution (default: true)\n"
         "  --sharpening X             Bit sharpening, 0..1 (default: 0.25)\n"
         "Output:\n"
         "  --display auto|true|false  Show the first backend's detections (default: auto,\n"
         "                             on when a display is available)\n"
         "  --compile-only true|false  Build and compile the graph, then exit without running\n"
         "                             it (default: false)\n";
}

void validate(Options& opts) {
  if (opts.period_ms == 0 || opts.period_ms > 1000) {
    throw std::invalid_argument("--period-ms must be 1..1000");
  }
  if (opts.count == 0) {
    throw std::invalid_argument("--count must be positive");
  }
  const at::SyntheticScene scene = at::make_scene(opts.detector.family, 0, opts.width, opts.height);
  opts.intrinsics = scene.intrinsics;
  opts.tag_size = scene.tag_size;
  at::validate_options(opts.detector);
  for (const at::Backend backend : opts.backends) {
    at::check_available(backend);
  }
}

double percentile(std::vector<float> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
  return values[index];
}

void print_report(const apriltag_app::BackendReport& report, std::uint32_t expected_frames) {
  double mean = 0.0;
  for (const float ms : report.detect_ms) mean += ms;
  mean = report.detect_ms.empty() ? 0.0 : mean / static_cast<double>(report.detect_ms.size());
  std::cout << std::fixed << std::setprecision(2) << kAppName << ": backend=" << report.backend
            << " frames=" << report.frames << '/' << expected_frames << " tags=" << report.tags
            << " detect_ms(mean/p50/p95/max)=" << mean << '/' << percentile(report.detect_ms, 0.5)
            << '/' << percentile(report.detect_ms, 0.95) << '/'
            << percentile(report.detect_ms, 1.0);
  if (report.verified) {
    std::cout << " truth: missed=" << report.missed_tags << " extra=" << report.extra_tags
              << " misplaced=" << report.misplaced_tags << '/' << report.expected_tags
              << " max_corner_px=" << std::setprecision(3) << report.max_corner_error_px
              << " max_translation=" << std::setprecision(2) << 100.0 * report.max_translation_error
              << '%';
  }
  if (report.dropped != 0) {
    std::cout << " dropped=" << report.dropped;
  }
  if (report.truncated_frames != 0) {
    std::cout << " truncated_frames=" << report.truncated_frames;
  }
  std::cout << '\n';
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  holoscan::Graph graph{std::string(kAppName)};
  const auto period = std::chrono::milliseconds(opts.period_ms);
  const std::uint32_t frames = opts.count;
  auto source = graph.op<apriltag_app::FrameSourceOp>(
      "source", apriltag_app::FrameSourceConfig{.family = opts.detector.family,
                                                .width = opts.width,
                                                .height = opts.height,
                                                .count = frames,
                                                .period = period,
                                                .device_output = opts.device_input});

  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool display = opts.display == DisplayMode::kOn ||
                       (opts.display == DisplayMode::kAuto && display_available &&
                        APRILTAG_DETECTION_ENABLE_VISUALIZATION);
  if (display && !APRILTAG_DETECTION_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add "
        "--display false");
  }

  // Lossless edges: every detector sees every frame, in order, for the ground truth.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                             .queue_depth = 2U};
  auto remaining = std::make_shared<std::atomic<int>>(static_cast<int>(opts.backends.size()));
  std::vector<std::shared_ptr<apriltag_app::BackendReport>> reports;
  std::optional<holoscan::OperatorHandle<at::AprilTagDetectorOp>> first_detector;
  for (const at::Backend backend : opts.backends) {
    const std::string name(at::backend_name(backend));
    auto detector = graph.op<at::AprilTagDetectorOp>(
        "detector_" + name, at::AprilTagDetectorConfig{.backend = backend,
                                                       .options = opts.detector,
                                                       .intrinsics = opts.intrinsics,
                                                       .tag_size = opts.tag_size});
    auto report = std::make_shared<apriltag_app::BackendReport>();
    report->backend = name;
    reports.push_back(report);
    auto reporter = graph.op<apriltag_app::DetectionReportOp>(
        "report_" + name,
        apriltag_app::DetectionReportConfig{
            .report = report,
            .remaining = remaining,
            .expected_frames = frames,
            .truth = apriltag_app::GroundTruth{opts.detector.family, opts.width, opts.height}});
    graph.add_flow(source->image, detector->image, lossless);
    graph.add_flow(detector->detections, reporter->detections, lossless);
    if (!first_detector) {
      first_detector.emplace(detector);
    }
  }

#if APRILTAG_DETECTION_ENABLE_VISUALIZATION
  if (display) {
    namespace viz = holoscan::visualization;
    viz::LayerStack layers;
    layers.add_image_layer(
        "video", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                     .max_width = static_cast<std::uint32_t>(opts.width),
                                     .max_height = static_cast<std::uint32_t>(opts.height)});
    // Holoviz blends RGBA image layers by their alpha, so the axes go over the video.
    layers.add_image_layer(
        "axes", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8A8Unorm,
                                    .max_width = static_cast<std::uint32_t>(opts.width),
                                    .max_height = static_cast<std::uint32_t>(opts.height)});
    layers.add_text_layer(
        "labels", viz::TextLayerDesc{.max_glyphs = static_cast<std::uint32_t>(
                                         at::kMaxDetectionsPerFrame * at::kOverlayLabelBytes),
                                     .normalized_coordinates = true});
    // Draw when the axes arrive, over the newest image, which is theirs;
    // triggering on the image would draw the previous frame's axes.
    const auto max_input_age = std::max(4 * period, std::chrono::milliseconds{250});
    auto visualizer = graph.op<viz::HolovizOp>(
        "holoviz", std::move(layers),
        viz::WindowSinkDesc{.title = "AprilTag Detection (" +
                                     std::string(at::backend_name(opts.backends.front())) + ")",
                            .width = static_cast<std::uint32_t>(opts.width),
                            .height = static_cast<std::uint32_t>(opts.height)},
        viz::FrameTrigger{viz::LayerTrigger{1}}, max_input_age);
    // Best effort: the display keeps the newest sample and never holds back the detectors.
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    auto overlay = graph.op<at::AprilTagOverlayOp>(
        "overlay", at::AprilTagOverlayConfig{.width = opts.width,
                                             .height = opts.height,
                                             .intrinsics = opts.intrinsics,
                                             .tag_size = opts.tag_size});
    // The frame's image must still be queued when its detections arrive, or Holoviz, which takes
    // the newest image no newer than them, falls back to an older one. The lossless detector
    // edges keep the source at most three frames ahead of the first detector, so four images
    // hold the frame unless the overlay and Holoviz take longer than the detector's next frame.
    const holoscan::ConnectionOptions recent{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 4U};
    graph.add_flow(source->image, visualizer->image_layer_input(0), recent);
    graph.add_flow((*first_detector)->detections, overlay->detections, latest);
    graph.add_flow(overlay->axes, visualizer->image_layer_input(1), latest);
    graph.add_flow(overlay->labels, visualizer->tensor_layer_input(2), latest);
  }
#endif

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions compile_options;
  // Device-memory frames are on GPU 0.
  if (opts.device_input) {
    compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "source", .output_port = "image", .device = holoscan::DeviceId{0}});
  }
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  // Warnings on an accepted plan, as well as the reasons for a rejected one.
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
    // A thread started before main() (profilers, sanitizers) makes the SDK refuse.
    std::cerr << kAppName << ": warning: running without managed SIGINT/SIGTERM handling: "
              << holoscan::runtime_error_message(signals.error());
    if (signals.error().code == holoscan::ErrorCode::kInvalidArgument) {
      std::cerr << " (another thread was already running)";
    }
    std::cerr << "; an interrupt ends the process without a report\n";
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + kTimeoutAllowance +
                        std::chrono::milliseconds(2LL * frames * opts.period_ms);
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
  // The SDK logs a failed run's diagnostics itself unless its report or logger is
  // turned down; show any other run's, such as a stop request's warning.
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

  // A stop request (SIGINT, SIGTERM, a closed window) is neither a pass nor a
  // detection failure.
  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
  bool passed = !timed_out && session.diagnostics().empty() &&
                termination == holoscan::RunTermination::kRequestedCompleted;
  for (const auto& report : reports) {
    print_report(*report, frames);
    passed = passed && report->frames == frames && report->passed();
  }
  std::cout << kAppName << ": verdict=" << (passed ? "PASS" : stopped ? "STOPPED" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM first: the SDK refuses once other threads run.
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
