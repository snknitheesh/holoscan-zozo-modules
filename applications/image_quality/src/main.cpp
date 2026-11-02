// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Image Quality: a synthetic camera cycles through degradations (sharp, blurred,
// dark, noisy, frozen, smudged, tinted) into ImageQualityOp; a reporter checks
// every frame's flags against what its segment should give. With a display,
// HoloViz shows the image, the tile overlay and the metrics; offscreen, it
// renders them into GPU memory, where a checker looks for each.

#include <algorithm>
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

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/image_quality/image_quality_op.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "camera_op.hpp"
#include "render_check_op.hpp"
#include "report_op.hpp"
#include "segments.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace iq = holoscan::image_quality;
namespace app = image_quality_app;
namespace viz = holoscan::visualization;

constexpr std::string_view kAppName = "image_quality";

enum class DisplayMode { kAuto, kWindow, kOffscreen, kOff };

// How long a finite run may take beyond its frames' time.
constexpr auto kTimeoutAllowance = std::chrono::seconds{120};

struct Options {
  app::Schedule schedule;
  std::uint64_t count{2ULL * app::kSegmentCount * 45};
  std::uint32_t period_ms{33};
  bool device_input{false};
  DisplayMode display{DisplayMode::kAuto};
  std::string snapshot;
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
  const auto flag = [](bool& target) -> Parse {
    return [&target](std::string_view value, std::string_view option) {
      target = parse_bool(value, option);
    };
  };
  const std::pair<std::string_view, Parse> kOptions[] = {
      {"--count", number(opts.count)},
      {"--segment-frames", number(opts.schedule.segment_frames)},
      {"--obstruction-frames", number(opts.schedule.obstruction_frames)},
      {"--period-ms", number(opts.period_ms)},
      {"--width", number(opts.schedule.width)},
      {"--height", number(opts.schedule.height)},
      {"--device-input", flag(opts.device_input)},
      {"--display",
       [&](std::string_view value, std::string_view option) {
         if (value == "auto") {
           opts.display = DisplayMode::kAuto;
         } else if (value == "offscreen") {
           opts.display = DisplayMode::kOffscreen;
         } else {
           opts.display = parse_bool(value, option) ? DisplayMode::kWindow : DisplayMode::kOff;
         }
       }},
      {"--snapshot", [&](std::string_view value, std::string_view) { opts.snapshot = value; }},
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
         "Camera (a moving scene, cycling through sharp, blurred, dark, noisy, frozen,\n"
         "smudged and tinted segments):\n"
         "  --count N                    Frames (default: 630, two cycles)\n"
         "  --segment-frames N           Frames per segment (default: 45)\n"
         "  --period-ms N                Frame period, 1..1000 ms (default: 33)\n"
         "  --width N --height N         Image size (default: 640x360)\n"
         "  --device-input true|false    Frames in GPU memory (default: false)\n"
         "Checks:\n"
         "  --obstruction-frames N       Frames a tile must stay low-texture to count as\n"
         "                               obstructed; less than --segment-frames (default: 30)\n"
         "Output:\n"
         "  --display auto|true|false|offscreen\n"
         "                               Show the image, the tile overlay and the metrics in a\n"
         "                               window (auto: when a display is available), or render\n"
         "                               them offscreen and check the frames (default: auto)\n"
         "  --snapshot PATH              Offscreen: save the frame showing the most overlay as\n"
         "                               a PPM image\n"
         "  --compile-only true|false    Build and compile the graph, then exit without\n"
         "                               running it (default: false)\n";
}

void validate(const Options& opts) {
  if (opts.period_ms == 0 || opts.period_ms > 1000) {
    throw std::invalid_argument("--period-ms must be 1..1000");
  }
  if (opts.count == 0) {
    throw std::invalid_argument("--count must be positive");
  }
  if (opts.schedule.obstruction_frames == 0 ||
      opts.schedule.segment_frames <= opts.schedule.obstruction_frames) {
    throw std::invalid_argument(
        "--obstruction-frames must be positive and less than --segment-frames");
  }
  if (opts.schedule.width < 16 || opts.schedule.height < 16 || opts.schedule.width > 4096 ||
      opts.schedule.height > 4096) {
    throw std::invalid_argument("--width and --height must be 16..4096");
  }
  if (!opts.snapshot.empty() && opts.display != DisplayMode::kOffscreen) {
    throw std::invalid_argument("--snapshot needs --display offscreen");
  }
}

void print_segments(const app::RunReport& report) {
  std::cout << std::fixed;
  for (int s = 0; s < app::kSegmentCount; ++s) {
    const app::SegmentTally& t = report.segments[static_cast<std::size_t>(s)];
    if (t.frames == 0) continue;
    const double n = static_cast<double>(t.frames);
    std::cout << kAppName << ": segment=" << app::segment_name(static_cast<app::Segment>(s))
              << " frames=" << t.frames << " wrong=" << t.wrong
              << " flags=" << iq::flag_names(t.flags) << std::setprecision(3)
              << " focus=" << t.focus / n << std::setprecision(1)
              << " mean_luma=" << t.mean_luma / n << std::setprecision(2)
              << " noise_sigma=" << t.noise_sigma / n << std::setprecision(3)
              << " color_cast=" << t.color_cast / n << '\n';
  }
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  DisplayMode display = opts.display;
  if (display == DisplayMode::kAuto) {
    display = display_available ? DisplayMode::kWindow : DisplayMode::kOff;
  }
  if (display == DisplayMode::kWindow && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the window needs a display server; set DISPLAY, or use --display offscreen or false");
  }
  const bool render = display != DisplayMode::kOff;

  holoscan::Graph graph{std::string(kAppName)};
  const auto period = std::chrono::milliseconds(opts.period_ms);
  const app::Schedule& schedule = opts.schedule;
  auto camera = graph.op<app::SyntheticCameraOp>(
      "camera", app::CameraConfig{.schedule = schedule,
                                  .count = opts.count,
                                  .period = period,
                                  .device_output = opts.device_input});
  iq::ImageQualityConfig quality_config;
  quality_config.thresholds.obstruction_frames = static_cast<int>(schedule.obstruction_frames);
  quality_config.overlay_output = render;
  auto quality = graph.op<iq::ImageQualityOp>("quality", quality_config);
  auto run_report = std::make_shared<app::RunReport>();
  auto reporter = graph.op<app::QualityReportOp>(
      "report", app::QualityReportConfig{
                    .report = run_report, .schedule = schedule, .expected_frames = opts.count});

  // Lossless: every frame is checked against its segment, in order.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                             .queue_depth = 2U};
  graph.add_flow(camera->image, quality->image, lossless);
  graph.add_flow(quality->metrics, reporter->metrics, lossless);
  graph.add_flow(quality->flags, reporter->flags, lossless);

  auto render_report = std::make_shared<app::RenderReport>();
  if (render) {
    viz::LayerStack layers;
    layers.add_image_layer(
        "video", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                     .max_width = static_cast<std::uint32_t>(schedule.width),
                                     .max_height = static_cast<std::uint32_t>(schedule.height)});
    // Holoviz stretches each image layer over the frame and blends RGBA by its alpha.
    layers.add_image_layer(
        "tiles",
        viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8A8Unorm,
                            .max_width = static_cast<std::uint32_t>(quality_config.stats.tiles_x *
                                                                    quality_config.overlay_cell),
                            .max_height = static_cast<std::uint32_t>(quality_config.stats.tiles_y *
                                                                     quality_config.overlay_cell)});
    layers.add_text_layer("metrics", viz::TextLayerDesc{.max_glyphs = static_cast<std::uint32_t>(
                                                            app::kTextRows * app::kTextColumns),
                                                        .normalized_coordinates = true});
    const auto width = static_cast<std::uint32_t>(schedule.width);
    const auto height = static_cast<std::uint32_t>(schedule.height);
    viz::SinkDesc sink =
        viz::WindowSinkDesc{.title = "Image Quality", .width = width, .height = height};
    if (display == DisplayMode::kOffscreen) {
      sink = viz::CudaSinkDesc{
          .width = width, .height = height, .format = viz::ImageFormat::kR8G8B8A8Unorm};
    }
    // Draw when the tile overlay arrives, over the newest image, which is its frame's.
    const auto max_input_age = std::max(4 * period, std::chrono::milliseconds{250});
    auto visualizer =
        graph.op<viz::HolovizOp>("holoviz", std::move(layers), std::move(sink),
                                 viz::FrameTrigger{viz::LayerTrigger{1}}, max_input_age);
    auto text = graph.op<app::QualityTextOp>("text", schedule);
    // Best effort: the display keeps the newest sample and never holds back the checks.
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    // The frame's image must still be queued when its overlay arrives; the lossless
    // edges keep the camera at most a few frames ahead.
    const holoscan::ConnectionOptions recent{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 4U};
    graph.add_flow(camera->image, visualizer->image_layer_input(0), recent);
    graph.add_flow(quality->overlay, visualizer->image_layer_input(1), latest);
    graph.add_flow(quality->metrics, text->metrics, latest);
    graph.add_flow(quality->flags, text->flags, latest);
    graph.add_flow(text->text, visualizer->tensor_layer_input(2), latest);
    if (display == DisplayMode::kOffscreen) {
      auto checker = graph.op<app::RenderCheckOp>(
          "render_check",
          app::RenderCheckConfig{.report = render_report, .smudge = schedule.smudge()});
      const holoscan::ConnectionOptions frames{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                               .queue_depth = 2U};
      graph.add_flow(visualizer->output(), checker->frame, frames);
    }
  }

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions compile_options;
  if (opts.device_input) {
    compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "camera", .output_port = "image", .device = holoscan::DeviceId{0}});
  }
  if (display == DisplayMode::kOffscreen) {
    compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "holoviz", .output_port = "output", .device = holoscan::DeviceId{0}});
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
    // A thread started before main() (profilers, sanitizers) makes the SDK refuse.
    std::cerr << kAppName << ": warning: running without managed SIGINT/SIGTERM handling: "
              << holoscan::runtime_error_message(signals.error())
              << "; an interrupt ends the process without a report\n";
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  // The camera renders and degrades each frame on the CPU, which can take longer than
  // its period on large images.
  const auto pixels = static_cast<std::int64_t>(schedule.width) * schedule.height;
  const auto deadline =
      std::chrono::steady_clock::now() + kTimeoutAllowance +
      static_cast<std::int64_t>(opts.count) * (2 * period + std::chrono::nanoseconds(100 * pixels));
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

  // A stop request (SIGINT, SIGTERM, a closed window) is neither a pass nor a failure.
  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
  const app::RunReport& report = *run_report;
  print_segments(report);
  for (const std::string& mismatch : report.mismatches) {
    std::cout << kAppName << ": mismatch: " << mismatch << '\n';
  }
  std::uint64_t wrong = 0;
  for (const app::SegmentTally& t : report.segments) wrong += t.wrong;
  std::cout << kAppName << ": frames=" << report.frames << '/' << opts.count << " wrong=" << wrong
            << '\n';
  bool passed = !timed_out && session.diagnostics().empty() &&
                termination == holoscan::RunTermination::kRequestedCompleted &&
                report.frames == opts.count && wrong == 0;
  if (display == DisplayMode::kOffscreen) {
    const app::RenderReport& r = *render_report;
    std::cout << kAppName << ": rendered=" << r.frames << " with_scene=" << r.frames_with_scene
              << " with_text=" << r.frames_with_text << " with_overlay=" << r.frames_with_overlay
              << '\n';
    passed = passed && r.frames > 0 && r.frames_with_scene == r.frames && r.frames_with_text > 0 &&
             r.frames_with_overlay > 0;
    if (!opts.snapshot.empty() && !r.best.empty()) {
      app::write_ppm(opts.snapshot.c_str(), r.best, r.width, r.height);
      std::cout << kAppName << ": snapshot=" << opts.snapshot << '\n';
    }
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
