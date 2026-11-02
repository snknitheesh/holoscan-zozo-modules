// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Foundation Stereo: a stereo source (the module's synthetic scene with known
// disparity, or a recorded camera clip) feeds FoundationStereoOp, whose
// disparity a reporter checks against the scene's truth, the camera's own
// depth, or, with the stub model, the CPU reference of the operator's chain;
// HoloViz can show the left image beside the colorized disparity.

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
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
#include <holoscan/foundation_stereo/foundation_stereo_op.hpp>
#include <holoscan/foundation_stereo/stub_model.hpp>
#include <holoscan/foundation_stereo/synthetic_scene.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if FOUNDATION_STEREO_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "disparity_color_op.hpp"
#include "disparity_report_op.hpp"
#include "recording.hpp"
#include "render_check_op.hpp"
#include "stereo_sources.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace fs = holoscan::foundation_stereo;
namespace app = foundation_stereo_app;

constexpr std::string_view kAppName = "foundation_stereo";

enum class Source { kSynthetic, kRecording };
enum class DisplayMode { kAuto, kOn, kOff, kOffscreen };

// The stub model's input size: not a synthetic image's, so both resizes show.
constexpr int kStubWidth = 320;
constexpr int kStubHeight = 192;

// How long a finite run may take beyond its expected time; the first run of a
// real model also builds its TensorRT engine before the graph starts.
constexpr auto kTimeoutAllowance = std::chrono::seconds{120};

// What the real model must reach on the synthetic scene, over the pixels both
// cameras see: the share with a disparity, the share of those within 1 px of
// the truth, and their median error. Measured with the pinned
// C-Fast-FoundationStereo model (FP16) on an RTX 5090 and Jetson Orin: valid
// 1.0, within 1 px 0.99, median 0.02 px; set with margin.
constexpr double kMinTruthValid = 0.95;
constexpr double kMinWithin1px = 0.95;
constexpr double kMaxMedianErrorPx = 0.25;

// Against the camera's own depth, on a clip where the camera has depth for at
// least --min-camera-coverage of the pixels: the computed depth must cover
// this share of the camera's pixels, with this median relative error and this
// share within 5 %. The camera's depth is itself semi-global matching of the
// same infrared pair, not ground truth; see the README for the measurements.
constexpr double kMinCoverage = 0.85;
constexpr double kMaxMedianError = 0.03;
constexpr double kMinWithin5pct = 0.60;

struct Options {
  Source source{Source::kSynthetic};
  // An .onnx model or a TensorRT engine; "stub" for the stub model.
  std::string model;
  bool fp16{true};
  bool geometry_input{false};
  bool invalidate_out_of_view{true};
  // Synthetic.
  int width{1280};
  int height{720};
  bool color{true};
  bool device_input{true};
  // Recording.
  std::optional<std::filesystem::path> recording;
  std::string clip;
  bool loop{false};
  double min_camera_coverage{0.2};
  std::optional<std::uint32_t> count;
  std::uint32_t period_ms{33};
  // Drop-oldest edges of depth 1, as for a live camera, instead of lossless ones.
  bool realtime{false};
  DisplayMode display{DisplayMode::kAuto};
  // With the offscreen display: write the newest rendered frame here.
  std::filesystem::path snapshot;
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
      {"--source",
       [&](std::string_view value, std::string_view) {
         if (value == "synthetic") {
           opts.source = Source::kSynthetic;
         } else if (value == "recording") {
           opts.source = Source::kRecording;
         } else {
           throw std::invalid_argument("--source must be synthetic or recording");
         }
       }},
      {"--model", [&](std::string_view value, std::string_view) { opts.model = value; }},
      {"--fp16", flag(opts.fp16)},
      {"--geometry-input", flag(opts.geometry_input)},
      {"--invalidate-out-of-view", flag(opts.invalidate_out_of_view)},
      {"--width", number(opts.width)},
      {"--height", number(opts.height)},
      {"--color", flag(opts.color)},
      {"--device-input", flag(opts.device_input)},
      {"--recording",
       [&](std::string_view value, std::string_view) { opts.recording = std::string(value); }},
      {"--clip", [&](std::string_view value, std::string_view) { opts.clip = value; }},
      {"--loop", flag(opts.loop)},
      {"--min-camera-coverage", number(opts.min_camera_coverage)},
      {"--count",
       [&](std::string_view value, std::string_view option) {
         std::uint32_t count = 0;
         parse_number(value, option, count);
         opts.count = count;
       }},
      {"--period-ms", number(opts.period_ms)},
      {"--realtime", flag(opts.realtime)},
      {"--display",
       [&](std::string_view value, std::string_view option) {
         opts.display = value == "auto"             ? DisplayMode::kAuto
                        : value == "offscreen"      ? DisplayMode::kOffscreen
                        : parse_bool(value, option) ? DisplayMode::kOn
                                                    : DisplayMode::kOff;
       }},
      {"--snapshot",
       [&](std::string_view value, std::string_view) { opts.snapshot = std::string(value); }},
      {"--compile-only", flag(opts.compile_only)},
  };
  for (int i = 1; i < argc; ++i) {
    const std::string_view option{argv[i]};
    if (option == "--help" || option == "-h") {
      opts.help = true;
      return opts;
    }
    const auto* entry = std::find_if(std::begin(kOptions), std::end(kOptions),
                                     [option](const auto& known) { return known.first == option; });
    if (entry == std::end(kOptions)) {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(option));
    }
    entry->second(argv[++i], option);
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " --model MODEL [options]\n"
         "Model:\n"
         "  --model PATH|stub          A Fast-FoundationStereo .onnx model (its TensorRT engine\n"
         "                             is built and cached beside it) or engine, or stub for\n"
         "                             the module's stub model, written to the working\n"
         "                             directory and checked against its CPU reference\n"
         "  --fp16 true|false          Build the engine with FP16 (default: true)\n"
         "  --invalidate-out-of-view true|false  Zero pixels matched left of the right\n"
         "                             image (default: true)\n"
         "Input:\n"
         "  --source synthetic|recording  Rendered planes with known disparity, or a\n"
         "                             recorded stereo clip (default: synthetic)\n"
         "  --count N                  Frames (default: 300 synthetic, every frame of a\n"
         "                             recording)\n"
         "  --period-ms N              Frame period, 1..1000 ms (default: 33)\n"
         "  --width N --height N       Synthetic image size (default: 1280x720)\n"
         "  --color true|false         Synthetic RGB8, or MONO8 (default: true)\n"
         "  --device-input true|false  Synthetic frames in GPU memory (default: true)\n"
         "  --recording DIR            A recording's folder (calibration.json, index.csv, PGM)\n"
         "  --clip NAME                Or a clip in $HOLOSCAN_TEST_DATA/stereo-camera/\n"
         "  --loop true|false          Start the recording over until --count frames\n"
         "  --geometry-input true|false  Send the geometry with every frame instead of\n"
         "                             configuring it (default: false)\n"
         "  --realtime true|false      Drop-oldest edges of depth 1, as for a live camera:\n"
         "                             frames the model cannot keep up with are dropped and\n"
         "                             counted (default: false, lossless)\n"
         "Checks and output:\n"
         "  --min-camera-coverage X    Check accuracy against a recording's depth only when\n"
         "                             the camera has depth for this share of pixels\n"
         "                             (default: 0.2)\n"
         "  --display auto|true|false|offscreen  Show the left image and the disparity\n"
         "                             (default: auto, on when a display is available);\n"
         "                             offscreen renders the same into GPU memory and\n"
         "                             checks what it shows\n"
         "  --snapshot FILE            With --display offscreen, write the newest rendered\n"
         "                             frame as a PPM file\n"
         "  --compile-only true|false  Build and compile the graph, then exit without running\n"
         "                             it (default: false)\n";
}

// The recording's folder: --recording, or --clip in $HOLOSCAN_TEST_DATA.
std::filesystem::path recording_folder(const Options& opts) {
  if (opts.recording && !opts.clip.empty()) {
    throw std::invalid_argument("give --recording or --clip, not both");
  }
  if (opts.recording) return *opts.recording;
  if (opts.clip.empty()) {
    throw std::invalid_argument("--source recording needs --recording DIR or --clip NAME");
  }
  const char* data = std::getenv("HOLOSCAN_TEST_DATA");
  if (data == nullptr || *data == '\0') {
    throw std::invalid_argument(
        "--clip needs HOLOSCAN_TEST_DATA, the folder that holds stereo-camera/");
  }
  return std::filesystem::path(data) / "stereo-camera" / opts.clip;
}

void validate(const Options& opts) {
  if (opts.model.empty()) {
    throw std::invalid_argument("--model PATH or --model stub is required");
  }
  if (opts.model != "stub" && !std::filesystem::is_regular_file(opts.model)) {
    throw std::invalid_argument("no model at " + opts.model +
                                "; prepare it with the module's prepare_model.py");
  }
  if (opts.period_ms == 0 || opts.period_ms > 1000) {
    throw std::invalid_argument("--period-ms must be 1..1000");
  }
  if (opts.count && *opts.count == 0) {
    throw std::invalid_argument("--count must be positive");
  }
  if (!(opts.min_camera_coverage >= 0.0 && opts.min_camera_coverage <= 1.0)) {
    throw std::invalid_argument("--min-camera-coverage must be 0..1");
  }
  if (opts.source == Source::kSynthetic && (opts.recording || !opts.clip.empty() || opts.loop)) {
    throw std::invalid_argument("--recording, --clip and --loop apply to --source recording");
  }
  if (!opts.snapshot.empty() && opts.display != DisplayMode::kOffscreen) {
    throw std::invalid_argument("--snapshot needs --display offscreen");
  }
}

double percentile(std::vector<float> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
  return values[index];
}

double ratio(std::uint64_t part, std::uint64_t whole) {
  return whole == 0 ? 0.0 : static_cast<double>(part) / static_cast<double>(whole);
}

// The run's line, and whether its results pass.
bool print_report(const fs::FoundationStereoStats& stats, const app::DisparityReport& report,
                  std::uint32_t frames, const Options& opts) {
  const std::uint64_t timed = stats.timed_frames.load();
  const double gpu_ms = timed == 0 ? 0.0 : static_cast<double>(stats.gpu_ns_total) / 1e6 / timed;
  std::cout << std::fixed << std::setprecision(2) << kAppName << ": model=" << stats.model_width
            << 'x' << stats.model_height << " frames=" << report.frames << '/' << frames
            << " gpu_ms(mean/max)=" << gpu_ms << '/' << static_cast<double>(stats.gpu_ns_max) / 1e6
            << " latency_ms(p50/p95)=" << percentile(report.latency_ms, 0.5) << '/'
            << percentile(report.latency_ms, 0.95) << std::setprecision(0)
            << " model_memory_mib=" << static_cast<double>(stats.model_bytes) / (1024.0 * 1024.0)
            << std::setprecision(3) << " valid=" << ratio(report.valid_pixels, report.pixels);
  // A real-time run may drop frames; a lossless one must not.
  bool passed = opts.realtime ? report.frames > 0 && report.frames + report.frame_gaps >= frames
                              : report.frames == frames && report.frame_gaps == 0;
  if (report.stub_checked) {
    std::cout << " stub_reference: pixels=" << report.stub_pixels << " wrong=" << report.stub_wrong
              << " worst_px=" << report.stub_worst;
    passed = passed && report.stub_pixels > 0 && report.stub_wrong == 0;
  } else if (report.verified) {
    const double valid = ratio(report.matched_pixels, report.truth_pixels);
    const double within = ratio(report.within_1px, report.matched_pixels);
    const double mae = report.matched_pixels == 0
                           ? 0.0
                           : report.abs_error_sum / static_cast<double>(report.matched_pixels);
    const double median = report.truth_median();
    std::cout << " truth: valid=" << valid << " within_1px=" << within << " median_px=" << median
              << " epe_px=" << mae;
    passed =
        passed && valid >= kMinTruthValid && within >= kMinWithin1px && median <= kMaxMedianErrorPx;
  }
  if (report.compared) {
    const double camera = ratio(report.camera_pixels, report.pixels);
    const double coverage = ratio(report.both_pixels, report.camera_pixels);
    const double median = report.error_quantile(0.5);
    const double within = report.share_within(0.05);
    // The stub model does not match stereo: its depth is only reported.
    const bool stub = opts.model == "stub";
    const bool checked = camera >= opts.min_camera_coverage && !stub;
    std::cout << " camera_depth: camera_valid=" << camera << " coverage=" << coverage
              << " median_rel_err=" << std::setprecision(4) << median
              << " p90_rel_err=" << report.error_quantile(0.9) << std::setprecision(3)
              << " within_5pct=" << within
              << (checked ? " (checked)"
                  : stub  ? " (stub model; reported only)"
                          : " (too little camera depth to check; reported only)");
    if (checked) {
      passed = passed && coverage >= kMinCoverage && median <= kMaxMedianError &&
               within >= kMinWithin5pct;
    }
  }
  if (report.frame_gaps != 0) std::cout << " dropped=" << report.frame_gaps;
  std::cout << '\n';
  return passed;
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  holoscan::Graph graph{std::string(kAppName)};
  const auto period = std::chrono::milliseconds(opts.period_ms);
  const bool recorded = opts.source == Source::kRecording;
  const bool stub = opts.model == "stub";
  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };

  // The source, and the pair's geometry.
  int width = opts.width;
  int height = opts.height;
  std::uint32_t frames = 0;
  fs::StereoGeometry geometry;
  std::optional<double> camera_depth_scale;
  const holoscan::Output<holoscan::schema::ImageT>* left_out = nullptr;
  const holoscan::Output<holoscan::schema::ImageT>* right_out = nullptr;
  const holoscan::Output<holoscan::schema::ImageT>* view_out = nullptr;
  const holoscan::Output<holoscan::schema::ImageT>* camera_depth_out = nullptr;
  const holoscan::Output<fs::schema::StereoGeometryT>* geometry_out = nullptr;
  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool offscreen = opts.display == DisplayMode::kOffscreen;
  const bool display = opts.display == DisplayMode::kOn || offscreen ||
                       (opts.display == DisplayMode::kAuto && display_available &&
                        FOUNDATION_STEREO_ENABLE_VISUALIZATION);
  if (display && !FOUNDATION_STEREO_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !offscreen && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add --display false");
  }
  if (recorded) {
    app::Recording recording;
    try {
      recording = app::read_recording(recording_folder(opts));
    } catch (const std::runtime_error& error) {
      // A recording that cannot be read is a usage error.
      throw std::invalid_argument(error.what());
    }
    const app::RecordingCalibration& calibration = recording.calibration;
    width = calibration.width;
    height = calibration.height;
    geometry = app::recording_geometry(calibration);
    camera_depth_scale = calibration.depth_scale;
    std::cout << kAppName << ": recording " << recording.folder.string() << " ("
              << calibration.camera << ") " << width << 'x' << height << ", "
              << recording.frames.size() << " frames, fx=" << std::setprecision(2) << std::fixed
              << calibration.fx << " px, baseline=" << 1000.0 * calibration.baseline << " mm"
              << (camera_depth_scale ? ", with the camera's depth" : "") << '\n';
    app::RecordingSourceConfig config{.recording = std::move(recording),
                                      .count = opts.count.value_or(0),
                                      .loop = opts.loop,
                                      .period = period,
                                      .depth = camera_depth_scale.has_value(),
                                      .geometry = opts.geometry_input,
                                      .view = display};
    // Configurations are checked here, where a bad one is a usage error; the
    // graph reports a constructor's exception as its own failure.
    app::validate_config(config);
    auto source = graph.op<app::RecordingSourceOp>("source", std::move(config));
    frames = source->total();
    left_out = &source->left;
    right_out = &source->right;
    if (display) view_out = &source->view;
    if (camera_depth_scale) camera_depth_out = &source->depth;
    if (opts.geometry_input) geometry_out = &source->geometry;
  } else {
    if (opts.geometry_input) {
      throw std::invalid_argument("--geometry-input applies to --source recording");
    }
    frames = opts.count.value_or(300);
    const app::SyntheticSourceConfig config{.width = width,
                                            .height = height,
                                            .count = frames,
                                            .period = period,
                                            .color = opts.color,
                                            .device_output = opts.device_input};
    app::validate_config(config);
    auto source = graph.op<app::SyntheticStereoSourceOp>("source", config);
    geometry = fs::scene_geometry(fs::make_scene(0, width, height));
    left_out = &source->left;
    right_out = &source->right;
    // A color pair shows its left image; a MONO8 one the disparity alone.
    if (opts.color) view_out = &source->left;
    if (opts.device_input) {
      on_gpu("source", "left");
      on_gpu("source", "right");
    }
  }

  // The model, its engine built before the graph runs.
  fs::FoundationStereoConfig config{
      .model = stub ? fs::write_stub_model(std::filesystem::current_path(), kStubWidth, kStubHeight)
                    : std::filesystem::path(opts.model),
      .fp16 = opts.fp16,
      .geometry_input = opts.geometry_input,
      .depth = camera_depth_scale.has_value(),
      .invalidate_out_of_view = opts.invalidate_out_of_view,
      .max_width = width,
      .max_height = height,
      .stats = std::make_shared<fs::FoundationStereoStats>()};
  if (!opts.geometry_input) config.geometry = geometry;
  fs::validate_config(config);
  if (!opts.compile_only) {
    fs::prepare_engine(config);
  }
  auto stats = config.stats;
  auto render_check = std::make_shared<app::RenderCheck>();
  auto stereo = graph.op<fs::FoundationStereoOp>("stereo", config);
  auto report = std::make_shared<app::DisparityReport>();
  app::DisparityReportConfig report_config{.report = report,
                                           .expected_frames = frames,
                                           .remaining = std::make_shared<std::atomic<int>>(1),
                                           .truth = !recorded && !stub,
                                           .width = width,
                                           .height = height,
                                           .camera_depth_scale = camera_depth_scale,
                                           .color = opts.color};
  if (stub && !recorded) {
    report_config.stub_model_size = std::array<int, 2>{kStubWidth, kStubHeight};
    // FP16 holds the stub's 0..255 values to about 0.1.
    report_config.stub_tolerance = opts.fp16 ? 1e-2 : 1e-4;
  }
  auto reporter = graph.op<app::DisparityReportOp>("report", std::move(report_config));

  // Lossless edges for the checks; drop-oldest ones of depth 1 for a live
  // source, so the model always takes the newest pair and never holds back
  // the source.
  const holoscan::ConnectionOptions edge =
      opts.realtime
          ? holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                        .queue_depth = 1U}
          : holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(*left_out, stereo->left, edge);
  graph.add_flow(*right_out, stereo->right, edge);
  if (geometry_out != nullptr) graph.add_flow(*geometry_out, stereo->geometry, edge);
  graph.add_flow(stereo->disparity, reporter->disparity, edge);
  on_gpu("stereo", "disparity");
  if (camera_depth_scale) {
    graph.add_flow(stereo->depth, reporter->depth, edge);
    graph.add_flow(*camera_depth_out, reporter->camera_depth, edge);
    on_gpu("stereo", "depth");
  }

#if FOUNDATION_STEREO_ENABLE_VISUALIZATION
  if (display) {
    namespace viz = holoscan::visualization;
    const auto w = static_cast<std::uint32_t>(width);
    const auto h = static_cast<std::uint32_t>(height);
    const bool show_left = view_out != nullptr;
    viz::LayerStack layers;
    if (show_left) {
      layers.add_image_layer("left",
                             viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                                 .max_width = w,
                                                 .max_height = h,
                                                 .view = viz::ViewTransform{.scale_x = 0.5F}});
    }
    layers.add_image_layer(
        "disparity", viz::ImageLayerDesc{
                         .format = viz::ImageFormat::kR8G8B8A8Unorm,
                         .max_width = w,
                         .max_height = h,
                         .view = show_left ? viz::ViewTransform{.offset_x = 0.5F, .scale_x = 0.5F}
                                           : viz::ViewTransform{}});
    const double columns = show_left ? 2.0 : 1.0;
    const double scale = std::min(1.0, 1920.0 / (columns * w));
    const auto max_input_age = std::max(4 * period, std::chrono::milliseconds{250});
    const auto window_width = static_cast<std::uint32_t>(columns * w * scale);
    const auto window_height = static_cast<std::uint32_t>(h * scale);
    viz::SinkDesc sink = viz::WindowSinkDesc{
        .title = "Foundation Stereo", .width = window_width, .height = window_height};
    if (offscreen) {
      sink = viz::CudaSinkDesc{.width = window_width,
                               .height = window_height,
                               .format = viz::ImageFormat::kR8G8B8A8Unorm};
    }
    auto visualizer = graph.op<viz::HolovizOp>(
        "holoviz", std::move(layers), std::move(sink),
        viz::FrameTrigger{viz::LayerTrigger{show_left ? 1U : 0U}}, max_input_age);
    if (offscreen) {
      auto checker = graph.op<app::RenderCheckOp>(
          "render_check", app::RenderCheckConfig{.check = render_check,
                                                 .width = static_cast<int>(window_width),
                                                 .height = static_cast<int>(window_height),
                                                 .left_image = show_left,
                                                 .write = opts.snapshot});
      graph.add_flow(visualizer->output(), checker->frame,
                     holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                                 .queue_depth = 2U});
      on_gpu("holoviz", "output");
    }
    // The color scale: up to the nearest the scene comes, 0.3 m from a camera,
    // or the stub's largest value.
    const float near = recorded ? static_cast<float>(geometry.fx * geometry.baseline / 0.3)
                       : stub   ? static_cast<float>(width) / kStubWidth * 56.0F
                                : fs::max_scene_disparity(width);
    auto colors = graph.op<app::DisparityColorOp>(
        "colors",
        app::DisparityColorConfig{.max_disparity = near, .max_width = width, .max_height = height});
    // Best effort: the display keeps the newest sample and never holds back the model.
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    // The frame's left image must still be queued when its disparity arrives,
    // or HoloViz, which takes the newest image no newer than it, falls back to
    // an older one: keep a few.
    const holoscan::ConnectionOptions recent{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 4U};
    if (show_left) {
      graph.add_flow(*view_out, visualizer->image_layer_input(0), recent);
    }
    graph.add_flow(stereo->disparity, colors->disparity, latest);
    graph.add_flow(colors->image, visualizer->image_layer_input(show_left ? 1U : 0U), latest);
    on_gpu("colors", "image");
  }
#endif

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
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
  // In real time, frames the model drops never reach the reporter: stop once
  // the source has sent its last frame and the pipeline has drained.
  const auto source_done = std::chrono::milliseconds(1LL * frames * opts.period_ms) + 2000ms;
  const auto started = std::chrono::steady_clock::now();
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    if (opts.realtime && std::chrono::steady_clock::now() - started > source_done) {
      session.request_stop();
      break;
    }
    std::this_thread::sleep_for(10ms);
  }
  const bool timed_out = session.is_running() && !opts.realtime;
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
  // failure, except the one that ends a real-time run.
  const bool completed = termination == holoscan::RunTermination::kRequestedCompleted ||
                         (opts.realtime && termination == holoscan::RunTermination::kStopped);
  const bool stopped =
      !timed_out && !completed && termination == holoscan::RunTermination::kStopped;
  bool passed =
      !timed_out && completed &&
      (session.diagnostics().empty() || termination == holoscan::RunTermination::kStopped);
  passed = print_report(*stats, *report, frames, opts) && passed;
  if (offscreen) {
    std::cout << std::setprecision(3) << kAppName << ": offscreen: frames=" << render_check->frames
              << " failed=" << render_check->failed << " colored=" << render_check->colored
              << " left_brightness=" << std::setprecision(1) << render_check->left_brightness
              << (render_check->written ? " snapshot=" + opts.snapshot.string() : std::string())
              << '\n';
    passed = passed && render_check->frames > 0 && render_check->failed == 0 &&
             (opts.snapshot.empty() || render_check->written);
  }
  std::cout << kAppName << ": verdict=" << (passed ? "PASS" : stopped ? "STOPPED" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM first: the SDK refuses once other threads run,
  // and TensorRT starts its own while it builds or loads an engine.
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
