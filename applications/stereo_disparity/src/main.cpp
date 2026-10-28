// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Stereo Disparity: a stereo source (the module's synthetic scene with known
// disparity, or a recorded camera clip) fans out to a StereoDisparityOp per
// backend, each feeding a reporter that checks it against the scene's truth or
// the camera's own depth; HoloViz can show the first backend's disparity.

#include <algorithm>
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
#include <holoscan/logger/logger.hpp>
#include <holoscan/stereo_disparity/platform.hpp>
#include <holoscan/stereo_disparity/stereo_disparity_op.hpp>
#include <holoscan/stereo_disparity/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if STEREO_DISPARITY_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "disparity_color_op.hpp"
#include "disparity_report_op.hpp"
#include "recording.hpp"
#include "stereo_sources.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace sd = holoscan::stereo_disparity;
namespace app = stereo_disparity_app;

constexpr std::string_view kAppName = "stereo_disparity";

enum class Source { kSynthetic, kRecording };
enum class DisplayMode { kAuto, kOn, kOff };

// How long a finite run may take beyond its expected time.
constexpr auto kTimeoutAllowance = std::chrono::seconds{120};

// What every backend must reach on the synthetic scene, over the pixels both
// cameras see: the share with a disparity, the share of those within 1 px of
// the truth, and their mean error. The module's tests hold each backend to
// tighter bounds of its own.
constexpr double kMinTruthValid = 0.85;
constexpr double kMinWithin1px = 0.9;
constexpr double kMaxMeanErrorPx = 0.5;

// Against the camera's own depth, on a clip where the camera has depth for at
// least --min-camera-coverage of the pixels: the computed depth must cover
// this share of the camera's pixels, with this median relative error and this
// share within 5 %. Measured with VPI 4.1.4, the projector on, at 720p and
// 480p: coverage 0.98 to 1.0, median 0.9 % to 1.1 %, within 5 % 0.86 to 0.90 on
// cuda, ofa and pva alike. The camera and VPI match the same infrared pair,
// both with semi-global matching, so they should agree to a fraction of a
// pixel: at the ceiling's 2.5 m (13 px at 720p), 3 % is 0.4 px.
constexpr double kMinCoverage = 0.85;
constexpr double kMaxMedianError = 0.03;
constexpr double kMinWithin5pct = 0.75;

struct Options {
  Source source{Source::kSynthetic};
  std::vector<sd::Backend> backends{sd::Backend::kCuda};
  sd::DisparityParams params;
  bool confidence{false};
  bool geometry_input{false};
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

std::vector<sd::Backend> parse_backends(std::string_view value) {
  std::vector<sd::Backend> backends;
  const auto add = [&backends](sd::Backend backend) {
    if (std::find(backends.begin(), backends.end(), backend) != backends.end()) {
      throw std::invalid_argument("backend '" + std::string(sd::backend_name(backend)) +
                                  "' is listed twice");
    }
    backends.push_back(backend);
  };
  while (!value.empty()) {
    const std::size_t comma = value.find(',');
    const std::string_view name = value.substr(0, comma);
    if (name == "all") {
      for (const sd::Backend backend : sd::available_backends()) add(backend);
    } else if (const auto backend = sd::parse_backend(name)) {
      add(*backend);
    } else {
      throw std::invalid_argument("unknown backend '" + std::string(name) +
                                  "'; use cuda, ofa, pva, auto or all");
    }
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
      {"--backend",
       [&](std::string_view value, std::string_view) { opts.backends = parse_backends(value); }},
      {"--max-disparity", number(opts.params.max_disparity)},
      {"--confidence-threshold", number(opts.params.confidence_threshold)},
      {"--uniqueness", number(opts.params.uniqueness)},
      {"--p1", number(opts.params.p1)},
      {"--p2", number(opts.params.p2)},
      {"--window", number(opts.params.window)},
      {"--num-passes", number(opts.params.num_passes)},
      {"--confidence", flag(opts.confidence)},
      {"--geometry-input", flag(opts.geometry_input)},
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
      << " [options]\n"
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
         "Disparity:\n"
         "  --backend LIST             Comma-separated: cuda, ofa, pva, auto (ofa on a\n"
         "                             Jetson when it suits, else cuda), or all this\n"
         "                             platform has (default: cuda)\n"
         "  --max-disparity N          1..256 for cuda, 128 or 256 for ofa and pva\n"
         "                             (default: 128)\n"
         "  --confidence-threshold N   cuda, pva: 0..65535 (default: 32767)\n"
         "  --uniqueness X             cuda: 0..1, or -1 for none (default: -1)\n"
         "  --p1 N --p2 N              SGM penalties (default: 3 and 48)\n"
         "  --window N                 pva: 1, 3, 5 or 7 (default: 5)\n"
         "  --num-passes N             ofa, pva: 1..3 (default: 3)\n"
         "  --confidence true|false    Also output VPI's confidence, and report its mean;\n"
         "                             cuda and pva (default: false)\n"
         "Checks and output:\n"
         "  --min-camera-coverage X    Check accuracy against a recording's depth only when\n"
         "                             the camera has depth for this share of pixels\n"
         "                             (default: 0.2)\n"
         "  --display auto|true|false  Show the first backend's disparity (default: auto,\n"
         "                             on when a display is available)\n"
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
        "--clip needs HOLOSCAN_TEST_DATA, the folder that holds "
        "stereo-camera/");
  }
  return std::filesystem::path(data) / "stereo-camera" / opts.clip;
}

void validate(const Options& opts) {
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

// What the auto backend must resolve to on a CI runner, from
// HOLOSCAN_CI_PLATFORM: ofa on a Jetson, cuda on x86_64.
std::optional<sd::Backend> expected_auto_backend() {
  const char* platform = std::getenv("HOLOSCAN_CI_PLATFORM");
  if (platform == nullptr) return std::nullopt;
  if (std::string_view(platform) == "jetson") return sd::Backend::kOfa;
  if (std::string_view(platform) == "x86_64") return sd::Backend::kCuda;
  return std::nullopt;
}

// One backend's line, and whether its results pass.
bool print_report(sd::Backend requested, const sd::StereoDisparityStats& stats,
                  const app::DisparityReport& report, std::uint32_t frames,
                  double min_camera_coverage) {
  const std::uint64_t computed = stats.frames.load();
  const double mean_ms =
      computed == 0 ? 0.0 : static_cast<double>(stats.compute_ns_total) / 1e6 / computed;
  std::cout << std::fixed << std::setprecision(2) << kAppName << ": backend=";
  if (requested == sd::Backend::kAuto) std::cout << "auto:";
  std::cout << sd::backend_name(stats.backend.load()) << " frames=" << report.frames << '/'
            << frames << " compute_ms(mean/max)=" << mean_ms << '/'
            << static_cast<double>(stats.compute_ns_max) / 1e6
            << " latency_ms(p50/p95)=" << percentile(report.latency_ms, 0.5) << '/'
            << percentile(report.latency_ms, 0.95) << std::setprecision(3)
            << " valid=" << ratio(report.valid_pixels, report.pixels);
  bool passed = report.frames == frames && report.frame_gaps == 0;
  if (report.verified) {
    const double valid = ratio(report.matched_pixels, report.truth_pixels);
    const double within = ratio(report.within_1px, report.matched_pixels);
    const double mae = report.matched_pixels == 0
                           ? 0.0
                           : report.abs_error_sum / static_cast<double>(report.matched_pixels);
    std::cout << " truth: valid=" << valid << " within_1px=" << within << " mae_px=" << mae;
    passed = passed && valid >= kMinTruthValid && within >= kMinWithin1px && mae <= kMaxMeanErrorPx;
  }
  if (report.compared) {
    const double camera = ratio(report.camera_pixels, report.pixels);
    const double coverage = ratio(report.both_pixels, report.camera_pixels);
    const double median = report.error_quantile(0.5);
    const double within = report.share_within(0.05);
    const bool checked = camera >= min_camera_coverage;
    std::cout << " camera_depth: camera_valid=" << camera << " coverage=" << coverage
              << " median_rel_err=" << std::setprecision(4) << median
              << " p90_rel_err=" << report.error_quantile(0.9) << std::setprecision(3)
              << " within_5pct=" << within
              << (checked ? " (checked)" : " (too little camera depth to check; reported only)");
    if (checked) {
      passed = passed && coverage >= kMinCoverage && median <= kMaxMedianError &&
               within >= kMinWithin5pct;
    }
  }
  if (report.confidence_pixels != 0) {
    std::cout << " mean_confidence=" << std::setprecision(0)
              << report.confidence_sum / static_cast<double>(report.confidence_pixels)
              << std::setprecision(3);
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
  sd::StereoGeometry geometry;
  std::optional<double> camera_depth_scale;
  const holoscan::Output<holoscan::schema::ImageT>* left_out = nullptr;
  const holoscan::Output<holoscan::schema::ImageT>* right_out = nullptr;
  const holoscan::Output<holoscan::schema::ImageT>* camera_depth_out = nullptr;
  const holoscan::Output<sd::schema::StereoGeometryT>* geometry_out = nullptr;
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
                                      .geometry = opts.geometry_input};
    // Configurations are checked here, where a bad one is a usage error; the
    // graph reports a constructor's exception as its own failure.
    app::validate_config(config);
    auto source = graph.op<app::RecordingSourceOp>("source", std::move(config));
    frames = source->total();
    left_out = &source->left;
    right_out = &source->right;
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
    geometry = sd::scene_geometry(sd::make_scene(0, width, height));
    left_out = &source->left;
    right_out = &source->right;
    if (opts.device_input) {
      on_gpu("source", "left");
      on_gpu("source", "right");
    }
  }

  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool display = opts.display == DisplayMode::kOn ||
                       (opts.display == DisplayMode::kAuto && display_available &&
                        STEREO_DISPARITY_ENABLE_VISUALIZATION);
  if (display && !STEREO_DISPARITY_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add --display false");
  }

  // Lossless edges: every matcher sees every frame, in order, for the checks.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  auto remaining = std::make_shared<std::atomic<int>>(static_cast<int>(opts.backends.size()));
  std::vector<std::shared_ptr<app::DisparityReport>> reports;
  std::vector<std::shared_ptr<sd::StereoDisparityStats>> stats;
  std::optional<holoscan::OperatorHandle<sd::StereoDisparityOp>> first_matcher;
  for (const sd::Backend backend : opts.backends) {
    const std::string name(sd::backend_name(backend));
    // The OFA has no confidence map; its reporter goes without.
    const bool confidence = opts.confidence && backend != sd::Backend::kOfa;
    sd::StereoDisparityConfig config{.backend = backend,
                                     .params = opts.params,
                                     .geometry_input = opts.geometry_input,
                                     .depth = camera_depth_scale.has_value(),
                                     .confidence = confidence,
                                     .max_width = width,
                                     .max_height = height,
                                     .stats = std::make_shared<sd::StereoDisparityStats>()};
    if (!opts.geometry_input) config.geometry = geometry;
    sd::validate_config(config);
    stats.push_back(config.stats);
    auto matcher = graph.op<sd::StereoDisparityOp>("stereo_" + name, config);
    auto report = std::make_shared<app::DisparityReport>();
    reports.push_back(report);
    auto reporter = graph.op<app::DisparityReportOp>(
        "report_" + name, app::DisparityReportConfig{.report = report,
                                                     .expected_frames = frames,
                                                     .remaining = remaining,
                                                     .truth = !recorded,
                                                     .width = width,
                                                     .height = height,
                                                     .camera_depth_scale = camera_depth_scale,
                                                     .confidence = confidence});
    graph.add_flow(*left_out, matcher->left, lossless);
    graph.add_flow(*right_out, matcher->right, lossless);
    if (geometry_out != nullptr) graph.add_flow(*geometry_out, matcher->geometry, lossless);
    graph.add_flow(matcher->disparity, reporter->disparity, lossless);
    on_gpu("stereo_" + name, "disparity");
    if (camera_depth_scale) {
      graph.add_flow(matcher->depth, reporter->depth, lossless);
      graph.add_flow(*camera_depth_out, reporter->camera_depth, lossless);
      on_gpu("stereo_" + name, "depth");
    }
    if (confidence) {
      graph.add_flow(matcher->confidence, reporter->confidence, lossless);
      on_gpu("stereo_" + name, "confidence");
    }
    if (!first_matcher) first_matcher.emplace(matcher);
  }

#if STEREO_DISPARITY_ENABLE_VISUALIZATION
  if (display) {
    namespace viz = holoscan::visualization;
    const auto w = static_cast<std::uint32_t>(width);
    const auto h = static_cast<std::uint32_t>(height);
    // Color images beside the disparity; a MONO8 pair shows the disparity alone.
    const bool show_left = !recorded && opts.color;
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
    auto visualizer = graph.op<viz::HolovizOp>(
        "holoviz", std::move(layers),
        viz::WindowSinkDesc{.title = "Stereo Disparity (" +
                                     std::string(sd::backend_name(opts.backends.front())) + ")",
                            .width = static_cast<std::uint32_t>(columns * w * scale),
                            .height = static_cast<std::uint32_t>(h * scale)},
        viz::FrameTrigger{viz::LayerTrigger{show_left ? 1U : 0U}}, max_input_age);
    // The color scale: up to the nearest the scene comes, or 0.3 m from a camera.
    const float searched = static_cast<float>(sd::searched_disparity(
        opts.backends.front() == sd::Backend::kAuto ? sd::Backend::kCuda : opts.backends.front(),
        opts.params));
    const float near = recorded ? static_cast<float>(geometry.fx * geometry.baseline / 0.3)
                                : sd::max_scene_disparity(width);
    auto colors = graph.op<app::DisparityColorOp>(
        "colors",
        app::DisparityColorConfig{
            .max_disparity = std::min(near, searched), .max_width = width, .max_height = height});
    // Best effort: the display keeps the newest sample and never holds back the matchers.
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    // The frame's left image must still be queued when its disparity arrives,
    // or HoloViz, which takes the newest image no newer than it, falls back to
    // an older one: keep a few.
    const holoscan::ConnectionOptions recent{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 4U};
    if (show_left) {
      graph.add_flow(*left_out, visualizer->image_layer_input(0), recent);
    }
    graph.add_flow((*first_matcher)->disparity, colors->disparity, latest);
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
  // failure.
  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
  bool passed = !timed_out && session.diagnostics().empty() &&
                termination == holoscan::RunTermination::kRequestedCompleted;
  for (std::size_t i = 0; i < reports.size(); ++i) {
    passed =
        print_report(opts.backends[i], *stats[i], *reports[i], frames, opts.min_camera_coverage) &&
        passed;
  }
  // On a CI runner, auto must have chosen the platform's best backend.
  const auto auto_backend =
      std::find(opts.backends.begin(), opts.backends.end(), sd::Backend::kAuto);
  const auto expected_auto = expected_auto_backend();
  if (auto_backend != opts.backends.end() && expected_auto) {
    const sd::Backend chosen = stats[auto_backend - opts.backends.begin()]->backend;
    std::cout << kAppName << ": auto=" << sd::backend_name(chosen)
              << " expected=" << sd::backend_name(*expected_auto) << " ("
              << std::getenv("HOLOSCAN_CI_PLATFORM") << ")\n";
    passed = passed && chosen == *expected_auto;
  }
  std::cout << kAppName << ": verdict=" << (passed ? "PASS" : stopped ? "STOPPED" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM first: the SDK refuses once other threads run,
  // and VPI starts its own as soon as it is asked anything.
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
