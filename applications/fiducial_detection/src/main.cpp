// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Fiducial Detection: synthetic scenes of ArUco markers and of a ChArUco board, or a
// recording's color frames, each fanned out to a detector per backend whose reporter checks it
// against ground truth (or the backends against each other); or a V4L2 camera. Calibration is a
// ROS camera_info file sent by CameraInfoSourceOp: the synthetic camera's, written at start
// into a temporary folder; the recording's rgb.yaml; or the camera's, with --camera-info. The
// first detector can also draw on a HoloViz display.

#include <stdlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
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
#include <holoscan/fiducials/aruco_detector_op.hpp>
#include <holoscan/fiducials/charuco_detector_op.hpp>
#include <holoscan/fiducials/detector_backend.hpp>
#include <holoscan/fiducials/dictionary.hpp>
#include <holoscan/fiducials/pose.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if FIDUCIAL_DETECTION_ENABLE_VISUALIZATION
#include <holoscan/fiducials/fiducial_overlay_op.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "recording.hpp"
#include "report_op.hpp"
#include "sources.hpp"

namespace {

namespace fd = holoscan::fiducials;
namespace hm = holoscan::messages;
namespace fs = std::filesystem;
using fiducial_app::Report;
using fiducial_app::SceneKind;

constexpr std::string_view kAppName = "fiducial_detection";
// How long a finite run may take beyond its expected time.
constexpr auto kTimeoutAllowance = std::chrono::seconds{120};
// The markers' size in the synthetic marker scenes.
constexpr double kSceneMarkerLength = 0.1;

enum class DisplayMode { kAuto, kOn, kOff };
enum class Source { kSynthetic, kCamera, kRecording };

// A new folder for the synthetic camera's camera_info file, removed with it when the demo ends,
// however it ends but by a signal it does not handle (as SIGINT or SIGTERM when the managed
// signal domain could not be made), which leaves the folder in the system's temporary folder.
class CalibrationFolder {
 public:
  CalibrationFolder() {
    std::string pattern = (fs::temp_directory_path() / "fiducial_detection-XXXXXX").string();
    if (mkdtemp(pattern.data()) == nullptr) {
      throw std::system_error(errno, std::generic_category(),
                              "cannot make a temporary folder from " + pattern);
    }
    path_ = pattern;
  }
  ~CalibrationFolder() {
    std::error_code ignored;  // Nothing to do about a folder that cannot be removed.
    fs::remove_all(path_, ignored);
  }
  CalibrationFolder(const CalibrationFolder&) = delete;
  CalibrationFolder& operator=(const CalibrationFolder&) = delete;

  [[nodiscard]] const fs::path& path() const noexcept { return path_; }

 private:
  fs::path path_;
};

struct Options {
  std::vector<fd::Backend> backends{fd::Backend::kCuda};
  bool aruco{true};    // Detect loose markers.
  bool charuco{true};  // Detect a ChArUco board.
  std::string dictionary{"DICT_4X4_50"};
  int min_marker_px{fd::DetectorOptions{}.min_marker_px};
  fd::CharucoBoardConfig board;
  Source source{Source::kSynthetic};
  std::string device{"/dev/video0"};
  std::optional<fs::path> camera_info;  // The camera's camera_info file (--camera-info).
  std::optional<fs::path> recording;
  double marker_length{kSceneMarkerLength};
  int width{1280};
  int height{720};
  std::uint32_t count{300};
  std::uint32_t period_ms{33};
  bool device_input{false};
  DisplayMode display{DisplayMode::kAuto};
  bool compile_only{false};
  bool help{false};
  std::set<std::string, std::less<>> given;  // The options on the command line.

  // Filled in by validate(): the camera_info file CameraInfoSourceOp reads, if any, and for
  // synthetic scenes the folder it was written into.
  std::optional<fs::path> camera_info_file;
  std::shared_ptr<CalibrationFolder> synthetic_files;
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

// Splits "a<separator>b<separator>c".
std::vector<std::string_view> split(std::string_view value, char separator) {
  std::vector<std::string_view> parts;
  while (true) {
    const std::size_t at = value.find(separator);
    parts.push_back(value.substr(0, at));
    if (at == std::string_view::npos) return parts;
    value = value.substr(at + 1);
  }
}

std::vector<fd::Backend> parse_backends(std::string_view value) {
  std::vector<fd::Backend> backends;
  const auto add = [&backends](fd::Backend backend) {
    if (std::find(backends.begin(), backends.end(), backend) != backends.end()) {
      throw std::invalid_argument("backend '" + std::string(fd::backend_name(backend)) +
                                  "' is listed twice");
    }
    backends.push_back(backend);
  };
  for (const std::string_view name : split(value, ',')) {
    if (name == "all") {
      for (const fd::Backend backend : fd::available_backends()) add(backend);
    } else if (const auto backend = fd::parse_backend(name)) {
      add(*backend);
    } else {
      throw std::invalid_argument("unknown backend '" + std::string(name) +
                                  "'; use cpu, cuda, auto or all");
    }
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
      {"--scene",
       [&](std::string_view value, std::string_view) {
         if (value != "aruco" && value != "charuco" && value != "both") {
           throw std::invalid_argument("--scene takes aruco, charuco or both");
         }
         opts.aruco = value != "charuco";
         opts.charuco = value != "aruco";
       }},
      {"--dictionary", [&](std::string_view value, std::string_view) { opts.dictionary = value; }},
      {"--board",
       [&](std::string_view value, std::string_view option) {
         const auto parts = split(value, 'x');
         if (parts.size() != 2) throw std::invalid_argument("--board takes COLUMNSxROWS");
         parse_number(parts[0], option, opts.board.squares_x);
         parse_number(parts[1], option, opts.board.squares_y);
       }},
      {"--square-length", number(opts.board.square_length)},
      {"--board-marker-length", number(opts.board.marker_length)},
      {"--board-dictionary",
       [&](std::string_view value, std::string_view) { opts.board.dictionary = value; }},
      {"--legacy-pattern", flag(opts.board.legacy_pattern)},
      {"--source",
       [&](std::string_view value, std::string_view) {
         if (value != "synthetic" && value != "camera") {
           throw std::invalid_argument("--source takes synthetic or camera");
         }
         opts.source = value == "camera" ? Source::kCamera : Source::kSynthetic;
       }},
      {"--device", [&](std::string_view value, std::string_view) { opts.device = value; }},
      {"--camera-info",
       [&](std::string_view value, std::string_view) { opts.camera_info = fs::path(value); }},
      {"--recording",
       [&](std::string_view value, std::string_view) { opts.recording = fs::path(value); }},
      {"--marker-length", number(opts.marker_length)},
      {"--min-marker-px", number(opts.min_marker_px)},
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
    if (i + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(option));
    const auto* entry = std::find_if(std::begin(kOptions), std::end(kOptions),
                                     [option](const auto& known) { return known.first == option; });
    if (entry == std::end(kOptions)) {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
    entry->second(argv[++i], option);
    opts.given.emplace(option);
  }
  if (opts.recording) {
    if (opts.given.contains("--source")) {
      throw std::invalid_argument("--recording is a source of its own; leave out --source");
    }
    opts.source = Source::kRecording;
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "Input:\n"
         "  --source synthetic|camera  Rendered scenes with ground truth, or a V4L2 camera\n"
         "                             offering YUYV (default: synthetic)\n"
         "  --recording DIR            A recording's color frames instead (index.csv, RGB PPM\n"
         "                             images, rgb.yaml), every backend compared\n"
         "  --scene aruco|charuco|both What to detect (default: both)\n"
         "  --count N                  Frames per scene; 0 with the camera runs until stopped\n"
         "                             (default: 300; a recording's: all of them)\n"
         "  --period-ms N              Frame period of the scenes and the recording,\n"
         "                             1..1000 ms (default: 33)\n"
         "  --width N --height N       Image size of the scenes and the camera (default: "
         "1280x720)\n"
         "  --device-input true|false  Scene or recording frames in GPU memory (default: false)\n"
         "Camera:\n"
         "  --device PATH              The camera (default: /dev/video0)\n"
         "  --camera-info PATH         Its ROS camera_info file, of its image size, for poses\n"
         "  --marker-length M          The markers' size in meters, for poses (default: 0.1)\n"
         "Markers:\n"
         "  --dictionary NAME          For --scene aruco (default: DICT_4X4_50)\n"
         "ChArUco board:\n"
         "  --board CxR                Squares across and down (default: 5x7)\n"
         "  --square-length M          Meters (default: 0.04)\n"
         "  --board-marker-length M    Meters (default: 0.03)\n"
         "  --board-dictionary NAME    (default: DICT_4X4_50)\n"
         "  --legacy-pattern true|false  OpenCV's layout before 4.6 (default: false)\n"
         "Detection:\n"
         "  --backend LIST             Comma-separated: cpu, cuda, auto (cuda where there is a\n"
         "                             GPU, else cpu), or all (default: cuda)\n"
         "  --min-marker-px N          Smallest marker side to report, 4..1000 (default: 10)\n"
         "Output:\n"
         "  --display auto|true|false  Show the first detector's detections (default: auto, on\n"
         "                             when a display is available)\n"
         "  --compile-only true|false  Build and compile the graph, print its worker count, then\n"
         "                             exit (default: false)\n";
}

// Refuses `option` when it was given: it means nothing to `source`.
void refuse(const Options& opts, std::string_view option, std::string_view source) {
  if (opts.given.contains(option)) {
    throw std::invalid_argument(std::string(option) + " does not apply to " + std::string(source));
  }
}

void validate(Options& opts) {
  if (opts.period_ms == 0 || opts.period_ms > 1000) {
    throw std::invalid_argument("--period-ms must be 1..1000");
  }
  if (opts.count == 0 && opts.source == Source::kSynthetic) {
    throw std::invalid_argument("--count must be positive");
  }
  if (!(opts.marker_length > 0.0)) throw std::invalid_argument("--marker-length must be positive");
  fd::validate_options(
      fd::DetectorOptions{.dictionary = opts.dictionary, .min_marker_px = opts.min_marker_px});
  (void)fd::CharucoBoard(opts.board);
  switch (opts.source) {
    case Source::kSynthetic: {
      // The scenes have their own camera and markers.
      for (const char* option : {"--device", "--camera-info", "--marker-length"}) {
        refuse(opts, option, "synthetic scenes");
      }
      // Also checks the size.
      const hm::CameraInfo camera =
          fiducial_app::make_scene(SceneKind{.dictionary = opts.dictionary}, 0, opts.width,
                                   opts.height)
              .camera;
      opts.synthetic_files = std::make_shared<CalibrationFolder>();
      opts.camera_info_file = opts.synthetic_files->path() / "camera.yaml";
      hm::write_camera_info_yaml(camera, *opts.camera_info_file);
      break;
    }
    case Source::kCamera:
      refuse(opts, "--device-input", "the camera");
      if (opts.camera_info) {
        const hm::CameraInfo camera = hm::read_camera_info_yaml(*opts.camera_info);
        if (std::cmp_not_equal(camera.width, opts.width) ||
            std::cmp_not_equal(camera.height, opts.height)) {
          throw std::invalid_argument(
              opts.camera_info->string() + " is for " + std::to_string(camera.width) + "x" +
              std::to_string(camera.height) + " images, the camera is " +
              std::to_string(opts.width) + "x" + std::to_string(opts.height));
        }
        opts.camera_info_file = opts.camera_info;
      } else {
        refuse(opts, "--marker-length", "the camera without --camera-info");
      }
      break;
    case Source::kRecording: {
      for (const char* option : {"--device", "--camera-info", "--width", "--height"}) {
        refuse(opts, option, "a recording");
      }
      if (!opts.given.contains("--count")) opts.count = 0;  // All of its frames.
      // Reads the index and the first frame, which give the count and the size.
      const fiducial_app::FrameSourceConfig frames =
          fiducial_app::recording_source_config(*opts.recording, opts.count);
      opts.count = frames.count;
      opts.width = frames.width;
      opts.height = frames.height;
      opts.camera_info_file = fiducial_app::rgb_camera_info(*opts.recording);
      (void)hm::read_camera_info_yaml(*opts.camera_info_file);
      if (!opts.given.contains("--marker-length")) opts.marker_length = kSceneMarkerLength;
      break;
    }
  }
}

double percentile(std::vector<float> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  return values[static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1))];
}

// What auto must resolve to on a CI runner: every lane has a GPU.
bool on_ci_runner() { return env_set("HOLOSCAN_CI_PLATFORM"); }

struct Run {
  std::string scene;  // "aruco" or "charuco".
  fd::Backend requested;
  std::shared_ptr<Report> report;
};

void print_report(const Run& run, std::uint32_t expected_frames) {
  const Report& r = *run.report;
  double mean = 0.0;
  for (const float ms : r.detect_ms) mean += ms;
  mean = r.detect_ms.empty() ? 0.0 : mean / static_cast<double>(r.detect_ms.size());
  const double frames = std::max(1.0, static_cast<double>(r.frames));
  std::cout << std::fixed << std::setprecision(2) << kAppName << ": " << run.scene
            << " backend=" << (run.requested == fd::Backend::kAuto ? "auto:" : "") << r.backend
            << " frames=" << r.frames << '/' << expected_frames << " markers=" << r.markers;
  if (run.scene == "charuco") std::cout << " corners=" << r.corners;
  // The first frame warms the detector up and is left out.
  std::cout << " detect_ms(mean/p50/p95/max)=" << mean << '/' << percentile(r.detect_ms, 0.5) << '/'
            << percentile(r.detect_ms, 0.95) << '/' << percentile(r.detect_ms, 1.0)
            << " waits/frame=" << static_cast<double>(r.host_waits) / frames
            << " copied_bytes/frame=" << std::setprecision(0)
            << static_cast<double>(r.copied_bytes) / frames << std::setprecision(2);
  if (r.verified) {
    const double mean_px =
        r.measured_corners ? r.sum_corner_px / static_cast<double>(r.measured_corners) : 0.0;
    std::cout << " truth: missed=" << r.missed << " extra=" << r.extra
              << " misplaced=" << r.misplaced << '/' << r.expected
              << " without_pose=" << r.without_pose;
    std::cout << std::setprecision(3) << " corner_px(mean/max)=" << mean_px << '/'
              << r.max_corner_px << " max_translation=" << std::setprecision(2)
              << 100.0 * r.max_translation << "% max_rotation=" << r.max_rotation_deg << "deg";
  }
  if (r.dropped != 0) std::cout << " dropped=" << r.dropped;
  if (r.truncated_frames != 0) std::cout << " truncated_frames=" << r.truncated_frames;
  std::cout << '\n';
}

// Whether every backend's results for `scene` are those of the first, frame by frame.
bool backends_agree(const std::vector<Run>& runs, const std::string& scene) {
  const Report* first = nullptr;
  bool agree = true;
  for (const auto& run : runs) {
    if (run.scene != scene) continue;
    if (first == nullptr) {
      first = run.report.get();
    } else {
      agree = agree && run.report->results == first->results;
    }
  }
  return agree;
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;
using ImageOutput = holoscan::Output<holoscan::schema::ImageT>;
using CameraInfoOutput = holoscan::Output<hm::schema::CameraInfoT>;
using DetectionsOutput = holoscan::Output<fd::schema::FiducialDetectionsT>;

// Whether any diagnostic is an error; warnings neither fail a run nor a plan.
bool any_error(std::span<const holoscan::Diagnostic> diagnostics) {
  return std::any_of(diagnostics.begin(), diagnostics.end(), [](const holoscan::Diagnostic& d) {
    return d.severity == holoscan::DiagnosticSeverity::kError;
  });
}

int execute(const Options& opts, SignalDomain& signals) {
  holoscan::Graph graph{std::string(kAppName)};
  const auto period = std::chrono::milliseconds(opts.period_ms);
  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool display = opts.display == DisplayMode::kOn ||
                       (opts.display == DisplayMode::kAuto && display_available &&
                        FIDUCIAL_DETECTION_ENABLE_VISUALIZATION);
  if (display && !FIDUCIAL_DETECTION_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add "
        "--display false");
  }
  const bool camera_source = opts.source == Source::kCamera;
  // Scenes and recordings: every detector sees every frame, in order, for the ground truth and
  // the comparison of backends. The camera: every edge keeps only the newest sample, so nothing
  // waits for a slower consumer.
  const bool lossless = !camera_source;
  const holoscan::ConnectionOptions edge =
      lossless ? holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kBlocking}
               : holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
  const bool calibrated = opts.camera_info_file.has_value();
  // Scenes and recordings send the camera info with every frame (kEveryFrame), so no frame
  // waits for or misses it; the camera's comes at its own rate (kLatest).
  const hm::CameraInfoInputOptions camera_info_input{
      .timing = lossless ? hm::CameraInfoTiming::kEveryFrame : hm::CameraInfoTiming::kLatest};
  const std::uint32_t frames = opts.count;
  std::vector<Run> runs;
  auto remaining = std::make_shared<std::atomic<int>>(0);
  // The first detector's output and the frames it saw, for the display.
  std::optional<std::pair<const ImageOutput*, const DetectionsOutput*>> shown;

  // The calibration, read from its file (again when the run starts) and sent every second.
  std::optional<holoscan::OperatorHandle<hm::CameraInfoSourceOp>> camera_info;
  if (calibrated) {
    camera_info.emplace(graph.op<hm::CameraInfoSourceOp>(
        "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = *opts.camera_info_file}));
  }
  std::optional<holoscan::OperatorHandle<fiducial_app::CameraSourceOp>> camera;
  if (camera_source) {
    camera.emplace(graph.op<fiducial_app::CameraSourceOp>(
        "camera",
        fiducial_app::CameraSourceConfig{
            .device = opts.device, .width = opts.width, .height = opts.height, .count = frames}));
  }
  // A frame source, sending each frame with its camera info.
  std::vector<std::string> frame_sources;
  const auto frame_source = [&](const std::string& name, fiducial_app::FrameSourceConfig config) {
    config.count = frames;
    config.period = period;
    config.device_output = opts.device_input;
    auto source = graph.op<fiducial_app::FrameSourceOp>(name, std::move(config));
    graph.add_flow((*camera_info)->camera_info, source->camera_info,
                   hm::latest_camera_info_connection());
    frame_sources.push_back(name);
    return source;
  };
  std::optional<holoscan::OperatorHandle<fiducial_app::FrameSourceOp>> recording;
  if (opts.source == Source::kRecording) {
    recording.emplace(
        frame_source("recording", fiducial_app::recording_source_config(*opts.recording, frames)));
  }
  for (const bool board : {false, true}) {
    if ((board && !opts.charuco) || (!board && !opts.aruco)) continue;
    const std::string scene = board ? "charuco" : "aruco";
    const SceneKind kind{.board = board, .dictionary = opts.dictionary, .board_config = opts.board};
    const ImageOutput* images = nullptr;
    const CameraInfoOutput* image_camera_info = nullptr;
    if (camera) {
      images = &(*camera)->image;
    } else if (recording) {
      images = &(*recording)->image;
      image_camera_info = &(*recording)->image_camera_info;
    } else {
      auto source = frame_source(scene + "_source",
                                 fiducial_app::scene_source_config(kind, opts.width, opts.height));
      images = &source->image;
      image_camera_info = &source->image_camera_info;
    }
    for (const fd::Backend backend : opts.backends) {
      const std::string name = scene + "_" + std::string(fd::backend_name(backend));
      const DetectionsOutput* detections = nullptr;
      const holoscan::Input<holoscan::schema::ImageT>* input = nullptr;
      const holoscan::Input<hm::schema::CameraInfoT>* camera_info_in = nullptr;
      if (board) {
        auto detector = graph.op<fd::CharucoDetectorOp>(
            "detector_" + name,
            fd::CharucoDetectorConfig{.backend = backend,
                                      .board = opts.board,
                                      .options = {.min_marker_px = opts.min_marker_px},
                                      .estimate_pose = calibrated,
                                      .camera_info_input = camera_info_input});
        detections = &detector->detections;
        input = &detector->image;
        camera_info_in = &detector->camera_info;
      } else {
        auto detector = graph.op<fd::ArucoDetectorOp>(
            "detector_" + name,
            fd::ArucoDetectorConfig{
                .backend = backend,
                .options = {.dictionary = opts.dictionary, .min_marker_px = opts.min_marker_px},
                .estimate_pose = calibrated,
                .marker_length = opts.marker_length,
                .camera_info_input = camera_info_input});
        detections = &detector->detections;
        input = &detector->image;
        camera_info_in = &detector->camera_info;
      }
      auto report = std::make_shared<Report>();
      report->backend = std::string(fd::backend_name(backend));
      runs.push_back({scene, backend, report});
      std::optional<SceneKind> truth;
      if (opts.source == Source::kSynthetic) truth = kind;
      // The camera's reporters never finish: its source completes the run.
      const std::uint32_t expected = camera_source ? 0U : frames;
      if (expected != 0) ++*remaining;
      auto reporter = graph.op<fiducial_app::ReportOp>(
          "report_" + name, fiducial_app::ReportConfig{.report = report,
                                                       .remaining = remaining,
                                                       .expected_frames = expected,
                                                       .truth = truth,
                                                       .width = opts.width,
                                                       .height = opts.height,
                                                       .keep_results = recording.has_value()});
      graph.add_flow(*images, *input, edge);
      if (calibrated) {
        if (image_camera_info != nullptr) {
          graph.add_flow(*image_camera_info, *camera_info_in, edge);
        } else {
          graph.add_flow((*camera_info)->camera_info, *camera_info_in,
                         hm::latest_camera_info_connection());
        }
      }
      graph.add_flow(*detections, reporter->detections, edge);
      if (!shown) shown.emplace(images, detections);
    }
  }

#if FIDUCIAL_DETECTION_ENABLE_VISUALIZATION
  if (display && shown) {
    namespace viz = holoscan::visualization;
    viz::LayerStack layers;
    layers.add_image_layer(
        "video", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                     .max_width = static_cast<std::uint32_t>(opts.width),
                                     .max_height = static_cast<std::uint32_t>(opts.height)});
    // HoloViz blends RGBA image layers by their alpha, so the drawing goes over the video.
    layers.add_image_layer(
        "overlay", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8A8Unorm,
                                       .max_width = static_cast<std::uint32_t>(opts.width),
                                       .max_height = static_cast<std::uint32_t>(opts.height)});
    layers.add_text_layer(
        "labels", viz::TextLayerDesc{.max_glyphs = fd::kOverlayLabelRows * fd::kOverlayLabelBytes,
                                     .normalized_coordinates = true});
    // Draw when the overlay arrives, over the newest image, which is its own.
    const auto max_input_age = std::max(4 * period, std::chrono::milliseconds{250});
    auto visualizer = graph.op<viz::HolovizOp>(
        "holoviz", std::move(layers),
        viz::WindowSinkDesc{.title = "Fiducial Detection (" + runs.front().scene + ", " +
                                     std::string(fd::backend_name(runs.front().requested)) + ")",
                            .width = static_cast<std::uint32_t>(opts.width),
                            .height = static_cast<std::uint32_t>(opts.height)},
        viz::FrameTrigger{viz::LayerTrigger{1}}, max_input_age);
    auto overlay = graph.op<fd::FiducialOverlayOp>(
        "overlay", fd::FiducialOverlayConfig{.width = opts.width,
                                             .height = opts.height,
                                             .draw_axes = calibrated,
                                             .marker_length = opts.marker_length});
    // Best effort: the display keeps the newest sample and never holds back the detectors. The
    // frame's image must still be queued when its detections arrive: four frames cover a
    // detector up to three periods behind.
    const holoscan::ConnectionOptions newest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    const holoscan::ConnectionOptions recent{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 4U};
    graph.add_flow(*shown->first, visualizer->image_layer_input(0), recent);
    graph.add_flow(*shown->second, overlay->detections, newest);
    if (calibrated) {
      graph.add_flow((*camera_info)->camera_info, overlay->camera_info,
                     hm::latest_camera_info_connection());
    }
    graph.add_flow(overlay->overlay, visualizer->image_layer_input(1), newest);
    graph.add_flow(overlay->labels, visualizer->tensor_layer_input(2), newest);
  }
#endif

  if (lossless) {
    // One worker for the whole graph. The camera info is a second source, and the frames fan
    // out to every detector, so the automatic partitioning would give the camera info, each
    // source and each detector a worker of its own; a detector would then run while its
    // reporter's lossless queue is full, and fail on the refused emit. The camera's live edges
    // drop frames rather than wait, and keep the automatic partitioning.
    graph.partition(std::string(kAppName)).include_all();
  }
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions compile_options;
  if (opts.device_input) {
    for (const std::string& source : frame_sources) {
      compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = source, .output_port = "image", .device = holoscan::DeviceId{0}});
    }
  }
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  // Warnings on an accepted plan, as well as the reasons for a rejected one.
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) return 1;
  if (opts.compile_only) {
    // The workers the graph runs on, one per partition with operators.
    std::cout << kAppName << ": plan=OK workers=" << plan.scheduler_plan().worker_count << '\n';
    return 0;
  }
  if (!signals) {
    // A thread started before main() (profilers, sanitizers) makes the SDK refuse.
    std::cerr << kAppName << ": warning: running without managed SIGINT/SIGTERM handling: "
              << holoscan::runtime_error_message(signals.error())
              << "; an interrupt ends the process without a report\n";
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  std::cout << kAppName << ": running" << std::endl;
  // The camera without a count runs until it is stopped.
  const bool finite = frames != 0;
  bool timed_out = false;
  if (finite) {
    const auto allowed = kTimeoutAllowance +
                         std::chrono::milliseconds(2LL * frames * opts.period_ms) +
                         std::chrono::milliseconds(300LL * frames);  // Rendering, on slow boards.
    timed_out = !session.wait_for(allowed);
    if (timed_out) {
      std::cerr << kAppName << ": timed out\n";
      session.request_stop();
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
  // A stop request (SIGINT, SIGTERM, a closed window) is neither a pass nor a failure.
  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
  const bool completed = !timed_out && !any_error(session.diagnostics()) &&
                         termination == holoscan::RunTermination::kRequestedCompleted;
  bool passed = completed;
  bool any_frames = false;
  for (const Run& run : runs) {
    print_report(run, frames);
    any_frames = any_frames || run.report->frames != 0;
    if (!camera_source) passed = passed && run.report->frames == frames && run.report->passed();
    // On a CI runner, auto must have chosen cuda.
    if (run.requested == fd::Backend::kAuto && on_ci_runner()) {
      std::cout << kAppName << ": auto=" << run.report->backend << " expected=cuda ("
                << std::getenv("HOLOSCAN_CI_PLATFORM") << ")\n";
      passed = passed && run.report->backend == "cuda";
    }
  }
  if (recording) {
    // No ground truth: every backend must find what the first one finds, frame by frame.
    for (const char* scene : {"aruco", "charuco"}) {
      if ((std::string_view(scene) == "aruco" && !opts.aruco) ||
          (std::string_view(scene) == "charuco" && !opts.charuco)) {
        continue;
      }
      const bool agree = backends_agree(runs, scene);
      std::cout << kAppName << ": " << scene << " backends_agree=" << (agree ? "yes" : "NO")
                << '\n';
      passed = passed && agree;
    }
  }
  if (camera_source) {
    // The camera has no ground truth: a run that ends, by its count or a stop, with frames, is
    // done.
    const bool done = (stopped || completed) && any_frames;
    if (!any_frames) std::cout << kAppName << ": no frames came from the camera\n";
    std::cout << kAppName << ": verdict=" << (done ? "DONE" : "FAIL") << '\n';
    return done ? 0 : 1;
  }
  std::cout << kAppName << ": verdict=" << (passed ? "PASS" : stopped ? "STOPPED" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM first: the SDK refuses once other threads run.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  // Destroyed after the run, however it ends: the synthetic camera's folder goes with it.
  auto opts = parse_args(argc, argv);
  if (opts.help) {
    usage(argv[0]);
    return 0;
  }
  validate(opts);
  if (opts.source == Source::kCamera && !opts.compile_only) {
    fiducial_app::check_camera(fiducial_app::CameraSourceConfig{
        .device = opts.device, .width = opts.width, .height = opts.height});
  }
  return execute(opts, signals);
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
