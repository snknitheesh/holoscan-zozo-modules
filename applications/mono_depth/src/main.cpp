// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Mono Depth: images in, depth out, on the GPU. A source (a synthetic scene
// with known depth, JPEG and PPM files, or a webcam) feeds MonoDepthOp, whose
// depth a reporter checks, times and optionally writes as colorized PPM
// files. With metric depth, CameraInfoSourceOp sends the camera, from a
// camera_info file (written at run time for the synthetic scene). HoloViz
// can show each image beside its colorized depth, in a window or rendered
// offscreen, where the rendering is checked.

#include <cuda_runtime_api.h>
#include <stdlib.h>

#include <algorithm>
#include <array>
#include <cerrno>
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
#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/mono_depth/image_fit.hpp>
#include <holoscan/mono_depth/mono_depth_op.hpp>
#include <holoscan/mono_depth/reference.hpp>
#include <holoscan/mono_depth/stub_model.hpp>
#include <holoscan/mono_depth/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if MONO_DEPTH_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "depth_report_op.hpp"
#include "edges.hpp"
#include "image_files.hpp"
#include "image_sources.hpp"
#include "render_check_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace app = mono_depth_app;
namespace hm = holoscan::messages;
namespace md = holoscan::mono_depth;

constexpr std::string_view kAppName = "mono_depth";
constexpr std::string_view kStubModel = "stub";
// How long a run may take to stop once asked to, before the demo gives up on
// it: a stage that never returns would otherwise hold the process forever.
constexpr std::chrono::seconds kStopAllowance{30};

enum class Source { kSynthetic, kFiles, kCamera };
enum class DisplayMode { kAuto, kWindow, kOffscreen, kNone };

struct Options {
  Source source{Source::kSynthetic};
  std::filesystem::path images;
  // Found by validate() for --source files, with each file's header, read
  // once.
  std::vector<std::filesystem::path> image_files;
  std::vector<app::ImageFileInfo> image_infos;
  std::filesystem::path device{"/dev/video0"};
  std::filesystem::path camera_info;
  bool loop{false};
  std::optional<std::uint32_t> count;
  std::uint32_t period_ms{33};
  std::optional<int> width;
  std::optional<int> height;
  md::PixelFormat encoding{md::PixelFormat::kRgb8};
  bool host_input{false};
  bool still{false};
  bool realtime{false};
  std::uint32_t report_delay_ms{0};
  std::string model;
  bool metric_model{false};
  int input_width{518};
  int input_height{518};
  md::ResizeMode resize{md::ResizeMode::kLetterbox};
  bool fp16{true};
  std::optional<float> metric_scale;
  float metric_shift{0.0F};
  std::optional<float> color_min;
  std::optional<float> color_max;
  std::optional<bool> verify;
  double min_spearman{0.85};
  std::filesystem::path image_dir;
  std::filesystem::path render_file;
  std::uint32_t timeout_seconds{120};
  std::uint32_t startup_timeout_seconds{1800};
  DisplayMode display{DisplayMode::kAuto};
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

md::PixelFormat parse_encoding(std::string_view value) {
  if (value == "rgb8") return md::PixelFormat::kRgb8;
  if (value == "bgr8") return md::PixelFormat::kBgr8;
  if (value == "rgba8") return md::PixelFormat::kRgba8;
  if (value == "bgra8") return md::PixelFormat::kBgra8;
  throw std::invalid_argument("--encoding must be rgb8, bgr8, rgba8 or bgra8");
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
    if (option == "--source") {
      if (value == "synthetic") {
        opts.source = Source::kSynthetic;
      } else if (value == "files") {
        opts.source = Source::kFiles;
      } else if (value == "camera") {
        opts.source = Source::kCamera;
      } else {
        throw std::invalid_argument("--source must be synthetic, files or camera");
      }
    } else if (option == "--images") {
      opts.images = value;
    } else if (option == "--device") {
      opts.device = value;
    } else if (option == "--camera-info") {
      opts.camera_info = value;
    } else if (option == "--loop") {
      opts.loop = parse_bool(value, option);
    } else if (option == "--count") {
      opts.count = parse_number<std::uint32_t>(value, option);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--width") {
      opts.width = parse_number<int>(value, option);
    } else if (option == "--height") {
      opts.height = parse_number<int>(value, option);
    } else if (option == "--encoding") {
      opts.encoding = parse_encoding(value);
    } else if (option == "--host-input") {
      opts.host_input = parse_bool(value, option);
    } else if (option == "--still") {
      opts.still = parse_bool(value, option);
    } else if (option == "--realtime") {
      opts.realtime = parse_bool(value, option);
    } else if (option == "--report-delay-ms") {
      opts.report_delay_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--model") {
      opts.model = value;
    } else if (option == "--model-output") {
      if (value != "relative" && value != "metric") {
        throw std::invalid_argument("--model-output must be relative or metric");
      }
      opts.metric_model = value == "metric";
    } else if (option == "--input-width") {
      opts.input_width = parse_number<int>(value, option);
    } else if (option == "--input-height") {
      opts.input_height = parse_number<int>(value, option);
    } else if (option == "--resize") {
      if (value != "letterbox" && value != "stretch") {
        throw std::invalid_argument("--resize must be letterbox or stretch");
      }
      opts.resize = value == "stretch" ? md::ResizeMode::kStretch : md::ResizeMode::kLetterbox;
    } else if (option == "--fp16") {
      opts.fp16 = parse_bool(value, option);
    } else if (option == "--metric-scale") {
      opts.metric_scale = parse_number<float>(value, option);
    } else if (option == "--metric-shift") {
      opts.metric_shift = parse_number<float>(value, option);
    } else if (option == "--color-min") {
      opts.color_min = parse_number<float>(value, option);
    } else if (option == "--color-max") {
      opts.color_max = parse_number<float>(value, option);
    } else if (option == "--verify") {
      opts.verify = parse_bool(value, option);
    } else if (option == "--min-spearman") {
      opts.min_spearman = parse_number<double>(value, option);
    } else if (option == "--image-dir") {
      opts.image_dir = value;
    } else if (option == "--render-file") {
      opts.render_file = value;
    } else if (option == "--timeout-seconds") {
      opts.timeout_seconds = parse_number<std::uint32_t>(value, option);
    } else if (option == "--startup-timeout-seconds") {
      opts.startup_timeout_seconds = parse_number<std::uint32_t>(value, option);
    } else if (option == "--display") {
      if (value == "auto") {
        opts.display = DisplayMode::kAuto;
      } else if (value == "window") {
        opts.display = DisplayMode::kWindow;
      } else if (value == "offscreen") {
        opts.display = DisplayMode::kOffscreen;
      } else if (value == "none") {
        opts.display = DisplayMode::kNone;
      } else {
        throw std::invalid_argument("--display must be auto, window, offscreen or none");
      }
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
      << " --model PATH|stub [options]\n"
         "Model:\n"
         "  --model PATH|stub          A Depth Anything V2 .onnx export (its TensorRT engine is\n"
         "                             built once and cached beside it) or a TensorRT engine;\n"
         "                             'stub' writes and uses a stand-in model with a known\n"
         "                             response, which needs no download\n"
         "  --model-output relative|metric\n"
         "                             What the model gives (default: relative inverse depth)\n"
         "  --input-width N --input-height N\n"
         "                             The model's input size (default: 518x518)\n"
         "  --resize letterbox|stretch How other aspect ratios fill it (default: letterbox)\n"
         "  --fp16 true|false          Build the engine with FP16 allowed (default: true)\n"
         "  --metric-scale F --metric-shift F\n"
         "                             Metric depth from relative inverse depth d, as\n"
         "                             1 / z = scale * d + shift (default: none)\n"
         "  --camera-info FILE         With metric depth: the camera's ROS camera_info file,\n"
         "                             for image files and the camera (the synthetic scene's\n"
         "                             is written for each run)\n"
         "  --color-min F --color-max F\n"
         "                             The inverse depths at the colormap's far and near ends\n"
         "                             (default: each frame's range)\n"
         "Input:\n"
         "  --source synthetic|files|camera\n"
         "                             The synthetic scene, image files, or a V4L2 webcam\n"
         "                             (default: synthetic)\n"
         "  --images PATH              A JPEG or PPM file, or a folder of them, in name order\n"
         "  --device PATH              The webcam (default: /dev/video0)\n"
         "  --loop true|false          Repeat the files until --count frames (default: false)\n"
         "  --count N                  Frames (default: 100 synthetic, every file once,\n"
         "                             camera until Ctrl+C)\n"
         "  --period-ms N              Source period, 1..1000 ms (default: 33)\n"
         "  --encoding rgb8|bgr8|rgba8|bgra8\n"
         "                             Image encoding (default: rgb8; files: rgb8 or bgr8)\n"
         "  --width N --height N       Synthetic frame size (default: 1280x720), or a YUYV\n"
         "                             size the camera offers (default: 640x480)\n"
         "  --host-input true|false    Send synthetic frames in host memory (default: false)\n"
         "  --still true|false         Show the synthetic scene's first frame throughout\n"
         "  --realtime true|false      Connect synthetic frames as the camera, keeping only\n"
         "                             the newest sample on every edge (default: false)\n"
         "Output:\n"
         "  --verify true|false        Check the depth: against the host reference with\n"
         "                             --model stub, against the scene's true depth with a\n"
         "                             real model (default: true for synthetic frames)\n"
         "  --min-spearman F           The rank correlation a real model must reach on the\n"
         "                             synthetic scene (default: 0.85)\n"
         "  --image-dir DIR            Write each frame's colorized depth as\n"
         "                             DIR/depth_NNNNNN.ppm\n"
         "  --display auto|window|offscreen|none\n"
         "                             Show each image beside its colorized depth in a HoloViz\n"
         "                             window, or render it offscreen (default: auto, a window\n"
         "                             for the camera when a display is available)\n"
         "  --render-file FILE         Offscreen: write the newest rendered frame as a PPM\n"
         "  --compile-only true|false  Build and check the graph without running it\n"
         "  --timeout-seconds N        Fail when no result arrives for N seconds plus two\n"
         "                             source periods (default: 120)\n"
         "  --startup-timeout-seconds N\n"
         "                             The same before the first result, while the estimator\n"
         "                             loads its engine, which is built before the run starts\n"
         "                             (default: 1800)\n"
         "  --report-delay-ms N        Slow the reporter down by N ms per frame, 0..10000\n"
         "                             (default: 0)\n";
}

bool is_stub(const Options& opts) { return opts.model == kStubModel; }

// Synthetic frames, or the camera's once validate() has set its size.
int frame_width(const Options& opts) { return opts.width.value_or(1280); }
int frame_height(const Options& opts) { return opts.height.value_or(720); }

app::SyntheticImageSourceConfig synthetic_source_config(const Options& opts) {
  return app::SyntheticImageSourceConfig{.width = frame_width(opts),
                                         .height = frame_height(opts),
                                         .count = opts.count.value_or(0),
                                         .period = std::chrono::milliseconds(opts.period_ms),
                                         .format = opts.encoding,
                                         .device_output = !opts.host_input,
                                         .still = opts.still};
}

app::ImageFileSourceConfig file_source_config(const Options& opts) {
  return app::ImageFileSourceConfig{.files = opts.image_files,
                                    .infos = opts.image_infos,
                                    .count = opts.count.value_or(0),
                                    .loop = opts.loop,
                                    .period = std::chrono::milliseconds(opts.period_ms),
                                    .bgr = opts.encoding == md::PixelFormat::kBgr8};
}

app::CameraSourceConfig camera_source_config(const Options& opts) {
  return app::CameraSourceConfig{.device = opts.device,
                                 .width = frame_width(opts),
                                 .height = frame_height(opts),
                                 .count = opts.count.value_or(0)};
}

bool shows(const Options& opts) {
  return opts.display == DisplayMode::kWindow || opts.display == DisplayMode::kOffscreen;
}

// The estimator's settings; the stub's model path is filled in by execute().
md::MonoDepthConfig estimator_config(const Options& opts) {
  md::MonoDepthConfig config{.model = is_stub(opts) ? std::filesystem::path("mono_depth_stub.onnx")
                                                    : std::filesystem::path(opts.model),
                             .input_width = opts.input_width,
                             .input_height = opts.input_height,
                             .resize_mode = opts.resize,
                             .model_output = opts.metric_model
                                                 ? md::ModelOutput::kMetricDepth
                                                 : md::ModelOutput::kRelativeInverseDepth,
                             .colorize = !opts.image_dir.empty() || shows(opts),
                             .fp16 = opts.fp16};
  if (opts.metric_scale) {
    config.metric_from_relative =
        md::InverseDepthAffine{.scale = *opts.metric_scale, .shift = opts.metric_shift};
  }
  if (opts.color_min || opts.color_max) {
    config.color_range =
        std::array<float, 2>{opts.color_min.value_or(0.0F), opts.color_max.value_or(0.0F)};
  }
  // The output buffers fit the largest image the source gives.
  if (opts.source == Source::kFiles) {
    for (const app::ImageFileInfo& info : opts.image_infos) {
      config.max_width = std::max(config.max_width, info.width);
      config.max_height = std::max(config.max_height, info.height);
    }
  } else if (opts.width && opts.height) {
    config.max_width = std::max(config.max_width, *opts.width);
    config.max_height = std::max(config.max_height, *opts.height);
  }
  return config;
}

void validate(Options& opts) {
  if (opts.model.empty()) {
    throw std::invalid_argument(
        "--model is required: a Depth Anything .onnx file, an engine, or "
        "stub");
  }
  if (!is_stub(opts)) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(opts.model, error)) {
      throw std::invalid_argument("--model: no such file: " + opts.model +
                                  "; see the README to prepare the model, or use --model stub");
    }
  }
  if (opts.period_ms == 0 || opts.period_ms > 1000 || opts.timeout_seconds == 0 ||
      opts.startup_timeout_seconds == 0) {
    throw std::invalid_argument(
        "--period-ms must be 1..1000, and --timeout-seconds and --startup-timeout-seconds "
        "positive");
  }
  if (opts.metric_scale && opts.metric_model) {
    throw std::invalid_argument(
        "--metric-scale converts a relative model's output; a metric "
        "model gives meters already");
  }
  if (std::chrono::milliseconds(opts.report_delay_ms) > app::kMaxReportDelay) {
    throw std::invalid_argument("--report-delay-ms must be 0..10000");
  }
  const bool metric = opts.metric_model || opts.metric_scale.has_value();
  if (!opts.camera_info.empty()) {
    if (!metric) {
      throw std::invalid_argument(
          "--camera-info goes with metric depth (--model-output metric or --metric-scale)");
    }
    if (opts.source == Source::kSynthetic) {
      throw std::invalid_argument(
          "--camera-info is for image files and the camera; the synthetic scene's is written "
          "for each run");
    }
    // Read now, so that a file that cannot be used fails before the run.
    try {
      (void)hm::read_camera_info_yaml(opts.camera_info);
    } catch (const std::exception& error) {
      throw std::invalid_argument(std::string("--camera-info: ") + error.what());
    }
  } else if (metric && opts.source != Source::kSynthetic) {
    throw std::invalid_argument(
        "metric depth needs the camera: give its camera_info file with --camera-info");
  }
  if (opts.color_min.has_value() != opts.color_max.has_value()) {
    throw std::invalid_argument("--color-min and --color-max go together");
  }
  if (!(opts.min_spearman >= -1.0 && opts.min_spearman <= 1.0)) {
    throw std::invalid_argument("--min-spearman must be in [-1, 1]");
  }
  if (opts.source != Source::kSynthetic && (opts.still || opts.realtime || opts.host_input)) {
    throw std::invalid_argument("--still, --realtime and --host-input apply to synthetic frames");
  }
  if (opts.source == Source::kFiles) {
    if (opts.images.empty()) {
      throw std::invalid_argument("--source files needs --images PATH");
    }
    if (opts.width || opts.height) {
      throw std::invalid_argument("--width and --height apply to synthetic frames and the camera");
    }
    if (opts.encoding != md::PixelFormat::kRgb8 && opts.encoding != md::PixelFormat::kBgr8) {
      throw std::invalid_argument("image files are decoded as rgb8 or bgr8");
    }
    if (opts.verify.value_or(false) && !is_stub(opts)) {
      throw std::invalid_argument("--verify with image files needs --model stub");
    }
    try {
      opts.image_files = app::list_image_files(opts.images);
      // Each file's header, read here once: a file that cannot be used fails
      // before the graph is built.
      for (const auto& file : opts.image_files) {
        opts.image_infos.push_back(app::read_image_info(file));
      }
    } catch (const std::runtime_error& error) {
      throw std::invalid_argument(error.what());
    }
    if (!opts.count) {
      opts.count = static_cast<std::uint32_t>(opts.image_files.size());
    }
    if (*opts.count == 0) {
      throw std::invalid_argument("--count must be positive");
    }
    opts.verify = opts.verify.value_or(is_stub(opts));
    app::validate_config(file_source_config(opts));
  } else if (opts.source == Source::kCamera) {
    if (!opts.images.empty() || opts.loop) {
      throw std::invalid_argument("--images and --loop need --source files");
    }
    if (opts.verify.value_or(false)) {
      throw std::invalid_argument("--verify does not apply to the camera");
    }
    if (opts.encoding != md::PixelFormat::kRgb8) {
      throw std::invalid_argument("the camera gives rgb8 frames in host memory");
    }
    if (opts.count && *opts.count == 0) {
      throw std::invalid_argument("--count must be positive");
    }
    opts.verify = false;
    opts.width = opts.width.value_or(640);
    opts.height = opts.height.value_or(480);
    app::check_camera(camera_source_config(opts));
  } else {
    if (!opts.images.empty() || opts.loop) {
      throw std::invalid_argument("--images and --loop need --source files");
    }
    opts.verify = opts.verify.value_or(true);
    if (!opts.count) {
      opts.count = 100;
    }
    if (*opts.count == 0) {
      throw std::invalid_argument("--count must be positive");
    }
    app::validate_config(synthetic_source_config(opts));
  }
  if (!opts.image_dir.empty()) {
    std::error_code error;
    if (!std::filesystem::is_directory(opts.image_dir, error)) {
      throw std::invalid_argument("--image-dir is not a directory: " + opts.image_dir.string() +
                                  (error ? " (" + error.message() + ")" : ""));
    }
  }
  md::validate_config(estimator_config(opts));

  // The display: a window for the camera when a display is available, unless set.
  if (opts.display == DisplayMode::kAuto) {
    opts.display =
        MONO_DEPTH_ENABLE_VISUALIZATION && opts.source == Source::kCamera && display_available()
            ? DisplayMode::kWindow
            : DisplayMode::kNone;
  }
  if (shows(opts)) {
    if (opts.source == Source::kFiles || opts.encoding != md::PixelFormat::kRgb8) {
      throw std::invalid_argument("--display shows rgb8 synthetic frames and the camera");
    }
    if (!MONO_DEPTH_ENABLE_VISUALIZATION) {
      throw std::invalid_argument("--display needs a build with visualization enabled");
    }
    if (opts.display == DisplayMode::kWindow && !display_available() && !opts.compile_only) {
      throw std::invalid_argument(
          "the window needs a display server; set DISPLAY, or use --display offscreen or none");
    }
  }
  if (!opts.render_file.empty() && opts.display != DisplayMode::kOffscreen) {
    throw std::invalid_argument("--render-file needs --display offscreen");
  }
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
  return values[index];
}

// GPU memory free now, in bytes; 0 if it cannot be read.
std::size_t gpu_free_bytes() {
  std::size_t free = 0;
  std::size_t total = 0;
  if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return free;
}

#if MONO_DEPTH_ENABLE_VISUALIZATION
// The offscreen frame expected for synthetic frame `frame_id` with the stub
// model: the image on the left, its colorized depth on the right, opaque.
std::vector<std::uint8_t> expected_render(const Options& opts, const md::MonoDepthConfig& config,
                                          std::uint64_t frame_id) {
  const int width = frame_width(opts);
  const int height = frame_height(opts);
  const auto scene_frame =
      opts.still || frame_id == 0 ? 0U : static_cast<std::uint32_t>(frame_id - 1U);
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3U);
  md::render_depth_scene(md::make_depth_scene(width, height, scene_frame), md::PixelFormat::kRgb8,
                         rgb.data(), static_cast<std::size_t>(width) * 3U);
  const md::ImageFit fit =
      md::make_fit(width, height, config.input_width, config.input_height, config.input_width,
                   config.input_height, config.resize_mode);
  const std::vector<float> input =
      md::reference_preprocess(rgb.data(), static_cast<std::size_t>(width) * 3U,
                               md::PixelFormat::kRgb8, fit, config.normalization);
  const md::InverseDepthAffine* affine =
      config.metric_from_relative ? &*config.metric_from_relative : nullptr;
  const md::ReferenceDepth depth =
      md::reference_postprocess(md::reference_stub(input, config.input_width, config.input_height),
                                fit, config.model_output, affine);
  const std::array<float, 2> range =
      config.color_range.value_or(md::reference_range(depth.inverse));
  const std::vector<std::uint8_t> color = md::reference_colorize(depth.inverse, range[0], range[1]);
  std::vector<std::uint8_t> frame(static_cast<std::size_t>(2 * width) * height * 4U);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t source = static_cast<std::size_t>(y) * width + x;
      std::uint8_t* left = frame.data() + (static_cast<std::size_t>(y) * 2 * width + x) * 4U;
      std::uint8_t* right = left + static_cast<std::size_t>(width) * 4U;
      left[0] = rgb[3 * source];
      left[1] = rgb[3 * source + 1];
      left[2] = rgb[3 * source + 2];
      left[3] = 255;
      std::copy_n(color.data() + 4 * source, 4, right);
    }
  }
  return frame;
}
#endif

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

// Where `--model stub` writes the stub model, whose engines are cached beside
// it: $XDG_CACHE_HOME/holoscan-mono-depth, ~/.cache/holoscan-mono-depth, or
// the temporary folder's. The model's file name carries a digest of its
// contents, so a changed stub never meets an engine built from another.
std::filesystem::path stub_directory() {
  std::filesystem::path base;
  if (env_set("XDG_CACHE_HOME")) {
    base = std::getenv("XDG_CACHE_HOME");
  } else if (env_set("HOME")) {
    base = std::filesystem::path(std::getenv("HOME")) / ".cache";
  } else {
    base = std::filesystem::temp_directory_path();
  }
  const std::filesystem::path directory = base / "holoscan-mono-depth";
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    throw std::runtime_error("cannot make " + directory.string() +
                             " for the stub model: " + error.message());
  }
  return directory;
}

// The synthetic scene's camera, as CameraInfo: an undistorted pinhole. The
// scene casts each pixel's ray through its center, at (u + 0.5, v + 0.5) from
// the image's corner, and CameraInfo puts pixel centers at integer
// coordinates.
hm::CameraInfo synthetic_camera(int width, int height) {
  const md::DepthScene scene = md::make_depth_scene(width, height, 0);
  hm::CameraInfo info;
  info.header.frame_id = "mono_depth_camera";
  info.width = static_cast<std::uint32_t>(width);
  info.height = static_cast<std::uint32_t>(height);
  info.state = hm::CameraState::kUndistorted;
  info.fx = scene.focal;
  info.fy = scene.focal;
  info.cx = static_cast<double>(scene.cx) - 0.5;
  info.cy = static_cast<double>(scene.cy) - 0.5;
  return info;
}

// The synthetic camera's camera_info file, written for CameraInfoSourceOp to
// read, in a folder of its own that mkdtemp() names unpredictably, so that
// nothing can be planted where the file goes. Removed with this object, on
// every path out of the run.
class SyntheticCameraInfoFile {
 public:
  explicit SyntheticCameraInfoFile(const hm::CameraInfo& camera) {
    std::string folder =
        (std::filesystem::temp_directory_path() / (std::string(kAppName) + "_XXXXXX")).string();
    if (::mkdtemp(folder.data()) == nullptr) {
      throw std::system_error(errno, std::generic_category(),
                              "cannot make a temporary folder for the camera_info file");
    }
    folder_ = folder;
    path_ = folder_ / "camera_info.yaml";
    try {
      hm::write_camera_info_yaml(camera, path_);
    } catch (...) {
      remove();
      throw;
    }
  }
  ~SyntheticCameraInfoFile() { remove(); }
  SyntheticCameraInfoFile(const SyntheticCameraInfoFile&) = delete;
  SyntheticCameraInfoFile& operator=(const SyntheticCameraInfoFile&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

 private:
  void remove() noexcept {
    std::error_code ignored;
    std::filesystem::remove_all(folder_, ignored);
  }

  std::filesystem::path folder_;
  std::filesystem::path path_;
};

int execute(const Options& opts, SignalDomain& signals) {
  md::MonoDepthConfig estimator = estimator_config(opts);
  if (is_stub(opts)) {
    // Written to the cache folder, where its engines are cached beside it.
    estimator.model = md::write_stub_model(stub_directory(), opts.input_width, opts.input_height);
  }

  holoscan::Graph graph{std::string(kAppName)};
  // Frames the run expects; 0 for a camera that runs until stopped.
  const std::uint32_t frames = opts.count.value_or(0);
  auto report = std::make_shared<app::DepthReport>();
  // The camera runs in real time, the other sources losslessly (edges.hpp),
  // unless asked otherwise.
  const bool realtime = opts.source == Source::kCamera || opts.realtime;
  const bool metric = md::gives_metric_depth(estimator);
  // Lossless, every frame's operators run on one worker: the SDK 5.0 gives
  // each automatic partition a worker of its own, and a source on its own
  // worker would run ahead of the estimator and the reporter and meet full
  // queues (see holoscan-messages' README). With a display, whose edges drop
  // their oldest sample and never hold anything back, HoloViz and the render
  // check run on a second worker: with HoloViz's offscreen output on the
  // frames' worker, the SDK 5.0 often crashes as it tears the run down (a
  // segmentation fault in its output pools, after a correct run). The SDK
  // refuses that split for a graph with the camera info source (metric
  // depth), which therefore runs on one worker, display included. In real time
  // the automatic partitioning stays: its stages overlap, and its drop-oldest
  // edges lose frames rather than wait.
  std::optional<holoscan::PartitionHandle> frames_partition;
  std::optional<holoscan::PartitionHandle> display_partition;
  if (!realtime) {
    if (shows(opts) && !metric) {
      frames_partition.emplace(graph.partition(std::string(kAppName)));
      display_partition.emplace(graph.partition("display"));
    } else {
      graph.partition(std::string(kAppName)).include_all();
    }
  }
  const holoscan::Output<holoscan::schema::ImageT>* image_out = nullptr;
  if (opts.source == Source::kSynthetic) {
    auto source = graph.op<app::SyntheticImageSourceOp>("source", synthetic_source_config(opts));
    image_out = &source->image;
    if (frames_partition) frames_partition->add(source);
  } else if (opts.source == Source::kFiles) {
    auto source = graph.op<app::ImageFileSourceOp>("source", file_source_config(opts));
    image_out = &source->image;
    if (frames_partition) frames_partition->add(source);
  } else {
    image_out = &graph.op<app::CameraSourceOp>("source", camera_source_config(opts))->image;
  }
  auto estimator_op = graph.op<md::MonoDepthOp>("estimator", estimator);
  if (frames_partition) frames_partition->add(estimator_op);

  const bool verify = opts.verify.value_or(false);
  app::Verify check = app::Verify::kNone;
  if (verify) {
    check = is_stub(opts) ? app::Verify::kReference : app::Verify::kOrdering;
  }
  // Metric depth needs the camera: the synthetic scene's, written for this
  // run (CameraInfoSourceOp reads the file again when the run starts, so it
  // stays until the run is over), or --camera-info.
  std::optional<hm::CameraInfo> camera;
  std::optional<SyntheticCameraInfoFile> synthetic_file;
  std::filesystem::path camera_info_file = opts.camera_info;
  if (metric && opts.source == Source::kSynthetic) {
    camera = synthetic_camera(frame_width(opts), frame_height(opts));
    camera_info_file = synthetic_file.emplace(*camera).path();
  }
  auto reporter = graph.op<app::DepthReportOp>(
      "report", app::DepthReportConfig{.report = report,
                                       .expected_frames = frames,
                                       .realtime = realtime,
                                       .delay = std::chrono::milliseconds(opts.report_delay_ms),
                                       .verify = check,
                                       .estimator = estimator,
                                       .tolerance = opts.fp16 ? 5e-3F : 2e-5F,
                                       .scene_width = frame_width(opts),
                                       .scene_height = frame_height(opts),
                                       .still = opts.still,
                                       .min_spearman = opts.min_spearman,
                                       .files = opts.source == Source::kFiles
                                                    ? opts.image_files
                                                    : std::vector<std::filesystem::path>{},
                                       .print = opts.source == Source::kFiles,
                                       .image_dir = opts.image_dir,
                                       .camera = camera,
                                       .allowed_before_camera_info = metric ? 1U : 0U});
  if (frames_partition) frames_partition->add(reporter);

  // Lossless, every frame's depth is estimated and reported, in order. In
  // real time, the estimator takes the newest frame and the reporter the
  // newest depth; the reporter pairs the depth with its image by capture
  // time, so its images keep a few frames.
  const holoscan::ConnectionOptions edge = realtime ? app::newest() : app::lossless();
  graph.add_flow(*image_out, estimator_op->image, edge);
  graph.add_flow(estimator_op->depth, reporter->depth, edge);
  if (metric) {
    auto camera_info = graph.op<hm::CameraInfoSourceOp>(
        "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = camera_info_file});
    if (frames_partition) frames_partition->add(camera_info);
    graph.add_flow(camera_info->camera_info, estimator_op->camera_info,
                   hm::latest_camera_info_connection());
    graph.add_flow(estimator_op->metric_depth, reporter->metric_depth, edge);
    graph.add_flow(estimator_op->depth_camera_info, reporter->depth_camera_info, edge);
  }
  if (check == app::Verify::kReference) {
    graph.add_flow(*image_out, reporter->image, realtime ? app::recent_frames() : app::lossless());
  }
  if (!opts.image_dir.empty()) {
    graph.add_flow(estimator_op->colorized, reporter->colorized, edge);
  }

  auto render_check = std::make_shared<app::RenderCheck>();
#if MONO_DEPTH_ENABLE_VISUALIZATION
  if (shows(opts)) {
    namespace viz = holoscan::visualization;
    const auto width = static_cast<std::uint32_t>(frame_width(opts));
    const auto height = static_cast<std::uint32_t>(frame_height(opts));
    viz::LayerStack layers;
    // The image on the left half of the frame, its depth on the right.
    layers.add_image_layer("video",
                           viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                               .max_width = width,
                                               .max_height = height,
                                               .view = viz::ViewTransform{.scale_x = 0.5F}});
    layers.add_image_layer("depth", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8A8Unorm,
                                                        .max_width = width,
                                                        .max_height = height,
                                                        .view = viz::ViewTransform{
                                                            .offset_x = 0.5F, .scale_x = 0.5F}});
    const auto max_input_age =
        std::max(4 * std::chrono::milliseconds(opts.period_ms), std::chrono::milliseconds{250});
    viz::SinkDesc sink =
        viz::WindowSinkDesc{.title = "Mono Depth", .width = 2 * width, .height = height};
    if (opts.display == DisplayMode::kOffscreen) {
      sink = viz::CudaSinkDesc{
          .width = 2 * width, .height = height, .format = viz::ImageFormat::kR8G8B8A8Unorm};
    }
    // Draw when the depth arrives, over the newest image, which is its own.
    auto visualizer =
        graph.op<viz::HolovizOp>("holoviz", std::move(layers), std::move(sink),
                                 viz::FrameTrigger{viz::LayerTrigger{1}}, max_input_age);
    if (display_partition) display_partition->add(visualizer);
    // Drop-oldest in either mode: the display never holds back the estimator.
    // HoloViz pairs the depth with the newest image no newer than it, which
    // must still be queued when the depth arrives, an estimator's latency
    // later: that edge keeps a few (edges.hpp).
    graph.add_flow(*image_out, visualizer->image_layer_input(0), app::recent_frames());
    graph.add_flow(estimator_op->colorized, visualizer->image_layer_input(1), app::newest());
    if (opts.display == DisplayMode::kOffscreen) {
      // The synthetic scene through the stub has a known rendering, frame by
      // frame. Lossless only: in real time HoloViz renders a few of many
      // frames, too far apart to tell which.
      std::function<std::vector<std::uint8_t>(std::uint64_t)> expected;
      if (is_stub(opts) && opts.source == Source::kSynthetic && !realtime) {
        expected = [opts, estimator](std::uint64_t frame_id) {
          return expected_render(opts, estimator, frame_id);
        };
      }
      auto checker = graph.op<app::RenderCheckOp>(
          "render_check", app::RenderCheckConfig{.check = render_check,
                                                 .width = 2 * frame_width(opts),
                                                 .height = frame_height(opts),
                                                 .expected = std::move(expected),
                                                 .last_id = frames,
                                                 .write = opts.render_file});
      graph.add_flow(visualizer->output(), checker->frame, app::newest());
      if (display_partition) display_partition->add(checker);
    }
  }
#endif
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // GPU outputs are on device 0.
  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  on_gpu("estimator", "depth");
  if (estimator.colorize) {
    on_gpu("estimator", "colorized");
  }
  if (metric) {
    on_gpu("estimator", "metric_depth");
  }
  if (opts.source == Source::kFiles || (opts.source == Source::kSynthetic && !opts.host_input)) {
    on_gpu("source", "image");
  }
  if (opts.display == DisplayMode::kOffscreen) {
    on_gpu("holoviz", "output");
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
    // The workers the graph runs on, one per partition.
    std::cout << kAppName << ": plan=OK workers=" << plan.scheduler_plan().worker_count << '\n';
    return 0;
  }
  if (!signals) {
    // Profilers, sanitizers and preloaded libraries can start a thread before
    // main(); the SDK then refuses to take over SIGINT and SIGTERM.
    std::cerr << kAppName << ": warning: running without managed SIGINT/SIGTERM handling: "
              << holoscan::runtime_error_message(signals.error());
    if (signals.error().code == holoscan::ErrorCode::kInvalidArgument) {
      std::cerr << " (another thread was already running)";
    }
    std::cerr << "; an interrupt ends the process without a report\n";
  }
  // Build the TensorRT engine now rather than in start(): see prepare_engine().
  md::prepare_engine(estimator);
  // The GPU memory the run takes: free memory before it, less the least free
  // during it. Other processes on the GPU change it too.
  const std::size_t free_before = gpu_free_bytes();
  std::size_t least_free = free_before;
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  // The run times out when no result arrives for this long. Before the first
  // one, the estimator may still be loading its engine. The sums cannot
  // overflow: the period is at most 1000 ms (validate()).
  const auto margin = std::chrono::milliseconds(2 * opts.period_ms);
  const auto patience =
      std::chrono::steady_clock::duration(std::chrono::seconds(opts.timeout_seconds) + margin);
  const auto startup_patience = std::chrono::steady_clock::duration(
      std::chrono::seconds(opts.startup_timeout_seconds) + margin);
  std::uint32_t seen = 0;
  auto last_progress = std::chrono::steady_clock::now();
  std::optional<std::uint32_t> timed_out_after;
  while (session.is_running()) {
    std::this_thread::sleep_for(10ms);
    if (free_before != 0) {
      least_free = std::min(least_free, gpu_free_bytes());
    }
    const auto now = std::chrono::steady_clock::now();
    if (const std::uint32_t reported = report->frames.load(); reported != seen) {
      seen = reported;
      last_progress = now;
    } else if (now - last_progress >= (seen == 0 ? startup_patience : patience)) {
      timed_out_after = seen == 0 ? opts.startup_timeout_seconds : opts.timeout_seconds;
      break;
    }
  }
  if (timed_out_after) {
    std::cerr << kAppName << ": timed out: no result for " << *timed_out_after << " s\n";
    session.request_stop();
  }
  if (!session.wait_for(kStopAllowance)) {
    // A stage that never returns (a hung CUDA or TensorRT wait) would hold the
    // session, and its destructor, forever: give the verdict and leave.
    std::cerr << kAppName << ": the run did not stop within " << kStopAllowance.count() << " s\n";
    std::cout << kAppName << ": verdict=FAIL" << std::endl;
    std::_Exit(1);
  }
  session.wait();
  const holoscan::RunTermination termination = session.termination();
  // The SDK logs a failed run's diagnostics itself, unless its report or its
  // logger is turned down; show those of any other run, such as the warning a
  // stop request leaves.
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

  // A stop request (SIGINT, SIGTERM, a closed window) ends the run early;
  // that is not a failure, but not a pass either. For a live camera it is the
  // normal end.
  const bool stopped = !timed_out_after && termination == holoscan::RunTermination::kStopped;
  // In real time, the frames that came while the estimator or the reporter
  // was busy are skipped; the last one must arrive. Lossless, every frame
  // must, but for the first with metric depth, should it come before the
  // first camera info (CameraInfoSourceOp sends when the run starts, a source
  // period before the first frame).
  const bool all_frames = realtime ? report->last_frame_id >= frames
                                   : report->frames + report->before_camera_info == frames &&
                                         report->before_camera_info <= (metric ? 1U : 0U);
  // Offscreen, at least one frame was rendered, and every one checked matched.
  const bool rendered = opts.display != DisplayMode::kOffscreen ||
                        (render_check->frames > 0 && render_check->failed == 0 &&
                         (opts.render_file.empty() || render_check->written));
  const bool passed = !timed_out_after && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted && all_frames &&
                      report->passed() && rendered;
  const std::vector<double>& latencies = report->latencies_ms;
  double mean = 0.0;
  for (const double value : latencies) {
    mean += value / static_cast<double>(latencies.size());
  }
  std::cout << std::fixed << std::setprecision(3) << kAppName
            << ": frames=" << report->frames.load();
  if (frames != 0) {
    std::cout << '/' << frames;
  }
  if (!latencies.empty()) {
    std::cout << " latency_ms: mean=" << mean << " p50=" << percentile(latencies, 0.5)
              << " p99=" << percentile(latencies, 0.99)
              << " max=" << *std::max_element(latencies.begin(), latencies.end());
  }
  if (free_before != 0) {
    std::cout << " gpu_memory_mib=" << std::setprecision(0)
              << static_cast<double>(free_before - least_free) / (1024.0 * 1024.0)
              << std::setprecision(3);
  }
  if (report->frames != 0) {
    std::cout << " inverse_depth=" << report->min_depth << ".." << report->max_depth;
    if (metric) {
      std::cout << " meters=" << report->min_meters << ".." << report->max_meters;
    }
  }
  if (!opts.image_dir.empty()) {
    std::cout << " files_written=" << report->files_written;
  }
  if (report->frame_gaps != 0) {
    std::cout << " frame_gaps=" << report->frame_gaps;
  }
  if (report->skipped != 0) {
    std::cout << " skipped=" << report->skipped;
  }
  if (metric) {
    std::cout << " before_camera_info=" << report->before_camera_info
              << " camera_info: checked=" << report->camera_info_checked
              << " failed=" << report->camera_info_failed;
  }
  if (report->reference_checked != 0) {
    std::cout << " reference: checked=" << report->reference_checked
              << " failed=" << report->reference_failed << std::scientific << std::setprecision(2)
              << " max_error=" << report->max_reference_error << std::fixed << std::setprecision(3);
  }
  if (report->ordering_checked != 0) {
    std::cout << " ordering: checked=" << report->ordering_checked
              << " failed=" << report->ordering_failed << std::setprecision(4)
              << " spearman_min=" << report->min_spearman
              << " spearman_mean=" << report->sum_spearman / report->ordering_checked
              << std::setprecision(3);
  }
  if (opts.display == DisplayMode::kOffscreen) {
    std::cout << " rendered=" << render_check->frames.load();
    if (render_check->checked != 0) {
      std::cout << " render: checked=" << render_check->checked
                << " failed=" << render_check->failed << std::setprecision(4)
                << " worst_mismatch=" << render_check->worst_fraction << std::setprecision(3);
    }
  }
  std::cout << '\n'
            << kAppName << ": verdict="
            << (passed    ? "PASS"
                : stopped ? "STOPPED"
                          : "FAIL")
            << '\n';
  return passed || (stopped && frames == 0) ? 0 : 1;
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
