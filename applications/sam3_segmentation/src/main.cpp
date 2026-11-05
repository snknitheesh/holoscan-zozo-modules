// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// SAM 3 Segmentation: images in, every instance of every prompt out, on the
// GPU. A source (a synthetic scene with known shapes, JPEG and PPM files, or
// a webcam) feeds Sam3SegmentationOp, whose instances a reporter checks,
// times, prints and optionally writes over their images as PPM files.
// HoloViz can show each image under its instances (Sam3OverlayOp), in a
// window or rendered offscreen, where the rendering is checked. With
// --interactive, new prompts typed on the standard input replace the old
// ones while it runs.

#include <cuda_runtime_api.h>

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
#include <holoscan/sam3/pixel_math.hpp>
#include <holoscan/sam3/prompts.hpp>
#include <holoscan/sam3/reference.hpp>
#include <holoscan/sam3/sam3_segmentation_op.hpp>
#include <holoscan/sam3/stub_model.hpp>
#include <holoscan/sam3/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if SAM3_SEGMENTATION_ENABLE_VISUALIZATION
#include <holoscan/sam3/sam3_overlay_op.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "edges.hpp"
#include "image_files.hpp"
#include "image_sources.hpp"
#include "render_check_op.hpp"
#include "segmentation_report_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace app = sam3_app;
namespace s3 = holoscan::sam3;

constexpr std::string_view kAppName = "sam3_segmentation";
constexpr std::string_view kStubModel = "stub";
constexpr std::uint8_t kFillAlpha = 110;

enum class Source { kSynthetic, kFiles, kCamera };
enum class DisplayMode { kAuto, kWindow, kOffscreen, kNone };

struct Options {
  Source source{Source::kSynthetic};
  std::filesystem::path images;
  // Found by validate() for --source files.
  std::vector<std::filesystem::path> image_files;
  std::filesystem::path device{"/dev/video0"};
  bool loop{false};
  std::optional<std::uint32_t> count;
  std::uint32_t period_ms{33};
  std::optional<int> width;
  std::optional<int> height;
  s3::PixelFormat encoding{s3::PixelFormat::kRgb8};
  bool host_input{false};
  bool still{false};
  bool realtime{false};
  std::uint32_t report_delay_ms{0};
  std::string model;
  std::optional<std::string> prompts;
  std::vector<std::array<float, 4>> boxes;
  std::optional<s3::SceneShape> box_on;
  float score_threshold{0.5F};
  float mask_threshold{0.5F};
  int max_instances{16};
  bool fp16{true};
  std::size_t workspace_mib{1024};
  std::optional<bool> verify;
  double min_iou{0.9};
  std::optional<bool> print;
  bool interactive{false};
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

s3::PixelFormat parse_encoding(std::string_view value) {
  if (value == "rgb8") return s3::PixelFormat::kRgb8;
  if (value == "bgr8") return s3::PixelFormat::kBgr8;
  if (value == "rgba8") return s3::PixelFormat::kRgba8;
  if (value == "bgra8") return s3::PixelFormat::kBgra8;
  throw std::invalid_argument("--encoding must be rgb8, bgr8, rgba8 or bgra8");
}

s3::SceneShape parse_shape(std::string_view value) {
  if (value == "circle") return s3::SceneShape::kCircle;
  if (value == "rectangle") return s3::SceneShape::kRectangle;
  if (value == "triangle") return s3::SceneShape::kTriangle;
  throw std::invalid_argument("--box-on must be circle, rectangle or triangle");
}

std::array<float, 4> parse_box(std::string_view value) {
  std::array<float, 4> box{};
  std::size_t start = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    const std::size_t comma = value.find(',', start);
    if ((i < 3) != (comma != std::string_view::npos)) {
      throw std::invalid_argument("--box requires x1,y1,x2,y2");
    }
    box[i] = parse_number<float>(value.substr(start, comma - start), "--box");
    start = comma + 1;
  }
  return box;
}

// Comma-separated phrases, trimmed; empty ones dropped.
std::vector<std::string> split_prompts(std::string_view text) {
  std::vector<std::string> result;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = std::min(text.find(',', start), text.size());
    std::string_view phrase = text.substr(start, comma - start);
    while (!phrase.empty() && phrase.front() == ' ') phrase.remove_prefix(1);
    while (!phrase.empty() && phrase.back() == ' ') phrase.remove_suffix(1);
    if (!phrase.empty()) {
      result.emplace_back(phrase);
    }
    start = comma + 1;
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
    } else if (option == "--prompts") {
      opts.prompts = std::string(value);
    } else if (option == "--box") {
      opts.boxes.push_back(parse_box(value));
    } else if (option == "--box-on") {
      opts.box_on = parse_shape(value);
    } else if (option == "--score-threshold") {
      opts.score_threshold = parse_number<float>(value, option);
    } else if (option == "--mask-threshold") {
      opts.mask_threshold = parse_number<float>(value, option);
    } else if (option == "--max-instances") {
      opts.max_instances = parse_number<int>(value, option);
    } else if (option == "--fp16") {
      opts.fp16 = parse_bool(value, option);
    } else if (option == "--workspace-mib") {
      opts.workspace_mib = parse_number<std::size_t>(value, option);
    } else if (option == "--verify") {
      opts.verify = parse_bool(value, option);
    } else if (option == "--min-iou") {
      opts.min_iou = parse_number<double>(value, option);
    } else if (option == "--print") {
      opts.print = parse_bool(value, option);
    } else if (option == "--interactive") {
      opts.interactive = parse_bool(value, option);
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
      << " --model DIR|stub [options]\n"
         "Model:\n"
         "  --model DIR|stub           A folder with tools/prepare_model.py's SAM 3 export\n"
         "                             (sam3_encoder.onnx, sam3_decoder.onnx, sam3_prompts.bin;\n"
         "                             the TensorRT engines are built once and cached beside\n"
         "                             them); 'stub' writes and uses a stand-in model with a\n"
         "                             known response, which needs no download\n"
         "  --fp16 true|false          Build the engines with FP16 allowed (default: true)\n"
         "  --workspace-mib N          TensorRT's workspace while it builds an engine, in MiB\n"
         "                             (default: 1024; SAM 3's FP32 encoder needs 4096)\n"
         "Prompts:\n"
         "  --prompts 'a,b'            Phrases from the prompt table (default: circle, blue\n"
         "                             rectangle and triangle; red and blue for the stub)\n"
         "  --box x1,y1,x2,y2          An exemplar box in fractions of the image; the boxes\n"
         "                             make one more prompt, of boxes alone\n"
         "  --box-on circle|rectangle|triangle\n"
         "                             A box around that shape of the synthetic scene's first\n"
         "                             frame (implies --still true)\n"
         "  --score-threshold F        Instances need a score above F (default: 0.5)\n"
         "  --mask-threshold F         Masks keep pixels above this probability (default: 0.5)\n"
         "  --max-instances N          Instances per image, best first (default: 16)\n"
         "  --interactive true|false   Read new prompts from the standard input while it runs,\n"
         "                             one comma-separated line at a time (default: false)\n"
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
         "  --verify true|false        Check the instances: against the host reference with\n"
         "                             --model stub, against the scene's true shapes with SAM 3\n"
         "                             (default: true for synthetic frames)\n"
         "  --min-iou F                The IoU each checked prompt's best instance must reach\n"
         "                             on the synthetic scene (default: 0.9)\n"
         "  --print true|false         Print each image's instances (default: true for files)\n"
         "  --image-dir DIR            Write each image with its instances over it as\n"
         "                             DIR/segment_NNNNNN.ppm\n"
         "  --display auto|window|offscreen|none\n"
         "                             Show each image under its instances in a HoloViz\n"
         "                             window, or render it offscreen (default: auto, a window\n"
         "                             for the camera when a display is available)\n"
         "  --render-file FILE         Offscreen: write the newest rendered frame as a PPM\n"
         "  --compile-only true|false  Build and check the graph without running it\n"
         "  --timeout-seconds N        Fail when no result arrives for N seconds plus two\n"
         "                             source periods (default: 120)\n"
         "  --startup-timeout-seconds N\n"
         "                             The same before the first result, which waits for the\n"
         "                             engine build (default: 1800)\n";
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
                                    .count = opts.count.value_or(0),
                                    .loop = opts.loop,
                                    .period = std::chrono::milliseconds(opts.period_ms),
                                    .bgr = opts.encoding == s3::PixelFormat::kBgr8};
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

// The prompts: the phrases, then the boxes as one prompt of boxes alone.
std::vector<s3::Sam3Prompt> prompts_of(const Options& opts) {
  const std::string phrases =
      opts.prompts.value_or(is_stub(opts) ? "red,blue" : "circle,blue rectangle,triangle");
  std::vector<s3::Sam3Prompt> prompts;
  for (std::string& phrase : split_prompts(phrases)) {
    prompts.push_back(s3::Sam3Prompt{.text = std::move(phrase)});
  }
  s3::Sam3Prompt boxes;
  for (const auto& box : opts.boxes) {
    boxes.boxes.push_back(s3::PromptBox{.xyxy = box});
  }
  if (opts.box_on) {
    const s3::ShapeScene scene = s3::make_shape_scene(frame_width(opts), frame_height(opts), 0);
    boxes.boxes.push_back(s3::PromptBox{.xyxy = s3::shape_box(scene, *opts.box_on)});
  }
  if (!boxes.boxes.empty()) {
    prompts.push_back(std::move(boxes));
  }
  return prompts;
}

// The shape a prompt should find in the synthetic scene, if any.
std::optional<s3::SceneShape> expected_shape(const Options& opts, const s3::Sam3Prompt& prompt) {
  if (prompt.text.empty()) {
    return opts.box_on && opts.boxes.empty() ? opts.box_on : std::nullopt;
  }
  if (prompt.text.find("circle") != std::string::npos) return s3::SceneShape::kCircle;
  if (prompt.text.find("triangle") != std::string::npos) return s3::SceneShape::kTriangle;
  if (prompt.text.find("rectangle") != std::string::npos ||
      prompt.text.find("square") != std::string::npos) {
    return s3::SceneShape::kRectangle;
  }
  return std::nullopt;
}

// The segmenter's settings; the stub's files are filled in by execute().
s3::Sam3SegmentationConfig segmenter_config(const Options& opts) {
  const std::filesystem::path dir =
      is_stub(opts) ? std::filesystem::path(".") : std::filesystem::path(opts.model);
  s3::Sam3SegmentationConfig config{.encoder = dir / "sam3_encoder.onnx",
                                    .decoder = dir / "sam3_decoder.onnx",
                                    .prompt_table = dir / "sam3_prompts.bin",
                                    .prompts = prompts_of(opts),
                                    .score_threshold = opts.score_threshold,
                                    .mask_threshold = opts.mask_threshold,
                                    .max_instances = opts.max_instances,
                                    .fp16 = opts.fp16,
                                    .build_workspace_bytes = opts.workspace_mib << 20U};
  // The output buffers fit the largest image the source gives.
  if (opts.source == Source::kFiles) {
    for (const auto& file : opts.image_files) {
      const app::ImageFileInfo info = app::read_image_info(file);
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
    throw std::invalid_argument("--model is required: a folder with the SAM 3 export, or stub");
  }
  if (!is_stub(opts)) {
    for (const char* file : {"sam3_encoder.onnx", "sam3_decoder.onnx", "sam3_prompts.bin"}) {
      std::error_code error;
      if (!std::filesystem::is_regular_file(std::filesystem::path(opts.model) / file, error)) {
        throw std::invalid_argument("--model: no " + std::string(file) + " in " + opts.model +
                                    "; see the README to prepare the model, or use --model stub");
      }
    }
  }
  if (opts.period_ms == 0 || opts.period_ms > 1000 || opts.timeout_seconds == 0 ||
      opts.startup_timeout_seconds == 0) {
    throw std::invalid_argument(
        "--period-ms must be 1..1000, and --timeout-seconds and --startup-timeout-seconds "
        "positive");
  }
  if (opts.workspace_mib < 16 || opts.workspace_mib > 65536) {
    throw std::invalid_argument("--workspace-mib must be 16..65536");
  }
  if (!(opts.min_iou >= 0.0 && opts.min_iou <= 1.0)) {
    throw std::invalid_argument("--min-iou must be in [0, 1]");
  }
  if (opts.box_on) {
    if (opts.source != Source::kSynthetic) {
      throw std::invalid_argument("--box-on applies to synthetic frames");
    }
    opts.still = true;
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
    if (opts.encoding != s3::PixelFormat::kRgb8 && opts.encoding != s3::PixelFormat::kBgr8) {
      throw std::invalid_argument("image files are decoded as rgb8 or bgr8");
    }
    if (opts.verify.value_or(false) && !is_stub(opts)) {
      throw std::invalid_argument("--verify with image files needs --model stub");
    }
    try {
      opts.image_files = app::list_image_files(opts.images);
      for (const auto& file : opts.image_files) {
        (void)app::read_image_info(file);
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
    opts.print = opts.print.value_or(true);
    app::validate_config(file_source_config(opts));
  } else if (opts.source == Source::kCamera) {
    if (!opts.images.empty() || opts.loop) {
      throw std::invalid_argument("--images and --loop need --source files");
    }
    if (opts.verify.value_or(false)) {
      throw std::invalid_argument("--verify does not apply to the camera");
    }
    if (opts.encoding != s3::PixelFormat::kRgb8) {
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
  if (opts.interactive && opts.verify.value_or(false)) {
    throw std::invalid_argument("--interactive changes the prompts: use it with --verify false");
  }
  if (!opts.image_dir.empty()) {
    std::error_code error;
    if (!std::filesystem::is_directory(opts.image_dir, error)) {
      throw std::invalid_argument("--image-dir is not a directory: " + opts.image_dir.string() +
                                  (error ? " (" + error.message() + ")" : ""));
    }
  }
  s3::validate_config(segmenter_config(opts));

  // The display: a window for the camera when a display is available, unless set.
  if (opts.display == DisplayMode::kAuto) {
    opts.display = SAM3_SEGMENTATION_ENABLE_VISUALIZATION && opts.source == Source::kCamera &&
                           display_available()
                       ? DisplayMode::kWindow
                       : DisplayMode::kNone;
  }
  if (shows(opts)) {
    if (opts.source == Source::kFiles || opts.encoding != s3::PixelFormat::kRgb8) {
      throw std::invalid_argument("--display shows rgb8 synthetic frames and the camera");
    }
    if (!SAM3_SEGMENTATION_ENABLE_VISUALIZATION) {
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

#if SAM3_SEGMENTATION_ENABLE_VISUALIZATION
// The offscreen frame expected with the stub model and a still scene: the
// image under its instances' overlay, blended by alpha. The labels' text is
// not in it; the check allows for the pixels it covers.
std::vector<std::uint8_t> expected_render(const Options& opts,
                                          const s3::Sam3SegmentationConfig& config,
                                          const s3::StubShape& shape) {
  const int width = frame_width(opts);
  const int height = frame_height(opts);
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3U);
  s3::render_shape_scene(s3::make_shape_scene(width, height, 0), s3::PixelFormat::kRgb8, rgb.data(),
                         static_cast<std::size_t>(width) * 3U);
  const std::vector<float> fpn0 = s3::reference_stub_encoder(
      s3::reference_preprocess(rgb.data(), static_cast<std::size_t>(width) * 3U,
                               s3::PixelFormat::kRgb8, width, height, shape.input_size,
                               shape.input_size, s3::Normalization{}),
      shape);
  const s3::PromptTable table = s3::read_prompt_table(config.prompt_table);
  std::vector<s3::DecoderOutput> outputs;
  for (const s3::Sam3Prompt& prompt : config.prompts) {
    outputs.push_back(s3::reference_stub_decoder(
        fpn0, s3::make_decoder_prompt(prompt, table, shape.boxes), shape));
  }
  const s3::ReferenceInstances instances =
      s3::reference_instances(outputs, config.max_instances, config.score_threshold,
                              s3::logit_of(config.mask_threshold), width, height);
  const std::vector<std::uint8_t> overlay =
      s3::reference_overlay(instances.labels, width, height, kFillAlpha);
  std::vector<std::uint8_t> frame(static_cast<std::size_t>(width) * height * 4U);
  for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
    const float alpha = static_cast<float>(overlay[4 * i + 3]) / 255.0F;
    for (std::size_t c = 0; c < 3; ++c) {
      frame[4 * i + c] = static_cast<std::uint8_t>(
          std::lround(alpha * static_cast<float>(overlay[4 * i + c]) +
                      (1.0F - alpha) * static_cast<float>(rgb[3 * i + c])));
    }
    frame[4 * i + 3] = 255;
  }
  return frame;
}
#endif

// Reads comma-separated prompts from the standard input, one line at a
// time, and hands them to the segmenter; a line it cannot use is reported
// and ignored. Ends at the end of the input.
void read_prompts(std::shared_ptr<s3::Sam3PromptControl> control) {
  std::string line;
  std::cout << kAppName << ": type prompts, comma-separated, and Enter\n" << std::flush;
  while (std::getline(std::cin, line)) {
    std::vector<s3::Sam3Prompt> prompts;
    for (std::string& phrase : split_prompts(line)) {
      prompts.push_back(s3::Sam3Prompt{.text = std::move(phrase)});
    }
    try {
      control->set(std::move(prompts));
      std::cout << kAppName << ": prompts now: " << line << '\n' << std::flush;
    } catch (const std::invalid_argument& error) {
      std::cout << kAppName << ": " << error.what() << '\n' << std::flush;
    }
  }
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  s3::Sam3SegmentationConfig segmenter = segmenter_config(opts);
  const s3::StubShape stub_shape{};
  if (is_stub(opts)) {
    // Written to the working directory, where its engines are cached beside it.
    const s3::StubModel stub = s3::write_stub_model(std::filesystem::current_path(), stub_shape);
    segmenter.encoder = stub.encoder;
    segmenter.decoder = stub.decoder;
    segmenter.prompt_table = stub.prompt_table;
  }
  if (opts.interactive) {
    segmenter.prompt_control =
        std::make_shared<s3::Sam3PromptControl>(s3::read_prompt_table(segmenter.prompt_table));
  }
  std::vector<std::string> prompt_names;
  for (const s3::Sam3Prompt& prompt : segmenter.prompts) {
    prompt_names.push_back(s3::display_name(prompt));
  }

  holoscan::Graph graph{std::string(kAppName)};
  // Frames the run expects; 0 for a camera that runs until stopped.
  const std::uint32_t frames = opts.count.value_or(0);
  auto report = std::make_shared<app::SegmentationReport>();
  // The camera runs in real time, the other sources losslessly (edges.hpp),
  // unless asked otherwise.
  const bool realtime = opts.source == Source::kCamera || opts.realtime;
  // Lossless, the synthetic and file sources send a frame only once the
  // reporter is done with the one before: one frame in flight fits the depth
  // of two that every input declares. The reporter's count of frames done is
  // the signal, read by the source: the SDK refuses a graph with a cycle.
  const std::shared_ptr<const std::atomic<std::uint32_t>> done =
      realtime ? nullptr
               : std::shared_ptr<const std::atomic<std::uint32_t>>(report, &report->frames);
  const holoscan::Output<holoscan::schema::ImageT>* image_out = nullptr;
  if (opts.source == Source::kSynthetic) {
    auto config = synthetic_source_config(opts);
    config.done = done;
    image_out = &graph.op<app::SyntheticImageSourceOp>("source", std::move(config))->image;
  } else if (opts.source == Source::kFiles) {
    auto config = file_source_config(opts);
    config.done = done;
    image_out = &graph.op<app::ImageFileSourceOp>("source", std::move(config))->image;
  } else {
    image_out = &graph.op<app::CameraSourceOp>("source", camera_source_config(opts))->image;
  }
  auto segmenter_op = graph.op<s3::Sam3SegmentationOp>("segmenter", segmenter);

  app::Verify check = app::Verify::kNone;
  if (opts.verify.value_or(false)) {
    check = is_stub(opts) ? app::Verify::kReference : app::Verify::kTruth;
  }
  std::vector<std::optional<s3::SceneShape>> expected_shapes;
  for (const s3::Sam3Prompt& prompt : segmenter.prompts) {
    expected_shapes.push_back(expected_shape(opts, prompt));
  }
  auto reporter = graph.op<app::SegmentationReportOp>(
      "report", app::SegmentationReportConfig{
                    .report = report,
                    .expected_frames = frames,
                    .realtime = realtime,
                    .delay = std::chrono::milliseconds(opts.report_delay_ms),
                    .verify = check,
                    .segmenter = segmenter,
                    .stub = stub_shape,
                    .logit_slack = opts.fp16 ? 2e-2F : 1e-3F,
                    .scene_width = frame_width(opts),
                    .scene_height = frame_height(opts),
                    .still = opts.still,
                    .expected_shapes = expected_shapes,
                    .min_iou = opts.min_iou,
                    .files = opts.source == Source::kFiles ? opts.image_files
                                                           : std::vector<std::filesystem::path>{},
                    .print = opts.print.value_or(false),
                    .image_dir = opts.image_dir});

  // Lossless, every frame is segmented and reported, in order. In real time,
  // the segmenter takes the newest frame and the reporter the newest
  // instances; the reporter pairs them with their image by capture time, so
  // its images keep a few frames.
  const holoscan::ConnectionOptions edge = realtime ? app::newest() : app::lossless();
  graph.add_flow(*image_out, segmenter_op->image, edge);
  app::connect_instances(graph, segmenter_op, reporter, edge);
  if (check == app::Verify::kReference || !opts.image_dir.empty()) {
    graph.add_flow(*image_out, reporter->image, realtime ? app::recent_frames() : app::lossless());
  }

  auto render_check = std::make_shared<app::RenderCheck>();
#if SAM3_SEGMENTATION_ENABLE_VISUALIZATION
  if (shows(opts)) {
    namespace viz = holoscan::visualization;
    const auto width = static_cast<std::uint32_t>(frame_width(opts));
    const auto height = static_cast<std::uint32_t>(frame_height(opts));
    viz::LayerStack layers;
    layers.add_image_layer("video", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                                        .max_width = width,
                                                        .max_height = height});
    // HoloViz blends RGBA image layers by their alpha, so the instances go
    // over the video.
    layers.add_image_layer(
        "instances",
        viz::ImageLayerDesc{
            .format = viz::ImageFormat::kR8G8B8A8Unorm, .max_width = width, .max_height = height});
    layers.add_text_layer(
        "labels",
        viz::TextLayerDesc{.max_glyphs = 2U * static_cast<std::uint32_t>(opts.max_instances) *
                                         s3::kOverlayLabelBytes,
                           .normalized_coordinates = true});
    const auto max_input_age =
        std::max(4 * std::chrono::milliseconds(opts.period_ms), std::chrono::milliseconds{250});
    viz::SinkDesc sink =
        viz::WindowSinkDesc{.title = "SAM 3 Segmentation", .width = width, .height = height};
    if (opts.display == DisplayMode::kOffscreen) {
      sink = viz::CudaSinkDesc{
          .width = width, .height = height, .format = viz::ImageFormat::kR8G8B8A8Unorm};
    }
    // Draw when the instances arrive, over the newest image, which is theirs.
    auto visualizer =
        graph.op<viz::HolovizOp>("holoviz", std::move(layers), std::move(sink),
                                 viz::FrameTrigger{viz::LayerTrigger{1}}, max_input_age);
    auto overlay = graph.op<s3::Sam3OverlayOp>(
        "overlay", s3::Sam3OverlayConfig{.max_width = segmenter.max_width,
                                         .max_height = segmenter.max_height,
                                         .max_instances = opts.max_instances,
                                         .prompt_names = prompt_names,
                                         .prompt_control = segmenter.prompt_control,
                                         .fill_alpha = kFillAlpha});
    // Real time in either mode: the display keeps the newest sample and never
    // holds back the segmenter. HoloViz pairs the instances with the newest
    // image no newer than them, which must still be queued, so that edge
    // keeps a few.
    graph.add_flow(*image_out, visualizer->image_layer_input(0), app::recent_frames());
    graph.add_flow(segmenter_op->labels, overlay->labels, app::newest());
    graph.add_flow(segmenter_op->scores, overlay->scores, app::newest());
    graph.add_flow(segmenter_op->boxes, overlay->boxes, app::newest());
    graph.add_flow(segmenter_op->prompt_ids, overlay->prompt_ids, app::newest());
    graph.add_flow(segmenter_op->count, overlay->count, app::newest());
    graph.add_flow(overlay->overlay, visualizer->image_layer_input(1), app::newest());
    graph.add_flow(overlay->text, visualizer->tensor_layer_input(2), app::newest());
    if (opts.display == DisplayMode::kOffscreen) {
      // Only a still synthetic scene through the stub has a known rendering.
      std::vector<std::uint8_t> expected;
      if (is_stub(opts) && opts.still && opts.source == Source::kSynthetic && !opts.interactive) {
        expected = expected_render(opts, segmenter, stub_shape);
      }
      auto checker = graph.op<app::RenderCheckOp>(
          "render_check", app::RenderCheckConfig{.check = render_check,
                                                 .width = frame_width(opts),
                                                 .height = frame_height(opts),
                                                 .expected = std::move(expected),
                                                 .max_mismatched_fraction = 0.03,
                                                 .write = opts.render_file});
      graph.add_flow(visualizer->output(), checker->frame, app::newest());
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
  for (const char* port : {"masks", "labels", "scores", "boxes", "prompt_ids", "count"}) {
    on_gpu("segmenter", port);
  }
  if (opts.source == Source::kFiles || (opts.source == Source::kSynthetic && !opts.host_input)) {
    on_gpu("source", "image");
  }
  if (shows(opts)) {
    on_gpu("overlay", "overlay");
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
    std::cout << kAppName << ": plan=OK\n";
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
  // Build the TensorRT engines now rather than in start(): see prepare_engines().
  s3::prepare_engines(segmenter);
  if (opts.interactive) {
    // Detached: it may still wait for a line when the run ends.
    std::thread(read_prompts, segmenter.prompt_control).detach();
  }
  // The GPU memory the run takes: free memory before it, less the least free
  // during it. Other processes on the GPU change it too.
  const std::size_t free_before = gpu_free_bytes();
  std::size_t least_free = free_before;
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  // The run times out when no result arrives for this long. Before the first
  // one, the segmenter may still be loading its engines. The sums cannot
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
    } else if (!opts.interactive &&
               now - last_progress >= (seen == 0 ? startup_patience : patience)) {
      timed_out_after = seen == 0 ? opts.startup_timeout_seconds : opts.timeout_seconds;
      break;
    }
  }
  if (timed_out_after) {
    std::cerr << kAppName << ": timed out: no result for " << *timed_out_after << " s\n";
    session.request_stop();
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
  // In real time, the frames that came while the segmenter or the reporter
  // was busy are skipped; the last one must arrive.
  const bool all_frames = realtime ? report->last_frame_id == frames : report->frames == frames;
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
  std::cout << " prompts=" << segmenter.prompts.size() << " instances=" << report->instances;
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
  if (!opts.image_dir.empty()) {
    std::cout << " files_written=" << report->files_written;
  }
  if (report->frame_gaps != 0) {
    std::cout << " frame_gaps=" << report->frame_gaps;
  }
  if (report->skipped != 0) {
    std::cout << " skipped=" << report->skipped;
  }
  if (report->reference_checked != 0) {
    std::cout << " reference: checked=" << report->reference_checked
              << " failed=" << report->reference_failed
              << " worst_pixels=" << report->worst_reference_pixels;
  }
  if (report->truth_checked != 0) {
    std::cout << " truth: checked=" << report->truth_checked << " failed=" << report->truth_failed
              << std::setprecision(4) << " iou_min=" << report->min_iou
              << " iou_mean=" << (report->ious != 0 ? report->sum_iou / report->ious : 0.0)
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
            << '\n'
            << std::flush;
  const int status = passed || (stopped && frames == 0) ? 0 : 1;
  // The prompt reader may still wait for a line.
  if (opts.interactive) {
    std::quick_exit(status);
  }
  return status;
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
