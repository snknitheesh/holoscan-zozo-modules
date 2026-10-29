// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Records a stereo camera to a recording directory (recording.hpp), through the
// module's own backends: what it writes is what StereoCameraOp would emit, and
// the playback backend plays it back without the camera or its driver. Needs no
// Holoscan SDK.
//
//   stereo_camera_record --output DIR [--source realsense|v4l2-sbs] [--frames 60] ...

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include <holoscan/stereo_camera/recording.hpp>
#include <holoscan/stereo_camera/source.hpp>

namespace {

namespace sc = holoscan::stereo_camera;

struct Options {
  sc::SourceConfig source;
  std::filesystem::path output;
  std::uint32_t frames{60};
  std::uint32_t warmup{30};
  std::uint32_t timeout_seconds{30};
  bool verbose{false};
  bool help{false};
};

void usage(const char* program) {
  std::cout << "Usage: " << program << " --output DIR [options]\n"
            << "Records a stereo camera to DIR: calibration.json, index.csv and Netpbm images.\n\n"
            << "  --source realsense|v4l2-sbs   Camera backend (default realsense)\n"
            << "  --frames N                    Frames to record (default 60)\n"
            << "  --warmup N                    Frames to skip first, for exposure (default 30)\n"
            << "  --rgb true|false              Record the RGB image (realsense; default true)\n"
            << "  --depth true|false            Record the depth image (realsense; default true)\n"
            << "  --timeout-seconds N           Give up after N s without a frame (default 30)\n"
            << "  --verbose true|false          Print each frame's number and timestamp\n"
            << "realsense:\n"
            << "  --serial S                    Camera serial number (default: the first one)\n"
            << "  --width W --height H --fps F  Infrared size and rate (default 1280 720 30)\n"
            << "  --rgb-width W --rgb-height H  Color size (default 1280 720)\n"
            << "  --emitter true|false          Infrared projector (default true)\n"
            << "v4l2-sbs:\n"
            << "  --device PATH                 V4L2 device (default /dev/video0)\n"
            << "  --width W --height H --fps F  Whole side-by-side frame (default 2560 720 30)\n"
            << "  --format yuyv|gray            The camera's pixel format (default yuyv)\n"
            << "  --color true|false            Record the eyes in color (default false: gray)\n"
            << "  --calibration FILE            Calibration file (required)\n";
}

bool parse_bool(std::string_view value, std::string_view option) {
  if (value == "true") return true;
  if (value == "false") return false;
  throw std::invalid_argument(std::string(option) + " takes true or false");
}

template <typename T>
T parse_number(std::string_view value, std::string_view option) {
  T result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (value.empty() || error != std::errc() || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(option) + " takes a whole number");
  }
  return result;
}

Options parse_args(int argc, char** argv) {
  Options opts;
  opts.source.backend = sc::Backend::kRealSense;
  opts.source.rgb = true;
  opts.source.depth = true;
  std::optional<int> width;
  std::optional<int> height;
  std::optional<int> fps;
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
      if (!backend || *backend == sc::Backend::kPlayback) {
        throw std::invalid_argument("--source takes realsense or v4l2-sbs");
      }
      opts.source.backend = *backend;
    } else if (option == "--output") {
      opts.output = value;
    } else if (option == "--frames") {
      opts.frames = parse_number<std::uint32_t>(value, option);
    } else if (option == "--warmup") {
      opts.warmup = parse_number<std::uint32_t>(value, option);
    } else if (option == "--timeout-seconds") {
      opts.timeout_seconds = parse_number<std::uint32_t>(value, option);
    } else if (option == "--verbose") {
      opts.verbose = parse_bool(value, option);
    } else if (option == "--rgb") {
      opts.source.rgb = parse_bool(value, option);
    } else if (option == "--depth") {
      opts.source.depth = parse_bool(value, option);
    } else if (option == "--serial") {
      opts.source.realsense.serial = value;
    } else if (option == "--width") {
      width = parse_number<int>(value, option);
    } else if (option == "--height") {
      height = parse_number<int>(value, option);
    } else if (option == "--fps") {
      fps = parse_number<int>(value, option);
    } else if (option == "--rgb-width") {
      opts.source.realsense.rgb_width = parse_number<int>(value, option);
    } else if (option == "--rgb-height") {
      opts.source.realsense.rgb_height = parse_number<int>(value, option);
    } else if (option == "--emitter") {
      opts.source.realsense.emitter_enabled = parse_bool(value, option);
    } else if (option == "--device") {
      opts.source.v4l2.device = value;
    } else if (option == "--format") {
      const auto format = sc::parse_side_by_side_format(value);
      if (!format) throw std::invalid_argument("--format takes yuyv or gray");
      opts.source.v4l2.format = *format;
    } else if (option == "--color") {
      opts.source.v4l2.output =
          parse_bool(value, option) ? sc::PixelFormat::kRgb8 : sc::PixelFormat::kMono8;
    } else if (option == "--calibration") {
      opts.source.v4l2.calibration_file = value;
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (opts.source.backend == sc::Backend::kRealSense) {
    opts.source.realsense.width = width.value_or(opts.source.realsense.width);
    opts.source.realsense.height = height.value_or(opts.source.realsense.height);
    opts.source.realsense.fps = fps.value_or(opts.source.realsense.fps);
  } else {
    // A side-by-side camera has neither; only say so when asked for explicitly.
    opts.source.rgb = false;
    opts.source.depth = false;
    opts.source.v4l2.width = width.value_or(opts.source.v4l2.width);
    opts.source.v4l2.height = height.value_or(opts.source.v4l2.height);
    opts.source.v4l2.fps = fps.value_or(opts.source.v4l2.fps);
  }
  if (opts.output.empty()) {
    throw std::invalid_argument("--output is required");
  }
  if (opts.frames == 0) {
    throw std::invalid_argument("--frames must be positive");
  }
  return opts;
}

int record(const Options& opts) {
  auto source = sc::create_source(opts.source);
  source->open();
  const sc::StereoCalibration& calibration = source->calibration();
  sc::RecordingWriter writer(opts.output, calibration, opts.source.rgb, opts.source.depth);
  std::cout << "Recording " << opts.frames << " frames of " << calibration.camera.name << " "
            << calibration.camera.serial << " (firmware " << calibration.camera.firmware << ") to "
            << opts.output.string() << ", after " << opts.warmup << " warm-up frames\n";
  std::uint32_t seen = 0;
  std::uint64_t gaps = 0;
  std::optional<std::uint64_t> last_sequence;
  auto last_frame = std::chrono::steady_clock::now();
  const auto patience = std::chrono::seconds(opts.timeout_seconds);
  while (writer.frames() < opts.frames) {
    std::optional<sc::StereoFrameView> frame = source->grab();
    const auto now = std::chrono::steady_clock::now();
    if (!frame) {
      if (now - last_frame > patience) {
        throw std::runtime_error("no frame for " + std::to_string(opts.timeout_seconds) + " s");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    last_frame = now;
    if (opts.verbose) {
      std::cout << (seen < opts.warmup ? "warm-up " : "frame ") << frame->index
                << " device_sequence=" << frame->device_sequence
                << " timestamp_ns=" << frame->timestamp_ns << '\n';
    }
    if (seen++ < opts.warmup) {
      continue;
    }
    if (last_sequence && frame->device_sequence != 0 &&
        frame->device_sequence != *last_sequence + 1U) {
      ++gaps;
    }
    last_sequence = frame->device_sequence;
    writer.write(*frame);
  }
  writer.close();
  source->close();
  std::cout << "Recorded " << writer.frames() << " frames; " << gaps
            << " gap(s) in the camera's frame numbers\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  const Options opts = parse_args(argc, argv);
  if (opts.help) {
    usage(argv[0]);
    return 0;
  }
  return record(opts);
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
