// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The camera backends behind StereoCameraOp, and the interface they share. Each
// gives frames of left and right images, optionally an RGB image and a depth
// image, with the camera's calibration. No SDK: a backend can be used, and
// tested, on its own.

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/stereo_camera/calibration.hpp>
#include <holoscan/stereo_camera/image.hpp>
#include <holoscan/stereo_camera/recording.hpp>
#include <holoscan/stereo_camera/side_by_side.hpp>

namespace holoscan::stereo_camera {

enum class Backend : std::uint8_t {
  kRealSense,       // An Intel RealSense camera through librealsense; optional at build time.
  kV4l2SideBySide,  // A V4L2 camera that sends both eyes side by side, with a calibration file.
  kPlayback,        // A recording (recording.hpp); always built.
};

// "realsense", "v4l2-sbs" or "playback".
[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;
[[nodiscard]] std::optional<Backend> parse_backend(std::string_view name) noexcept;
// The backends compiled into this build, in a stable order.
[[nodiscard]] std::vector<Backend> available_backends();
// Whether this build has librealsense, and so the realsense backend.
[[nodiscard]] bool realsense_available() noexcept;

struct RealSenseOptions {
  // The camera's serial number; empty takes the first RealSense found.
  std::string serial;
  // Both infrared eyes, Y8 (rectified by the camera).
  int width{1280};
  int height{720};
  int fps{30};
  // The color camera, RGB8.
  int rgb_width{1280};
  int rgb_height{720};
  // The infrared projector. On, the eyes see its dot pattern, which helps the
  // camera's depth (and any stereo matcher) on surfaces without texture.
  bool emitter_enabled{true};
};

struct V4l2SideBySideOptions {
  std::filesystem::path device{"/dev/video0"};
  // The whole side-by-side frame: twice an eye's width.
  int width{2560};
  int height{720};
  int fps{30};
  SideBySideFormat format{SideBySideFormat::kYuyv};
  // The eyes as kMono8 (the luma) or kRgb8.
  PixelFormat output{PixelFormat::kMono8};
  // The calibration file, as a recording's calibration.json; its eyes must be
  // width / 2 x height.
  std::filesystem::path calibration_file;
};

struct PlaybackOptions {
  std::filesystem::path directory;
  // Start over after the last frame, rather than end.
  bool loop{false};
};

struct SourceConfig {
  Backend backend{Backend::kPlayback};
  // Also give the RGB image and the depth image. The realsense backend turns
  // those streams on; a recording must have them; v4l2-sbs has neither.
  bool rgb{false};
  bool depth{false};
  RealSenseOptions realsense;
  V4l2SideBySideOptions v4l2;
  PlaybackOptions playback;
};

// The streams a source gives, known before it opens.
struct SourceLayout {
  StreamLayout left;
  StreamLayout right;
  std::optional<StreamLayout> rgb;
  std::optional<StreamLayout> depth;
  // A live camera, whose frames come at its own pace and may be dropped; or a
  // recording, played losslessly.
  bool live{};
};

class StereoSource {
 public:
  virtual ~StereoSource() = default;
  StereoSource(const StereoSource&) = delete;
  StereoSource& operator=(const StereoSource&) = delete;

  [[nodiscard]] virtual Backend backend() const noexcept = 0;
  [[nodiscard]] virtual const SourceLayout& layout() const noexcept = 0;
  // Opens the camera or the recording and starts its streams. A device is opened
  // only here, never on construction: one camera serves one owner at a time.
  // @throws std::invalid_argument or std::runtime_error naming what failed.
  virtual void open() = 0;
  // The calibration; for a camera, as read from it by open().
  // @throws std::logic_error before open().
  [[nodiscard]] virtual const StereoCalibration& calibration() const = 0;
  // The newest frame since the last call, or nullopt when there is none yet (a
  // live camera) or none left (the end of a recording that does not loop). Never
  // blocks for long. The frame's views stay valid until the next grab() or close().
  // @throws std::runtime_error for a failed read; the source stays usable.
  [[nodiscard]] virtual std::optional<StereoFrameView> grab() = 0;
  // Whether the source has no more frames: the end of a recording that does not loop.
  [[nodiscard]] virtual bool finished() const noexcept { return false; }
  // Stops the streams and releases the device. Safe to call more than once.
  virtual void close() noexcept = 0;

 protected:
  StereoSource() = default;
};

// Checks `config` without touching a device: the backend is compiled in, the
// sizes and rates are usable, rgb and depth are possible for the backend; for
// playback, the recording loads and has what is asked for; for v4l2-sbs, the
// calibration file loads and matches the size.
// @throws std::invalid_argument naming the first problem.
void validate(const SourceConfig& config);

// Makes the source `config` describes, validated, but does not open it.
// @throws std::invalid_argument as validate() does.
[[nodiscard]] std::unique_ptr<StereoSource> create_source(const SourceConfig& config);

}  // namespace holoscan::stereo_camera
