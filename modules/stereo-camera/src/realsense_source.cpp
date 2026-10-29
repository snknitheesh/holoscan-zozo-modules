// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The realsense backend: an Intel RealSense camera through librealsense. Its two
// infrared cameras, Y8, are the eyes, rectified by the camera; optionally its
// color camera (RGB8) and its depth (Z16, in the left infrared camera's frame).
// librealsense's pipeline matches the streams of one frame by their hardware
// timestamps, so one frame set holds images taken together. The calibration is
// the camera's factory calibration, read from it at open().

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/stereo_camera/source.hpp>
#include <librealsense2/rs.hpp>

#include "sources.hpp"

namespace holoscan::stereo_camera::detail {
namespace {

CameraModel camera_model(const rs2_intrinsics& k) {
  CameraModel model{
      .width = k.width, .height = k.height, .fx = k.fx, .fy = k.fy, .cx = k.ppx, .cy = k.ppy};
  std::size_t count = 5;
  switch (k.model) {
    case RS2_DISTORTION_NONE:
      model.distortion_model = DistortionModel::kNone;
      count = 0;
      break;
    case RS2_DISTORTION_MODIFIED_BROWN_CONRADY:
      model.distortion_model = DistortionModel::kModifiedBrownConrady;
      break;
    case RS2_DISTORTION_INVERSE_BROWN_CONRADY:
      model.distortion_model = DistortionModel::kInverseBrownConrady;
      break;
    case RS2_DISTORTION_BROWN_CONRADY:
      model.distortion_model = DistortionModel::kBrownConrady;
      break;
    case RS2_DISTORTION_FTHETA:
      model.distortion_model = DistortionModel::kFTheta;
      count = 1;
      break;
    case RS2_DISTORTION_KANNALA_BRANDT4:
      model.distortion_model = DistortionModel::kKannalaBrandt4;
      count = 4;
      break;
    default:
      throw std::runtime_error(std::string("realsense: unsupported distortion model ") +
                               rs2_distortion_to_string(k.model));
  }
  model.distortion.assign(k.coeffs, k.coeffs + count);
  return model;
}

RigidTransform transform(const rs2_extrinsics& e) {
  RigidTransform result;
  // librealsense stores the rotation column-major.
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      result.rotation[row * 3 + column] = e.rotation[column * 3 + row];
    }
  }
  std::copy(e.translation, e.translation + 3, result.translation.begin());
  return result;
}

std::string info(const rs2::device& device, rs2_camera_info field) {
  return device.supports(field) ? device.get_info(field) : "";
}

struct FrameTime {
  std::int64_t timestamp_ns{};
  // Nanoseconds of the host's system clock, so that the frame's age is known.
  bool system_clock{};
};

// When the frame was taken. With the camera's hardware timestamps (UVC metadata,
// which needs librealsense's kernel support), librealsense's own timestamp: in
// global time, the hardware clock mapped to the host's. Without them, global time
// does not advance from frame to frame, so the time the host received the frame
// is the best there is.
FrameTime frame_time(const rs2::frame& frame) {
  const bool hardware = frame.supports_frame_metadata(RS2_FRAME_METADATA_SENSOR_TIMESTAMP) ||
                        frame.supports_frame_metadata(RS2_FRAME_METADATA_FRAME_TIMESTAMP);
  const auto milliseconds = [](double ms) {
    return static_cast<std::int64_t>(std::llround(ms * 1e6));
  };
  if (!hardware) {
    for (const rs2_frame_metadata_value field :
         {RS2_FRAME_METADATA_BACKEND_TIMESTAMP, RS2_FRAME_METADATA_TIME_OF_ARRIVAL}) {
      if (frame.supports_frame_metadata(field)) {
        // Whole milliseconds; in integers, as a double cannot hold nanoseconds since 1970.
        return FrameTime{.timestamp_ns = frame.get_frame_metadata(field) * 1'000'000LL,
                         .system_clock = true};
      }
    }
  }
  const rs2_timestamp_domain domain = frame.get_frame_timestamp_domain();
  return FrameTime{.timestamp_ns = milliseconds(frame.get_timestamp()),
                   .system_clock = domain == RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME ||
                                   domain == RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME};
}

class RealSenseSource final : public StereoSource {
 public:
  explicit RealSenseSource(const SourceConfig& config)
      : config_(config), layout_(realsense_layout(config)) {}
  ~RealSenseSource() override { close(); }

  [[nodiscard]] Backend backend() const noexcept override { return Backend::kRealSense; }
  [[nodiscard]] const SourceLayout& layout() const noexcept override { return layout_; }

  [[nodiscard]] const StereoCalibration& calibration() const override {
    if (!calibration_) {
      throw std::logic_error("realsense: calibration() before open()");
    }
    return *calibration_;
  }

  void open() override {
    close();
    const RealSenseOptions& options = config_.realsense;
    rs2::context context;
    rs2::device_list devices = context.query_devices();
    if (devices.size() == 0) {
      throw std::invalid_argument("realsense: no RealSense camera is connected");
    }
    rs2::config streams;
    if (!options.serial.empty()) {
      streams.enable_device(options.serial);
    }
    streams.enable_stream(RS2_STREAM_INFRARED, 1, options.width, options.height, RS2_FORMAT_Y8,
                          options.fps);
    streams.enable_stream(RS2_STREAM_INFRARED, 2, options.width, options.height, RS2_FORMAT_Y8,
                          options.fps);
    if (config_.rgb) {
      streams.enable_stream(RS2_STREAM_COLOR, options.rgb_width, options.rgb_height,
                            RS2_FORMAT_RGB8, options.fps);
    }
    if (config_.depth) {
      streams.enable_stream(RS2_STREAM_DEPTH, options.width, options.height, RS2_FORMAT_Z16,
                            options.fps);
    }
    pipeline_.emplace(context);
    if (!streams.can_resolve(*pipeline_)) {
      pipeline_.reset();
      throw std::invalid_argument(
          "realsense: the camera" + (options.serial.empty() ? "" : " " + options.serial) +
          " does not offer infrared Y8 " + std::to_string(options.width) + "x" +
          std::to_string(options.height) + " at " + std::to_string(options.fps) + " fps" +
          (config_.rgb ? ", color RGB8 " + std::to_string(options.rgb_width) + "x" +
                             std::to_string(options.rgb_height)
                       : "") +
          (config_.depth ? " and depth Z16" : "") + " together; see rs-enumerate-devices");
    }
    try {
      // The projector is set before streaming starts, so that the first frames have it.
      const rs2::device device = streams.resolve(*pipeline_).get_device();
      rs2::depth_sensor sensor = device.first<rs2::depth_sensor>();
      if (sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
        sensor.set_option(RS2_OPTION_EMITTER_ENABLED, options.emitter_enabled ? 1.0F : 0.0F);
      } else if (options.emitter_enabled) {
        throw std::invalid_argument("realsense: this camera has no projector to turn on");
      }
      const rs2::pipeline_profile profile = pipeline_->start(streams);
      streaming_ = true;
      index_ = 0;
      last_pair_.reset();
      calibration_ = read_calibration(profile, sensor);
    } catch (...) {
      close();
      throw;
    }
  }

  [[nodiscard]] std::optional<StereoFrameView> grab() override {
    if (!streaming_) {
      throw std::logic_error("realsense: grab() before open()");
    }
    // The newest complete frame set; older ones are stale.
    bool found = false;
    for (rs2::frameset frames; pipeline_->poll_for_frames(&frames);) {
      rs2::video_frame left = frames.get_infrared_frame(1);
      rs2::video_frame right = frames.get_infrared_frame(2);
      if (!left || !right || (config_.rgb && !frames.get_color_frame()) ||
          (config_.depth && !frames.get_depth_frame())) {
        continue;  // Incomplete: a stream missed this frame.
      }
      // The pipeline also sends a frame set when only another stream has a new
      // frame; the eyes' frame number tells a new pair from one already given.
      if (last_pair_ && left.get_frame_number() <= *last_pair_) {
        continue;
      }
      last_pair_ = left.get_frame_number();
      frames_ = std::move(frames);
      found = true;
    }
    if (!found) {
      return std::nullopt;
    }
    const rs2::video_frame left = frames_.get_infrared_frame(1);
    const rs2::video_frame right = frames_.get_infrared_frame(2);
    const auto view = [](const rs2::video_frame& frame, PixelFormat format) {
      return ImageView{.data = static_cast<const std::uint8_t*>(frame.get_data()),
                       .width = frame.get_width(),
                       .height = frame.get_height(),
                       .pitch = static_cast<std::size_t>(frame.get_stride_in_bytes()),
                       .format = format};
    };
    const FrameTime time = frame_time(left);
    StereoFrameView result{.left = view(left, PixelFormat::kMono8),
                           .right = view(right, PixelFormat::kMono8),
                           .index = index_++,
                           .timestamp_ns = time.timestamp_ns,
                           .device_sequence = left.get_frame_number()};
    if (config_.rgb) result.rgb = view(frames_.get_color_frame(), PixelFormat::kRgb8);
    if (config_.depth) result.depth = view(frames_.get_depth_frame(), PixelFormat::kMono16);
    if (time.system_clock) {
      const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
      if (const std::int64_t age = now - time.timestamp_ns; age >= 0) {
        result.age_ns = age;
      }
    }
    return result;
  }

  void close() noexcept override {
    frames_ = rs2::frameset();
    if (pipeline_ && streaming_) {
      try {
        pipeline_->stop();
      } catch (...) {  // NOLINT(bugprone-empty-catch): closing must not throw.
      }
    }
    streaming_ = false;
    pipeline_.reset();
  }

 private:
  StereoCalibration read_calibration(const rs2::pipeline_profile& profile,
                                     rs2::depth_sensor& sensor) const {
    const auto ir1 = profile.get_stream(RS2_STREAM_INFRARED, 1).as<rs2::video_stream_profile>();
    const auto ir2 = profile.get_stream(RS2_STREAM_INFRARED, 2).as<rs2::video_stream_profile>();
    StereoCalibration calibration;
    calibration.left = camera_model(ir1.get_intrinsics());
    calibration.right = camera_model(ir2.get_intrinsics());
    calibration.right_from_left = transform(ir1.get_extrinsics_to(ir2));
    // The infrared Y8 streams are rectified on the camera.
    calibration.rectified = calibration.left.undistorted() && calibration.right.undistorted();
    if (config_.rgb) {
      const auto color = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
      calibration.rgb = AuxiliaryCamera{.model = camera_model(color.get_intrinsics()),
                                        .from_left = transform(ir1.get_extrinsics_to(color))};
    }
    if (config_.depth) {
      const auto depth = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
      calibration.depth = DepthCamera{.model = camera_model(depth.get_intrinsics()),
                                      .from_left = transform(ir1.get_extrinsics_to(depth)),
                                      .scale = sensor.get_depth_scale()};
    }
    const rs2::device device = profile.get_device();
    calibration.camera.name = info(device, RS2_CAMERA_INFO_NAME);
    calibration.camera.serial = info(device, RS2_CAMERA_INFO_SERIAL_NUMBER);
    calibration.camera.firmware = info(device, RS2_CAMERA_INFO_FIRMWARE_VERSION);
    const RealSenseOptions& options = config_.realsense;
    calibration.camera.settings = {
        {"backend", "realsense"},
        {"librealsense", RS2_API_VERSION_STR},
        {"fps", std::to_string(options.fps)},
        {"emitter_enabled", options.emitter_enabled ? "true" : "false"},
        {"infrared", "Y8 " + std::to_string(options.width) + "x" + std::to_string(options.height)},
    };
    if (config_.rgb) {
      calibration.camera.settings["color"] =
          "RGB8 " + std::to_string(options.rgb_width) + "x" + std::to_string(options.rgb_height);
    }
    if (config_.depth) {
      calibration.camera.settings["depth"] =
          "Z16 " + std::to_string(options.width) + "x" + std::to_string(options.height);
    }
    validate(calibration);
    return calibration;
  }

  SourceConfig config_;
  SourceLayout layout_;
  std::optional<rs2::pipeline> pipeline_;
  bool streaming_{};
  rs2::frameset frames_;
  std::optional<StereoCalibration> calibration_;
  std::uint64_t index_{};
  // The frame number of the last pair given.
  std::optional<unsigned long long> last_pair_;  // NOLINT(runtime/int): librealsense's type.
};

}  // namespace

std::unique_ptr<StereoSource> create_realsense_source(const SourceConfig& config) {
  return std::make_unique<RealSenseSource>(config);
}

}  // namespace holoscan::stereo_camera::detail
