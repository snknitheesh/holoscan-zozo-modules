// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/color_convert/resize_op.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/color_convert/camera_info.hpp>
#include <holoscan/color_convert/converter.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/messages/conversion.hpp>

#include "op_support.hpp"

namespace holoscan::color_convert {
namespace {

// The output's capacity: its size in its format; when it keeps each input's,
// in the input's format if known, else in the widest format.
std::uint64_t max_bytes(const ResizeConfig& config) {
  if (const auto format = config.output_format ? config.output_format : config.input_format) {
    return packed_size(*format, config.width, config.height);
  }
  std::uint64_t most = 0;
  for (const PixelFormat format : kAllFormats) {
    most = std::max<std::uint64_t>(most, packed_size(format, config.width, config.height));
  }
  return most;
}

ConversionParams params_of(const ResizeConfig& config) {
  return ConversionParams{.matrix = config.matrix,
                          .range = config.range,
                          .mode = config.mode,
                          .crop = config.crop,
                          .letterbox = config.letterbox,
                          .fill_rgb = config.fill_rgb};
}

holoscan::Error invalid(const std::string& what) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "ResizeOp: " + what};
}

}  // namespace

namespace {

// What a resized camera is made for, besides the calibration: the input's
// size, the output's format and the camera's frame id (a calibration that is
// the same may name another frame).
struct MadeFor {
  int width{};
  int height{};
  PixelFormat format{};
  std::string frame_id;

  friend bool operator==(const MadeFor&, const MadeFor&) = default;
};

}  // namespace

// The newest camera info, and the resized camera made from it.
struct ResizeOp::CameraState {
  messages::LatestCameraInfo<messages::schema::CameraInfoT> latest;
  messages::schema::CameraInfoT resized;
  std::optional<MadeFor> made_for;
};

void validate_config(const ResizeConfig& config) {
  check_available(config.backend);
  if (config.width < 1 || config.height < 1 || config.width > kMaxImageSide ||
      config.height > kMaxImageSide) {
    throw std::invalid_argument("width and height must be 1.." + std::to_string(kMaxImageSide));
  }
  if (config.crop && (config.crop->width < 1 || config.crop->height < 1 || config.crop->x < 0 ||
                      config.crop->y < 0)) {
    throw std::invalid_argument("crop needs a nonnegative offset and a positive size");
  }
  if (config.output_format && needs_even_width(*config.output_format) && config.width % 2 != 0) {
    throw std::invalid_argument(std::string(format_name(*config.output_format)) +
                                " output needs an even width");
  }
  if (!config.output_format && config.input_format && needs_even_width(*config.input_format) &&
      config.width % 2 != 0) {
    throw std::invalid_argument(std::string(format_name(*config.input_format)) +
                                " output (input_format, kept) needs an even width");
  }
  if (config.custom_format && *config.custom_format != PixelFormat::kNv16) {
    throw std::invalid_argument(
        "custom_format names what CUSTOM images hold, which only nv16 needs: every other "
        "format has its own encoding");
  }
  if (config.with_camera_info &&
      config.camera_info_input.max_age <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("camera_info_input.max_age must be > 0");
  }
}

ResizeOp::ResizeOp(ResizeConfig config)
    : config_(std::move(config)), camera_(std::make_unique<CameraState>()) {
  validate_config(config_);
  runner_ = std::make_unique<detail::ConversionRunner>("ResizeOp", config_.backend);
}

ResizeOp::~ResizeOp() = default;

void ResizeOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots: one image in compute() while the next one arrives, as for
  // ColorConvertOp. An edge that sets no depth of its own keeps these two, a
  // drop-oldest one too (the newest two); give a live edge queue_depth 1 to
  // keep only the newest image.
  spec.input(image, "image").queue_depth(2U);
  spec.output(resized, "resized")
      .produces_tensor(detail::output_spec(config_.output_format, max_bytes(config_)))
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
  // The camera info ports exist only when configured: an unwired input fails
  // compile(), and an emit into an unconnected output fails.
  if (config_.with_camera_info) {
    spec.input(camera_info, "camera_info").queue_depth(2U);
    spec.output(resized_camera_info, "resized_camera_info");
  }
}

holoscan::Contract ResizeOp::contract() const {
  if (config_.with_camera_info) {
    return messages::camera_info_contract(config_.camera_info_input, camera_info, image);
  }
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void ResizeOp::start() {
  camera_->latest.reset();
  camera_->made_for.reset();
}

void ResizeOp::stop() { runner_->reset(); }

holoscan::expected<void, holoscan::Error> ResizeOp::compute(holoscan::ExecutionContext& context) {
  detail::Target target{
      .format = config_.output_format,
      .width = config_.width,
      .height = config_.height,
      .params = params_of(config_),
      .use_input_color_space = config_.use_input_color_space,
      .custom_format = config_.custom_format,
      .input_format = config_.input_format,
      .max_bytes = max_bytes(config_),
      .capacity_hint = "set output_format, or an input_format of more bytes per pixel",
      .keep_roi = false,
      .on_backend = config_.on_backend,
  };
  if (!config_.with_camera_info) {
    return runner_->run(context, image, resized, target);
  }
  CameraState& camera = *camera_;
  auto info = camera.latest.update(camera_info);
  if (!info) {
    return holoscan::make_unexpected(std::move(info).error());
  }
  if (*info == nullptr) {
    return {};  // No camera info yet: the image is skipped.
  }
  const messages::CameraInfo& input_camera = **info;
  target.before_emit =
      [&](int width, int height, PixelFormat format,
          const holoscan::EmitOptions& emit) -> holoscan::expected<void, holoscan::Error> {
    // The output's format matters to the letterbox's rounding (even offsets
    // and sizes where its chroma is subsampled).
    const MadeFor made_for{width, height, format, input_camera.header.frame_id};
    if (camera.latest.changed() || camera.made_for != made_for) {
      try {
        camera.resized = messages::to_message(color_convert::resized_camera_info(
            input_camera, width, height, config_.width, config_.height, format, target.params));
      } catch (const std::invalid_argument& error) {
        return holoscan::make_unexpected(invalid(error.what()));
      }
      camera.made_for = made_for;
    }
    camera.resized.header->capture_timestamp_ns =
        emit.capture_time ? emit.capture_time->timestamp_ns : 0;
    return resized_camera_info.emit_ref(camera.resized, emit);
  };
  return runner_->run(context, image, resized, target);
}

}  // namespace holoscan::color_convert
