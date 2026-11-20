// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/fiducials/fiducial_overlay_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/fiducials/fiducial_messages.hpp>
#include <holoscan/fiducials/pose.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "image_output.hpp"
#include "overlay_drawing.hpp"

namespace holoscan::fiducials {
namespace {

namespace viz = holoscan::visualization;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr std::uint32_t kLabelRowBytes = viz::kTextRunHeaderBytes + kOverlayLabelBytes;
constexpr std::uint32_t kLabelColor = 0xFF00FFFFU;  // Opaque yellow, 0xAABBGGRR.
constexpr float kLabelSizePx = 22.0F;

}  // namespace

FiducialOverlayOp::FiducialOverlayOp(FiducialOverlayConfig config) : config_(std::move(config)) {
  if (config_.width <= 0 || config_.height <= 0) {
    throw std::invalid_argument("FiducialOverlayOp: width and height must be positive");
  }
  if (config_.draw_axes) {
    if (!(std::isfinite(config_.marker_length) && config_.marker_length > 0)) {
      throw std::invalid_argument("FiducialOverlayOp: marker_length must be positive to draw axes");
    }
    if (config_.camera_info_input.max_age <= std::chrono::nanoseconds::zero()) {
      throw std::invalid_argument("FiducialOverlayOp: camera_info_input.max_age must be > 0");
    }
  }
}

void FiducialOverlayOp::setup(holoscan::OperatorSpec& spec) {
  // Two detections queued: the one being drawn and the next; the same for camera info.
  spec.input(detections, "detections").queue_depth(2U);
  if (config_.draw_axes) spec.input(camera_info, "camera_info").queue_depth(2U);
  image_output(spec, overlay, "overlay", holoscan::MemoryKind::kHost, config_.width, config_.height,
               4);
  spec.output(labels, "labels")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kHost,
                  .dtype = kUInt8,
                  .rank = 2U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(std::uint64_t{kOverlayLabelRows} * kLabelRowBytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract FiducialOverlayOp::contract() const {
  if (config_.draw_axes) {
    return messages::camera_info_contract(config_.camera_info_input, camera_info, detections);
  }
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{detections});
  return result;
}

void FiducialOverlayOp::start() {
  camera_.reset();
  unstamped_logged_ = false;
}

holoscan::expected<void, holoscan::Error> FiducialOverlayOp::compute(
    holoscan::ExecutionContext& context) {
  const messages::CameraInfo* camera = nullptr;
  if (config_.draw_axes) {
    auto info = camera_.update(camera_info);
    if (!info) {
      holoscan::Error error = std::move(info).error();
      error.message = "FiducialOverlayOp: " + error.message;
      return holoscan::make_unexpected(std::move(error));
    }
    if (*info == nullptr) return {};  // No camera info yet: skip these detections.
    camera = *info;
  }
  auto received = detections.receive();
  if (!received) return holoscan::make_unexpected(std::move(received).error());
  const schema::FiducialDetectionsT& message = received->data;
  const int width = message.image_width;
  const int height = message.image_height;
  if (width <= 0 || height <= 0 || width > config_.width || height > config_.height) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "FiducialOverlayOp: image size outside the configured " +
                            std::to_string(config_.width) + "x" + std::to_string(config_.height)});
  }
  if (camera != nullptr) {
    try {
      check_camera_size(*camera, width, height);
    } catch (const std::invalid_argument& error) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                          std::string("FiducialOverlayOp: ") + error.what()});
    }
  }
  const std::size_t markers = std::min<std::size_t>(message.markers.size(), kMaxMarkersPerFrame);
  // The frame's capture time, so HoloViz draws these over the image they came from. HoloViz
  // selects layers by capture time, so unstamped detections cannot be drawn without one: they
  // get this activation's time, which says when they were drawn, not captured, and a warning
  // says so once.
  holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*received);
  if (!received->metadata.source_clock_id.valid()) {
    emit.capture_time = context.activation_time();
    if (!unstamped_logged_) {
      unstamped_logged_ = true;
      HOLOSCAN_LOG_WARN(
          "FiducialOverlayOp: detections without a capture time; drawn with the time they are "
          "drawn at, so the overlay may not match its image");
    }
  }

  // Allocates a host tensor on `output` and has `fill` write it.
  const auto write =
      [](auto& output, holoscan::TensorShapeView shape,
         const auto& fill) -> holoscan::expected<holoscan::TensorOutputLoan, holoscan::Error> {
    auto loan =
        output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) return holoscan::make_unexpected(std::move(loan).error());
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      fill(writer.data_as<std::uint8_t>());
      return {};
    });
    if (!written) return holoscan::make_unexpected(std::move(written).error());
    return std::move(*loan);
  };

  // Transparent where nothing is drawn, so the previous frame's drawing disappears.
  const std::array<std::int64_t, 3> overlay_shape{height, width, 4};
  auto overlay_loan = write(overlay, overlay_shape, [&](std::uint8_t* out) {
    std::memset(out, 0, static_cast<std::size_t>(width) * height * 4);
    const RgbaCanvas canvas{out, width, height};
    for (std::size_t m = 0; m < markers; ++m) {
      const Marker marker = from_message(message.markers[m]);
      draw_marker(canvas, marker.corners);
      if (marker.pose && camera != nullptr) {
        if (const auto axes = target_axes(*marker.pose, *camera, 0.5 * config_.marker_length)) {
          draw_axes(canvas, *axes);
        }
      }
    }
    if (message.board) {
      for (const schema::CharucoCorner& corner : message.board->corners) {
        draw_corner(canvas, {corner.image().x(), corner.image().y()});
      }
      if (message.board->has_pose && message.board->pose && camera != nullptr) {
        if (const auto axes = target_axes(from_message(*message.board->pose), *camera,
                                          2.0 * message.board->square_length)) {
          draw_axes(canvas, *axes);
        }
      }
    }
  });
  if (!overlay_loan) return holoscan::make_unexpected(std::move(overlay_loan).error());

  const std::array<std::int64_t, 2> label_shape{kOverlayLabelRows, kLabelRowBytes};
  auto label_loan = write(labels, label_shape, [&](std::uint8_t* out) {
    std::memset(out, 0, std::size_t{kOverlayLabelRows} * kLabelRowBytes);
    const auto label = [&](std::size_t row, float x, float y, const char* text) {
      const viz::TextRunHeader header{.x = x / static_cast<float>(width),
                                      .y = y / static_cast<float>(height),
                                      .size_px = kLabelSizePx,
                                      .color = kLabelColor};
      std::memcpy(out + row * kLabelRowBytes, &header, sizeof(header));
      std::memcpy(out + row * kLabelRowBytes + viz::kTextRunHeaderBytes, text,
                  std::min<std::size_t>(std::strlen(text), kOverlayLabelBytes));
    };
    for (std::size_t m = 0; m < markers; ++m) {
      const schema::Marker& marker = message.markers[m];
      // Above the marker's highest corner, so the label does not cover it.
      float top = marker.corners()->Get(0)->y();
      float left = marker.corners()->Get(0)->x();
      for (int c = 1; c < 4; ++c) {
        top = std::min(top, marker.corners()->Get(c)->y());
        left = std::min(left, marker.corners()->Get(c)->x());
      }
      char text[kOverlayLabelBytes + 1] = {};
      std::snprintf(text, sizeof(text), "id %d", marker.id());
      label(m, left, std::max(0.0F, top - kLabelSizePx - 2.0F), text);
    }
    if (message.board) {
      char text[kOverlayLabelBytes + 1] = {};
      std::snprintf(text, sizeof(text), "corners %zu/%d", message.board->corners.size(),
                    message.board->corners_total);
      label(kOverlayLabelRows - 1, 8.0F, 8.0F, text);
    }
  });
  if (!label_loan) return holoscan::make_unexpected(std::move(label_loan).error());
  // Labels first: the overlay image triggers the frame, which takes the newest labels.
  auto sent = labels.emit(std::move(*label_loan), emit);
  if (!sent) return sent;
  holoscan::schema::ImageT descriptor;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGBA8;
  return overlay.emit_tensor(std::move(*overlay_loan), descriptor, emit);
}

}  // namespace holoscan::fiducials
