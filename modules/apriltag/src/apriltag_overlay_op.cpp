// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/apriltag/apriltag_overlay_op.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

#include <holoscan/apriltag/apriltag_detector_op.hpp>
#include <holoscan/apriltag/tag_axes.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "image_output.hpp"

namespace holoscan::apriltag {
namespace {

namespace viz = holoscan::visualization;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr std::uint32_t kLabelRowBytes = viz::kTextRunHeaderBytes + kOverlayLabelBytes;
constexpr std::uint32_t kLabelColor = 0xFF8B0000U;  // Opaque dark blue, 0xAABBGGRR.
constexpr float kLabelSizePx = 26.0F;

}  // namespace

void AprilTagOverlayOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(detections, "detections").queue_depth(2U);
  image_output(spec, axes, "axes", holoscan::MemoryKind::kHost, config_.width, config_.height, 4);
  spec.output(labels, "labels")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kHost,
                  .dtype = kUInt8,
                  .rank = 2U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(std::uint64_t{kMaxDetectionsPerFrame} * kLabelRowBytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract AprilTagOverlayOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{detections});
  return result;
}

holoscan::expected<void, holoscan::Error> AprilTagOverlayOp::compute(
    holoscan::ExecutionContext& context) {
  auto received = detections.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  const schema::AprilTagDetectionsT& message = received->data;
  const int width = message.image_width;
  const int height = message.image_height;
  if (width <= 0 || height <= 0 || width > config_.width || height > config_.height) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "AprilTagOverlayOp: image size outside the configured " +
                            std::to_string(config_.width) + "x" + std::to_string(config_.height)});
  }
  const float sx = 1.0F / static_cast<float>(width);
  const float sy = 1.0F / static_cast<float>(height);
  const std::size_t tags = std::min(message.detections.size(), kMaxDetectionsPerFrame);
  // The frame's capture time, so Holoviz draws these over the image they came from;
  // unstamped detections are stamped now, since Holoviz selects layers by capture time.
  holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*received);
  if (!received->metadata.source_clock_id.valid()) {
    emit.capture_time = context.activation_time();
  }

  // Allocates a host tensor on `output` and has `fill` write it.
  const auto write =
      [](auto& output, holoscan::TensorShapeView shape,
         const auto& fill) -> holoscan::expected<holoscan::TensorOutputLoan, holoscan::Error> {
    auto loan =
        output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      fill(writer.data_as<std::uint8_t>());
      return {};
    });
    if (!written) {
      return holoscan::make_unexpected(std::move(written).error());
    }
    return std::move(*loan);
  };

  // Transparent without tags, so the previous frame's axes disappear.
  const std::array<std::int64_t, 3> axes_shape{height, width, 4};
  auto axes_loan = write(axes, axes_shape, [&](std::uint8_t* out) {
    std::memset(out, 0, static_cast<std::size_t>(width) * height * 4);
    for (std::size_t t = 0; t < tags; ++t) {
      if (const auto tag =
              tag_axes(from_message(message.detections[t]), config_.intrinsics, config_.tag_size)) {
        draw_axes({out, width, height}, *tag);
      }
    }
  });
  if (!axes_loan) {
    return holoscan::make_unexpected(std::move(axes_loan).error());
  }

  const std::array<std::int64_t, 2> label_shape{static_cast<std::int64_t>(kMaxDetectionsPerFrame),
                                                kLabelRowBytes};
  auto label_loan = write(labels, label_shape, [&](std::uint8_t* out) {
    std::memset(out, 0, kMaxDetectionsPerFrame * kLabelRowBytes);
    for (std::size_t t = 0; t < tags; ++t) {
      const schema::TagDetection& tag = message.detections[t];
      std::uint8_t* row = out + t * kLabelRowBytes;
      // Above the tag's highest corner, so the label does not cover it.
      float top = tag.corners()->Get(0)->y();
      float left = tag.corners()->Get(0)->x();
      for (int c = 1; c < 4; ++c) {
        top = std::min(top, tag.corners()->Get(c)->y());
        left = std::min(left, tag.corners()->Get(c)->x());
      }
      const viz::TextRunHeader header{.x = left * sx,
                                      .y = std::max(0.0F, top - kLabelSizePx - 4.0F) * sy,
                                      .size_px = kLabelSizePx,
                                      .color = kLabelColor};
      std::memcpy(row, &header, sizeof(header));
      char text[kOverlayLabelBytes + 1] = {};
      if (tag.has_pose()) {
        // The tag's distance from the camera.
        const auto* t3 = tag.translation();
        std::snprintf(
            text, sizeof(text), "id %d %.2fm", tag.id(),
            std::sqrt(t3->Get(0) * t3->Get(0) + t3->Get(1) * t3->Get(1) + t3->Get(2) * t3->Get(2)));
      } else {
        std::snprintf(text, sizeof(text), "id %d", tag.id());
      }
      std::memcpy(row + viz::kTextRunHeaderBytes, text, std::strlen(text));
    }
  });
  if (!label_loan) {
    return holoscan::make_unexpected(std::move(label_loan).error());
  }
  // Labels first: the axes image triggers the frame, which takes the newest labels.
  if (auto sent = labels.emit(std::move(*label_loan), emit); !sent) {
    return sent;
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGBA8;
  return axes.emit_tensor(std::move(*axes_loan), descriptor, emit);
}

}  // namespace holoscan::apriltag
