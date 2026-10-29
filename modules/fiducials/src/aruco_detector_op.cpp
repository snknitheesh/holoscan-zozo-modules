// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/fiducials/aruco_detector_op.hpp"

#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/fiducials/fiducial_messages.hpp>
#include <holoscan/fiducials/pose.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "frame_detector.hpp"

namespace holoscan::fiducials {

void validate_config(const ArucoDetectorConfig& config) {
  validate_options(config.options);
  if (config.intrinsics) {
    validate_intrinsics(*config.intrinsics);
    if (!(std::isfinite(config.marker_length) && config.marker_length > 0)) {
      throw std::invalid_argument("marker_length must be positive to estimate poses");
    }
  }
}

ArucoDetectorOp::ArucoDetectorOp(ArucoDetectorConfig config) : config_(std::move(config)) {
  validate_config(config_);
  detector_ = std::make_unique<FrameDetector>("ArucoDetectorOp", config_.backend, config_.options);
}

ArucoDetectorOp::~ArucoDetectorOp() = default;

Backend ArucoDetectorOp::resolved_backend() const noexcept { return detector_->resolved(); }

void ArucoDetectorOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(image, "image").queue_depth(2U);
  spec.output(detections, "detections");
}

holoscan::Contract ArucoDetectorOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void ArucoDetectorOp::start() { detector_->start(); }

void ArucoDetectorOp::stop() { detector_->stop(); }

holoscan::expected<void, holoscan::Error> ArucoDetectorOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) return holoscan::make_unexpected(std::move(sample).error());
  auto frame = detector_->prepare(sample->data, context);
  if (!frame) return holoscan::make_unexpected(std::move(frame).error());

  FrameMarkers found;
  try {
    found = frame->backend->detect(frame->gray, frame->stream);
    if (config_.intrinsics) {
      for (Marker& marker : found.markers) {
        marker.pose = marker_pose(marker, *config_.intrinsics, config_.marker_length);
      }
    }
  } catch (const std::exception& exception) {
    return holoscan::make_unexpected(detector_->error(exception, frame->stream));
  }

  schema::FiducialDetectionsT message;
  message.frame = frame->number;
  message.device_sequence = sample->data.header ? sample->data.header->device_sequence : 0U;
  message.image_width = frame->gray.width;
  message.image_height = frame->gray.height;
  message.backend = std::string(frame->backend->name());
  message.dictionary = config_.options.dictionary;
  message.markers.reserve(found.markers.size());
  for (const Marker& marker : found.markers) message.markers.push_back(to_message(marker));
  message.dropped = static_cast<std::int32_t>(found.overflow.markers);
  message.truncated = found.overflow.truncated();
  if (found.overflow.truncated() || found.overflow.markers != 0) {
    detector_->warn_overflow(found.overflow);
  }
  message.detect_time_us =
      std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - frame->start)
          .count();
  return detections.emit(std::move(message), holoscan::sensor_io::derive_from(*sample));
}

}  // namespace holoscan::fiducials
