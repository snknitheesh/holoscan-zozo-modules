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

namespace {

constexpr const char* kName = "ArucoDetectorOp";

}  // namespace

void validate_config(const ArucoDetectorConfig& config) {
  validate_options(config.options);
  if (config.estimate_pose) {
    if (!(std::isfinite(config.marker_length) && config.marker_length > 0)) {
      throw std::invalid_argument("marker_length must be positive to estimate poses");
    }
    if (config.camera_info_input.max_age <= std::chrono::nanoseconds::zero()) {
      throw std::invalid_argument("camera_info_input.max_age must be > 0");
    }
  }
}

ArucoDetectorOp::ArucoDetectorOp(ArucoDetectorConfig config) : config_(std::move(config)) {
  try {
    validate_config(config_);
  } catch (const std::invalid_argument& error) {
    throw std::invalid_argument(std::string(kName) + ": " + error.what());
  }
  detector_ = std::make_unique<FrameDetector>(kName, config_.backend, config_.options);
}

ArucoDetectorOp::~ArucoDetectorOp() = default;

Backend ArucoDetectorOp::resolved_backend() const noexcept { return detector_->resolved(); }

void ArucoDetectorOp::setup(holoscan::OperatorSpec& spec) {
  // Two images queued: the one being detected in and the next, so a producer one frame ahead
  // does not wait on a lossless edge.
  spec.input(image, "image").queue_depth(2U);
  if (config_.estimate_pose) {
    // As for the images: a camera info sent with every image (kEveryFrame) keeps pace with them.
    spec.input(camera_info, "camera_info").queue_depth(2U);
  }
  spec.output(detections, "detections");
}

holoscan::Contract ArucoDetectorOp::contract() const {
  if (config_.estimate_pose) {
    return messages::camera_info_contract(config_.camera_info_input, camera_info, image);
  }
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void ArucoDetectorOp::start() {
  camera_.reset();
  detector_->start();
}

void ArucoDetectorOp::stop() { detector_->stop(); }

holoscan::expected<void, holoscan::Error> ArucoDetectorOp::compute(
    holoscan::ExecutionContext& context) {
  const messages::CameraInfo* camera = nullptr;
  if (config_.estimate_pose) {
    auto info = camera_.update(camera_info);
    if (!info) return holoscan::make_unexpected(named_error(kName, std::move(info).error()));
    if (*info == nullptr) return {};  // No camera info yet: skip the image.
    camera = *info;
  }
  auto sample = image.receive();
  if (!sample) return holoscan::make_unexpected(std::move(sample).error());
  auto frame = detector_->prepare(sample->data, context, camera);
  if (!frame) return holoscan::make_unexpected(std::move(frame).error());

  FrameMarkers found;
  try {
    found = frame->backend->detect(frame->gray, frame->stream);
    if (camera != nullptr) {
      for (Marker& marker : found.markers) {
        marker.pose = marker_pose(marker, *camera, config_.marker_length);
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
  const TransferCounts transfers = detector_->take_transfers();
  message.host_waits = transfers.host_waits;
  message.copied_bytes = transfers.copied_bytes;
  return detections.emit(std::move(message), holoscan::sensor_io::derive_from(*sample));
}

}  // namespace holoscan::fiducials
