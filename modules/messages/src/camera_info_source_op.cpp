// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/messages/camera_info_source_op.hpp"

#include <stdexcept>
#include <utility>

#include <holoscan/core/sample.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/conversion.hpp>

namespace holoscan::messages {

CameraInfoSourceOp::CameraInfoSourceOp(CameraInfoSourceConfig config) : config_(std::move(config)) {
  if (config_.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("CameraInfoSourceOp: the period must be > 0");
  }
  if (config_.camera_info_path.empty()) {
    throw std::invalid_argument("CameraInfoSourceOp: camera_info_path is required");
  }
  if (config_.right_camera_info_path.empty()) {
    mono_ = to_message(read_camera_info_yaml(config_.camera_info_path));
  } else {
    stereo_ = to_message(
        read_stereo_camera_info_yaml(config_.camera_info_path, config_.right_camera_info_path));
  }
}

void CameraInfoSourceOp::setup(holoscan::OperatorSpec& spec) {
  // An emit into an unconnected output fails, so only the one in use exists.
  if (mono_) {
    spec.output(camera_info, "camera_info");
  } else {
    spec.output(stereo_camera_info, "stereo_camera_info");
  }
}

holoscan::Contract CameraInfoSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void CameraInfoSourceOp::start() { sent_ = 0; }

holoscan::expected<void, holoscan::Error> CameraInfoSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (config_.count != 0 && sent_ >= config_.count) return {};
  const holoscan::TimePoint now = context.activation_time();
  const holoscan::EmitOptions options{.capture_time = now, .frame_id = sent_ + 1U};
  holoscan::expected<void, holoscan::Error> sent;
  if (mono_) {
    mono_->header->capture_timestamp_ns = now.timestamp_ns;
    sent = camera_info.emit_ref(*mono_, options);
  } else {
    for (holoscan::schema::HeaderT* header :
         {stereo_->header.get(), stereo_->left->header.get(), stereo_->right->header.get()}) {
      header->capture_timestamp_ns = now.timestamp_ns;
    }
    sent = stereo_camera_info.emit_ref(*stereo_, options);
  }
  if (!sent) {
    // A consumer with no room keeps the calibration it has; send again next time.
    if (sent.error().code == holoscan::ErrorCode::kBackpressured) return {};
    return sent;
  }
  ++sent_;
  return {};
}

}  // namespace holoscan::messages
