// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/messages/conversion.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace holoscan::messages {
namespace {

holoscan::schema::Matrix3d matrix_message(const Matrix3& m) {
  return holoscan::schema::Matrix3d(::flatbuffers::span<const double, 9>(m));
}

Matrix3 matrix_value(const holoscan::schema::Matrix3d& m) {
  Matrix3 result{};
  std::copy(m.m()->begin(), m.m()->end(), result.begin());
  return result;
}

schema::CameraInfoT camera_message(const CameraInfo& info) {
  schema::CameraInfoT message;
  message.header = std::make_shared<holoscan::schema::HeaderT>(to_message(info.header));
  message.width = info.width;
  message.height = info.height;
  message.state = static_cast<schema::CameraState>(info.state);
  message.fx = info.fx;
  message.fy = info.fy;
  message.cx = info.cx;
  message.cy = info.cy;
  message.skew = info.skew;
  message.distortion_model = static_cast<schema::DistortionModel>(info.distortion_model);
  message.coefficients = info.coefficients;
  if (info.rectification) {
    message.rectification =
        std::make_shared<holoscan::schema::Matrix3d>(matrix_message(*info.rectification));
  }
  if (info.roi) {
    message.roi = std::make_shared<schema::RegionOfInterest>(info.roi->x, info.roi->y,
                                                             info.roi->width, info.roi->height);
  }
  message.binning_x = info.binning_x;
  message.binning_y = info.binning_y;
  return message;
}

CameraInfo camera_value(const schema::CameraInfoT& message) {
  CameraInfo info;
  if (message.header) info.header = from_message(*message.header);
  info.width = message.width;
  info.height = message.height;
  info.state = static_cast<CameraState>(message.state);
  info.fx = message.fx;
  info.fy = message.fy;
  info.cx = message.cx;
  info.cy = message.cy;
  info.skew = message.skew;
  info.distortion_model = static_cast<DistortionModel>(message.distortion_model);
  info.coefficients = message.coefficients;
  if (message.rectification) info.rectification = matrix_value(*message.rectification);
  if (message.roi) {
    info.roi = RegionOfInterest{message.roi->x(), message.roi->y(), message.roi->width(),
                                message.roi->height()};
  }
  info.binning_x = std::max(message.binning_x, 1U);
  info.binning_y = std::max(message.binning_y, 1U);
  return info;
}

}  // namespace

holoscan::schema::HeaderT to_message(const Header& header) {
  holoscan::schema::HeaderT message;
  message.frame_id = header.frame_id;
  message.capture_timestamp_ns = header.capture_timestamp_ns;
  message.device_sequence = header.device_sequence;
  return message;
}

Header from_message(const holoscan::schema::HeaderT& header) {
  return Header{header.frame_id, header.capture_timestamp_ns, header.device_sequence};
}

schema::CameraInfoT to_message(const CameraInfo& info) {
  validate(info);
  return camera_message(info);
}

schema::StereoCameraInfoT to_message(const StereoCameraInfo& info) {
  validate(info);
  schema::StereoCameraInfoT message;
  message.header = std::make_shared<holoscan::schema::HeaderT>(to_message(info.header));
  message.left = std::make_shared<schema::CameraInfoT>(camera_message(info.left));
  message.right = std::make_shared<schema::CameraInfoT>(camera_message(info.right));
  const Vector3& t = info.right_from_left.translation;
  message.right_from_left = std::make_shared<schema::RigidTransform>(
      matrix_message(info.right_from_left.rotation), holoscan::schema::Vector3d(t[0], t[1], t[2]));
  message.baseline = info.baseline;
  message.doffs = info.doffs;
  return message;
}

CameraInfo from_message(const schema::CameraInfoT& message) {
  CameraInfo info = camera_value(message);
  validate(info);
  return info;
}

StereoCameraInfo from_message(const schema::StereoCameraInfoT& message) {
  if (!message.left || !message.right || !message.right_from_left) {
    throw std::invalid_argument(
        "stereo_camera_info: left, right and right_from_left must all be present");
  }
  StereoCameraInfo info;
  if (message.header) info.header = from_message(*message.header);
  info.left = camera_value(*message.left);
  info.right = camera_value(*message.right);
  info.right_from_left.rotation = matrix_value(message.right_from_left->rotation());
  const holoscan::schema::Vector3d& t = message.right_from_left->translation();
  info.right_from_left.translation = {t.x(), t.y(), t.z()};
  info.baseline = message.baseline;
  info.doffs = message.doffs;
  validate(info);
  return info;
}

}  // namespace holoscan::messages
