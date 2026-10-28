// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_camera/calibration_message.hpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace holoscan::stereo_camera {
namespace {

// The schema's enum has the same values as DistortionModel, in the same order.
static_assert(static_cast<int>(schema::DistortionModel_RATIONAL_POLYNOMIAL) ==
              static_cast<int>(DistortionModel::kRationalPolynomial));

std::shared_ptr<schema::CameraModelT> camera_message(const CameraModel& camera) {
  auto message = std::make_shared<schema::CameraModelT>();
  message->width = camera.width;
  message->height = camera.height;
  message->fx = camera.fx;
  message->fy = camera.fy;
  message->cx = camera.cx;
  message->cy = camera.cy;
  message->distortion_model = static_cast<schema::DistortionModel>(camera.distortion_model);
  message->distortion = camera.distortion;
  return message;
}

std::shared_ptr<schema::RigidTransform> transform_message(const RigidTransform& transform) {
  return std::make_shared<schema::RigidTransform>(
      ::flatbuffers::span<const double, 9>(transform.rotation),
      ::flatbuffers::span<const double, 3>(transform.translation));
}

CameraModel camera_of(const std::shared_ptr<schema::CameraModelT>& message,
                      const std::string& field) {
  if (!message) {
    throw std::invalid_argument(field + ": missing");
  }
  const int model = static_cast<int>(message->distortion_model);
  if (model < 0 || model > static_cast<int>(DistortionModel::kRationalPolynomial)) {
    throw std::invalid_argument(field + ": unknown distortion model " + std::to_string(model));
  }
  return CameraModel{.width = message->width,
                     .height = message->height,
                     .fx = message->fx,
                     .fy = message->fy,
                     .cx = message->cx,
                     .cy = message->cy,
                     .distortion_model = static_cast<DistortionModel>(model),
                     .distortion = message->distortion};
}

RigidTransform transform_of(const std::shared_ptr<schema::RigidTransform>& message,
                            const std::string& field) {
  if (!message) {
    throw std::invalid_argument(field + ": missing");
  }
  RigidTransform transform;
  for (std::size_t i = 0; i < 9; ++i) transform.rotation[i] = message->rotation()->Get(i);
  for (std::size_t i = 0; i < 3; ++i) transform.translation[i] = message->translation()->Get(i);
  return transform;
}

}  // namespace

schema::StereoCalibrationT to_message(const StereoCalibration& calibration) {
  schema::StereoCalibrationT message;
  message.left = camera_message(calibration.left);
  message.right = camera_message(calibration.right);
  message.right_from_left = transform_message(calibration.right_from_left);
  message.rectified = calibration.rectified;
  if (calibration.rgb) {
    message.rgb = camera_message(calibration.rgb->model);
    message.rgb_from_left = transform_message(calibration.rgb->from_left);
  }
  if (calibration.depth) {
    message.depth = camera_message(calibration.depth->model);
    message.depth_from_left = transform_message(calibration.depth->from_left);
    message.depth_scale = calibration.depth->scale;
  }
  message.camera_name = calibration.camera.name;
  message.camera_serial = calibration.camera.serial;
  message.camera_firmware = calibration.camera.firmware;
  return message;
}

StereoCalibration from_message(const schema::StereoCalibrationT& message) {
  StereoCalibration calibration;
  calibration.left = camera_of(message.left, "left");
  calibration.right = camera_of(message.right, "right");
  calibration.right_from_left = transform_of(message.right_from_left, "right_from_left");
  calibration.rectified = message.rectified;
  if (message.rgb) {
    calibration.rgb = AuxiliaryCamera{.model = camera_of(message.rgb, "rgb"),
                                      .from_left = transform_of(message.rgb_from_left, "rgb")};
  }
  if (message.depth) {
    calibration.depth = DepthCamera{.model = camera_of(message.depth, "depth"),
                                    .from_left = transform_of(message.depth_from_left, "depth"),
                                    .scale = message.depth_scale};
  }
  calibration.camera.name = message.camera_name;
  calibration.camera.serial = message.camera_serial;
  calibration.camera.firmware = message.camera_firmware;
  validate(calibration);
  return calibration;
}

}  // namespace holoscan::stereo_camera
