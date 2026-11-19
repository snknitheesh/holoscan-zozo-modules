// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StereoRectifyOp: a stereo pair's left and right images in, rectified images
// out, with the rectified pair's camera info, which a disparity or depth
// operator takes on its camera_info input. The calibration is read from the
// pair's ROS camera_info files and the rectification computed once, at
// construction; each pair is remapped on the GPU, the VIC or the PVA.

#pragma once

#include <filesystem>
#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/stereo_camera_info_schema_traits.hpp>
#include <holoscan/rectify/backend.hpp>
#include <holoscan/rectify/rectification.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::rectify {

struct StereoRectifyConfig {
  // The pair's ROS camera_info files, as a stereo calibration writes them
  // (messages::read_stereo_camera_info_yaml()). Both required. The operator
  // reads them at construction, and again when a run starts, since the SDK
  // constructs operators again then: keep them until the run has started.
  std::filesystem::path left_camera_info_path;
  std::filesystem::path right_camera_info_path;
  StereoRectifyOptions options;
  // kAuto lets the operator choose at the first pair; see resolve_auto().
  Backend backend{Backend::kCuda};
  // MONO16 images take kNearest only.
  Interpolation interpolation{Interpolation::kLinear};
};

// Reads the files and computes the rectification, as the operator does.
// @throws std::invalid_argument naming the first unusable file or field.
void validate_config(const StereoRectifyConfig& config);

class StereoRectifyOp final : public holoscan::Operator<> {
 public:
  // Reads the calibration and computes the rectification.
  // @throws std::invalid_argument if `config` or a file is not valid.
  explicit StereoRectifyOp(StereoRectifyConfig config);
  ~StereoRectifyOp() override;
  StereoRectifyOp(const StereoRectifyOp&) = delete;
  StereoRectifyOp& operator=(const StereoRectifyOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The calibration read from the files.
  [[nodiscard]] const messages::StereoCameraInfo& calibration() const noexcept {
    return calibration_;
  }
  // The rectification, available from construction, so that a graph can size
  // what comes after the operator before it runs.
  [[nodiscard]] const StereoRectification& rectification() const noexcept { return rectification_; }

  // The left and right images of one capture time: the same encoding, MONO8,
  // MONO16, RGB8, BGR8, RGBA8 or BGRA8, each of its camera's size, packed
  // rows, in host or CUDA device memory.
  holoscan::Input<holoscan::schema::ImageT> left;
  holoscan::Input<holoscan::schema::ImageT> right;
  // The rectified images: their input's encoding and header, the output size,
  // in CUDA device memory, with the pair's capture time.
  holoscan::Output<holoscan::schema::ImageT> rectified_left;
  holoscan::Output<holoscan::schema::ImageT> rectified_right;
  // One per pair, with the pair's capture time (in its sample and its
  // headers): rectification().camera_info, the rectified pair. Must be
  // connected: an emit into an unconnected output fails. A consumer takes it
  // with messages::camera_info_contract(), kEveryFrame or kLatest.
  holoscan::Output<messages::schema::StereoCameraInfoT> camera_info;

 private:
  struct State;

  StereoRectifyConfig config_;
  messages::StereoCameraInfo calibration_;
  StereoRectification rectification_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::rectify
