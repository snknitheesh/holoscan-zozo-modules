// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// UndistortOp: one camera's images in, undistorted images out, with the
// undistorted camera's camera info. The calibration is read from the camera's
// ROS camera_info file and the output camera computed once, at construction;
// each image is remapped on the GPU, the VIC or the PVA.

#pragma once

#include <filesystem>
#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info_schema_traits.hpp>
#include <holoscan/rectify/backend.hpp>
#include <holoscan/rectify/rectification.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::rectify {

struct UndistortConfig {
  // The camera's ROS camera_info file (messages::read_camera_info_yaml()).
  // Required. Read at construction, and again when a run starts, since the
  // SDK constructs operators again then: keep it until the run has started.
  std::filesystem::path camera_info_path;
  UndistortOptions options;
  // kAuto lets the operator choose at the first image; see resolve_auto().
  Backend backend{Backend::kCuda};
  // MONO16 images take kNearest only.
  Interpolation interpolation{Interpolation::kLinear};
};

// Reads the file and computes the undistortion, as the operator does.
// @throws std::invalid_argument naming the first unusable file or field.
void validate_config(const UndistortConfig& config);

class UndistortOp final : public holoscan::Operator<> {
 public:
  // Reads the calibration and computes the undistortion.
  // @throws std::invalid_argument if `config` or the file is not valid.
  explicit UndistortOp(UndistortConfig config);
  ~UndistortOp() override;
  UndistortOp(const UndistortOp&) = delete;
  UndistortOp& operator=(const UndistortOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The input and output cameras: what the images are undistorted into.
  [[nodiscard]] const Undistortion& undistortion() const noexcept { return undistortion_; }
  // MONO8, MONO16, RGB8, BGR8, RGBA8 or BGRA8 of the camera's size, packed
  // rows, in host or CUDA device memory.
  holoscan::Input<holoscan::schema::ImageT> image;
  // The undistorted image: the input's encoding and header, the output size,
  // in CUDA device memory.
  holoscan::Output<holoscan::schema::ImageT> undistorted;
  // One per image, with its capture time (in its sample and its header):
  // undistortion().output. Must be connected: an emit into an unconnected
  // output fails. A consumer takes it with messages::camera_info_contract().
  holoscan::Output<messages::schema::CameraInfoT> camera_info;

 private:
  struct State;

  UndistortConfig config_;
  Undistortion undistortion_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::rectify
