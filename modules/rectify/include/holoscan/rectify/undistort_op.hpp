// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// UndistortOp: one camera's images in, undistorted images out, remapped on the
// GPU, the VIC or the PVA. The output camera is computed once, at
// construction, from the calibration and the options.

#pragma once

#include <memory>
#include <string>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/rectify/backend.hpp>
#include <holoscan/rectify/camera_model.hpp>
#include <holoscan/rectify/rectification.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::rectify {

struct UndistortConfig {
  CameraModel camera;
  UndistortOptions options;
  // kAuto lets the operator choose at the first image; see resolve_auto().
  Backend backend{Backend::kCuda};
  // MONO16 images take kNearest only.
  Interpolation interpolation{Interpolation::kLinear};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const UndistortConfig& config);

class UndistortOp final : public holoscan::Operator<> {
 public:
  // Computes the undistortion.
  // @throws std::invalid_argument if `config` is not valid.
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

 private:
  struct State;

  UndistortConfig config_;
  Undistortion undistortion_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::rectify
