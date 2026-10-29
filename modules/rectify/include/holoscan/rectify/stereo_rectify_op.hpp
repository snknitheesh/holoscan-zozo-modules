// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StereoRectifyOp: a stereo pair's left and right images in, rectified images
// out, with optionally the rectified geometry a disparity or depth operator
// needs. The rectification is computed once, at construction, from the
// calibration and the options; each frame is remapped on the GPU, the VIC or
// the PVA.

#pragma once

#include <memory>
#include <string>
#include <string_view>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/rectify/backend.hpp>
#include <holoscan/rectify/camera_model.hpp>
#include <holoscan/rectify/rectification.hpp>
#include <holoscan/rectify/stereo_geometry_schema_traits.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::rectify {

struct StereoRectifyConfig {
  StereoCalibration calibration;
  StereoRectifyOptions options;
  // kAuto lets the operator choose at the first pair; see resolve_auto().
  Backend backend{Backend::kCuda};
  // MONO16 images take kNearest only.
  Interpolation interpolation{Interpolation::kLinear};
  // Declare the `geometry` output: one StereoGeometry message per pair.
  bool geometry_output{false};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const StereoRectifyConfig& config);

// The message StereoRectifyOp sends on `geometry`, for `rectification`
// remapped by `backend`.
[[nodiscard]] schema::StereoGeometryT make_geometry_message(
    const StereoRectification& rectification, std::string_view backend = {});

class StereoRectifyOp final : public holoscan::Operator<> {
 public:
  // Computes the rectification.
  // @throws std::invalid_argument if `config` is not valid.
  explicit StereoRectifyOp(StereoRectifyConfig config);
  ~StereoRectifyOp() override;
  StereoRectifyOp(const StereoRectifyOp&) = delete;
  StereoRectifyOp& operator=(const StereoRectifyOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The rectified geometry, available from construction, so that a graph can
  // configure its disparity operator before it runs.
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
  // With `geometry_output`: the rectified geometry and the backend that
  // remapped the pair (what auto chose), with the pair's capture time.
  holoscan::Output<schema::StereoGeometryT> geometry;

 private:
  struct State;

  StereoRectifyConfig config_;
  StereoRectification rectification_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::rectify
