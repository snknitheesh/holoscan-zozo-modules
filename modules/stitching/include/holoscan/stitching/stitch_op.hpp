// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StitchOp: one image from each camera of a calibrated rig in, one stitched
// image out, on the GPU: a cylindrical or spherical panorama, or a bird's-eye
// view of the ground. The remap tables are built once, at start; each frame
// is sampled bilinearly and blended, with optional exposure compensation.
//
// The operator runs once per frame set: when every camera's image with the
// same capture time (within `sync_window`) has arrived.

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/stitching/camera_rig.hpp>
#include <holoscan/stitching/projection.hpp>
#include <holoscan/stitching/remap.hpp>

namespace holoscan::stitching {

class Stitcher;

struct StitchConfig {
  // 1..kMaxCameras cameras; camera i's images arrive on input `image_<i>`.
  CameraRig rig;
  // What the output shows, and its size.
  OutputGeometry output;
  // Blending, exposure compensation, the output format and background.
  BlendSettings blend;
  // The largest spread of the capture times of one frame set; 0 pairs only
  // images with exactly the same capture time.
  std::chrono::nanoseconds sync_window{0};
  // The output's header frame_id: the rig's coordinate frame.
  std::string frame_id{"rig"};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const StitchConfig& config);

class StitchOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit StitchOp(StitchConfig config);
  ~StitchOp() override;
  StitchOp(const StitchOp&) = delete;
  StitchOp& operator=(const StitchOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  // Builds the remap tables on the GPU.
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The tables' GPU memory, once started; 0 before.
  [[nodiscard]] std::size_t table_bytes() const noexcept;

  // Input i, `image_<i>`, for camera i of the rig: MONO8, RGB8, BGR8, RGBA8 or
  // BGRA8 of the camera's size, uint8 [H, W] or [H, W, channels], rows
  // possibly padded, in host or CUDA device memory. Only the first
  // rig.cameras.size() are declared.
  std::array<holoscan::Input<holoscan::schema::ImageT>, kMaxCameras> images;
  // RGB8 or RGBA8 [H, W, channels] of the output geometry's size, in CUDA
  // device memory, with the earliest capture time of its frame set.
  holoscan::Output<holoscan::schema::ImageT> stitched;

 private:
  struct State;
  StitchConfig config_;
  std::array<std::string, kMaxCameras> port_names_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::stitching
