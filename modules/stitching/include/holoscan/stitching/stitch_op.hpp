// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StitchOp: one image from each camera of a rig in, one stitched image out,
// on the GPU: a cylindrical or spherical panorama, or a bird's-eye view of the
// ground. Each camera's calibration arrives as holoscan-messages' CameraInfo
// on a port of its own; its pose on the rig is in the configuration. The
// remap tables are built on the GPU when every camera's calibration has
// arrived, and again when one changes; each frame is sampled bilinearly and
// blended, with optional exposure compensation.
//
// The operator runs once per frame set: when every camera's image with the
// same capture time (within `sync_window`) has arrived.

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/stitching/camera_rig.hpp>
#include <holoscan/stitching/projection.hpp>
#include <holoscan/stitching/remap.hpp>

namespace holoscan::stitching {

struct StitchConfig {
  // Each camera's pose on the rig, camera_from_rig, in the frame its camera
  // info's intrinsics describe; 1..kMaxCameras of them. Camera i's images
  // arrive on input `image_<i>` and its calibration on `camera_info_<i>`.
  std::vector<RigidTransform> camera_from_rig;
  // What the output shows, and its size.
  OutputGeometry output;
  // Blending, exposure compensation, the output format and background.
  BlendSettings blend;
  // The largest spread of the capture times of one frame set; 0 pairs only
  // images with exactly the same capture time. Keep it below half the frame
  // period: wider, a dropped image could be paired with its neighbours of
  // the next frame set. A frame set that begins within 2 x sync_window of the
  // previous one fails the run.
  std::chrono::nanoseconds sync_window{0};
  // How the camera info arrives: with kLatest (the default), at any rate,
  // each frame set using the newest received before it; with kEveryFrame,
  // one per camera with every frame set, carrying its capture time.
  messages::CameraInfoInputOptions camera_info;
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
  // Forgets the camera info and the tables of a previous run.
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The tables' GPU memory, once built; 0 before.
  [[nodiscard]] std::size_t table_bytes() const noexcept;

  // Input i, `image_<i>`, for camera i: MONO8, RGB8, BGR8, RGBA8 or BGRA8 of
  // the size its camera info gives, uint8 [H, W] or [H, W, channels], rows
  // possibly padded, in host or CUDA device memory. Only the first
  // camera_from_rig.size() are declared.
  std::array<holoscan::Input<holoscan::schema::ImageT>, kMaxCameras> images;
  // Input i, `camera_info_<i>`, mandatory for each declared camera: its
  // calibration, read from a ROS camera_info file by CameraInfoSourceOp or
  // sent by an operator upstream. No skew, region of interest or binning.
  std::array<holoscan::Input<messages::schema::CameraInfoT>, kMaxCameras> camera_info;
  // RGB8 or RGBA8 [H, W, channels] of the output geometry's size, in CUDA
  // device memory, with the earliest capture time of its frame set.
  holoscan::Output<holoscan::schema::ImageT> stitched;

 private:
  struct State;
  StitchConfig config_;
  std::array<std::string, kMaxCameras> image_port_names_;
  std::array<std::string, kMaxCameras> camera_info_port_names_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::stitching
