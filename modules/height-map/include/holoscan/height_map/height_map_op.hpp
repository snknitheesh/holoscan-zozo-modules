// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// HeightMapOp: a 2.5D elevation map of point clouds on the GPU, fixed in the
// map frame or following the sensor. See height_mapper.hpp for the method,
// grid.hpp for the grid and grid_op.hpp for the outputs.
//
//   points             (input) float32 [N, 3] or organized [H, W, 3], packed,
//                      in host, pinned or device memory on the operator's
//                      GPU, in the sensor frame; a point with a non-finite
//                      coordinate is invalid (holoscan-point-cloud's layout).
//   pose               (input, with `pose_input`) float32 or float64 [4, 4],
//                      row-major, host or device memory: the rigid transform
//                      from the cloud's frame into the map frame, with the
//                      cloud's capture time.
//   elevation          float32 [rows, cols], meters, NaN where unknown.
//   variance           (outputs.variance) float32 [rows, cols], m^2.
//   count              (outputs.count) float32 [rows, cols], the points of the
//                      cell's latest observation.
//   min_elevation,
//   max_elevation      (outputs.min_max) float32 [rows, cols], meters, the
//                      lowest and highest point of the latest observation.
//   time_since_update  (outputs.time_since_update) float32 [rows, cols],
//                      seconds since the cell's latest observation.
//   map_info           float64 [kMapInfoSize], host memory: the grid of this
//                      map (in MapMode::kFollow it moves).
//   elevation_image    (outputs.image) RGB8 [rows, cols, 3] schema::ImageT of
//                      the elevation (views.hpp), its header holding
//                      `frame_id` and the capture time.
//
// The operator runs once per cloud; with `pose_input`, when the pose with
// exactly the same capture time has arrived. The map's time is the cloud's
// capture time, or the activation time when it has none. Every output carries
// the cloud's capture time and frame id; the layers are in device memory and
// nothing waits for the GPU. Each output exists only when configured, and
// every output that exists must be connected.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/height_map/grid_op.hpp>
#include <holoscan/height_map/height_mapper.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::height_map {

// The optional outputs.
struct HeightMapOutputs {
  bool variance{true};
  bool count{false};
  bool min_max{false};
  bool time_since_update{false};
  bool image{false};
};

struct HeightMapConfig {
  HeightMapParams map;
  // Declare the `pose` input; without it every cloud is taken into the map
  // frame by `sensor_pose`.
  bool pose_input{false};
  Transform sensor_pose;
  // The map frame's name, for the image header.
  std::string frame_id{"map"};
  HeightMapOutputs outputs;
  // The elevation image's color range in meters; with image_low >=
  // image_high, each map's own.
  float image_low{0.0F};
  float image_high{0.0F};
  // Largest cloud accepted, 1..kMaxPointsLimit points; a larger one fails the
  // frame. The default fits 1920x1080.
  std::uint32_t max_points{1920U * 1080U};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const HeightMapConfig& config);

class HeightMapOp final : public GridOp {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit HeightMapOp(HeightMapConfig config);
  ~HeightMapOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> points;
  holoscan::Input<holoscan::Tensor> pose;
  holoscan::Output<holoscan::Tensor> elevation;
  holoscan::Output<holoscan::Tensor> variance;
  holoscan::Output<holoscan::Tensor> count;
  holoscan::Output<holoscan::Tensor> min_elevation;
  holoscan::Output<holoscan::Tensor> max_elevation;
  holoscan::Output<holoscan::Tensor> time_since_update;
  holoscan::Output<holoscan::Tensor> map_info;
  holoscan::Output<holoscan::schema::ImageT> elevation_image;

 private:
  struct Impl;

  HeightMapConfig config_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::height_map
