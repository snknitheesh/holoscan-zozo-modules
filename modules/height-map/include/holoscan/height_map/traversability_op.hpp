// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// TraversabilityOp: slope, roughness, step height and a 0..1 cost of an
// elevation map on the GPU. See traversability.hpp for the method and
// grid_op.hpp for the outputs.
//
//   elevation        (input) float32 [rows, cols], NaN where unknown, host or
//                    device memory, as HeightMapOp emits it.
//   map_info         (input) float64 [kMapInfoSize] with the elevation's
//                    capture time: the grid, for its resolution.
//   cost             float32 [rows, cols], 0..1, NaN where unknown.
//   unknown          uint8 [rows, cols], 1 where the cost is unknown.
//   slope            (with `components`) float32 [rows, cols], degrees.
//   step_height      (with `components`) float32 [rows, cols], meters.
//   roughness        (with `components`) float32 [rows, cols], meters.
//   cost_image       (with `image`) RGB8 [rows, cols, 3] schema::ImageT of the
//                    cost (views.hpp), its header holding `frame_id` and the
//                    capture time.
//
// The operator runs once per elevation layer, when the map_info with exactly
// the same capture time has arrived. Every output carries the elevation's
// capture time and frame id; the layers are in device memory and nothing
// waits for the GPU, except to read map_info from device memory. Each output
// exists only when configured, and every output that exists must be
// connected.

#pragma once

#include <memory>
#include <string>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/height_map/grid_op.hpp>
#include <holoscan/height_map/traversability.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::height_map {

struct TraversabilityConfig {
  TraversabilityParams traversability;
  // Declare the slope, step_height and roughness outputs.
  bool components{false};
  // Declare the cost_image output.
  bool image{false};
  // The map frame's name, for the image header.
  std::string frame_id{"map"};
  // The largest map accepted, rows and cols; it sizes the outputs.
  int max_rows{1024};
  int max_cols{1024};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const TraversabilityConfig& config);

class TraversabilityOp final : public GridOp {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit TraversabilityOp(TraversabilityConfig config);
  ~TraversabilityOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> elevation;
  holoscan::Input<holoscan::Tensor> map_info;
  holoscan::Output<holoscan::Tensor> cost;
  holoscan::Output<holoscan::Tensor> unknown;
  holoscan::Output<holoscan::Tensor> slope;
  holoscan::Output<holoscan::Tensor> step_height;
  holoscan::Output<holoscan::Tensor> roughness;
  holoscan::Output<holoscan::schema::ImageT> cost_image;

 private:
  struct Impl;

  TraversabilityConfig config_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::height_map
