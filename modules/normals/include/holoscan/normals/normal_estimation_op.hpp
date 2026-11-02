// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// NormalEstimationOp: each point's surface normal, oriented toward a
// viewpoint, and its curvature, on the GPU. See normals.hpp for the method
// and cloud_op.hpp for the cloud.
//
//   points           (input) float32 [N, 3] or [H, W, 3].
//   colors           (input, with `color`) uint8 RGB of the points' shape.
//   filtered_points  float32: with OutputLayout::kKeep the input's shape,
//                    points without a normal NaN; with kDense [M, 3] of the
//                    points with a normal, in input order.
//   filtered_colors  (with `color`) uint8 RGB of filtered_points' shape,
//                    black where a point has no normal.
//   normals          float32 of filtered_points' shape: unit normals, NaN
//                    where a point has none.
//   curvature        float32, one per point of filtered_points ([H, W],
//                    [N] or [M]): the surface variation, NaN where a point
//                    has no normal.
//
// With colors, the operator runs once per cloud when the colors with exactly
// the same capture time have arrived. Every output is in device memory and
// carries the cloud's capture time and frame id; they go out in the order
// above. A dense output's size is known only once the normals are, so kDense
// waits for the GPU once per frame; kKeep never waits.

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/normals/cloud_op.hpp>
#include <holoscan/normals/normals.hpp>

namespace holoscan::normals {

struct NormalEstimationConfig {
  NormalParams normals;
  // Declare the `colors` input and the `filtered_colors` output.
  bool color{false};
  OutputLayout layout{OutputLayout::kKeep};
  // Largest cloud accepted, 1..kMaxPointsLimit points; a larger one fails the
  // frame. It sizes the outputs. The default fits 1920x1080.
  std::uint32_t max_points{1920U * 1080U};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const NormalEstimationConfig& config);

class NormalEstimationOp final : public CloudOp {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit NormalEstimationOp(NormalEstimationConfig config);
  ~NormalEstimationOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> points;
  holoscan::Input<holoscan::Tensor> colors;
  holoscan::Output<holoscan::Tensor> filtered_points;
  holoscan::Output<holoscan::Tensor> filtered_colors;
  holoscan::Output<holoscan::Tensor> normals;
  holoscan::Output<holoscan::Tensor> curvature;

 private:
  struct Impl;

  NormalEstimationConfig config_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::normals
