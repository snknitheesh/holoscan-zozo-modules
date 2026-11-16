// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// NormalEstimationOp: each point's surface normal, oriented toward a
// viewpoint, and its curvature, on the GPU. See normals.hpp for the method
// and cloud_op.hpp for the cloud.
//
//   point_cloud               (input) a PointCloud of float32 [N, 3] or
//                             [H, W, 3] points, with or without colors and
//                             intensities. Normals it carries are replaced.
//   point_cloud_with_normals  a PointCloud with the input's points, colors
//                             and intensities and the normals. With
//                             OutputLayout::kKeep it has the input's shape,
//                             and a point without a normal is NaN (black,
//                             NaN intensity); with kDense it is [M, 3] of the
//                             points with a normal, in input order, and
//                             counts them in valid_count.
//   curvature                 float32, one per point of the output cloud
//                             ([H, W], [N] or [M]): the surface variation,
//                             NaN where a point has no normal.
//
// Both outputs are in device memory and carry the cloud's capture time and
// frame id; the cloud goes out first. kKeep never waits for the GPU, once
// its scratch has reached the clouds' size and with input in device memory;
// a dense output's size is known only once the normals are, so kDense waits
// for that count once per frame. Host input is copied to the GPU every frame
// (see README.md, "Waits and copies").
//
// The operator is configured only by its constructor's NormalEstimationConfig.

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/normals/cloud_op.hpp>
#include <holoscan/normals/normals.hpp>

namespace holoscan::normals {

struct NormalEstimationConfig {
  NormalParams normals;
  // Room in the output for the clouds' colors and intensities, which it then
  // carries through; a cloud that carries a field without room fails the
  // frame, one without the field leaves it absent in the output.
  bool colors{false};
  bool intensity{false};
  OutputLayout layout{OutputLayout::kKeep};
  // Largest cloud accepted, 1..kMaxPointsLimit points; a larger one fails the
  // frame. It sizes the outputs' pools. The default fits 1920x1080.
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

  holoscan::Input<holoscan::messages::schema::PointCloudT> point_cloud;
  holoscan::Output<holoscan::messages::schema::PointCloudT> point_cloud_with_normals;
  holoscan::Output<holoscan::Tensor> curvature;

 private:
  struct Impl;

  NormalEstimationConfig config_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::normals
