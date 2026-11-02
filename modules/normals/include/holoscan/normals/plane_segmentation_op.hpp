// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// PlaneSegmentationOp: the planes of a cloud with normals, by RANSAC with
// normal agreement on the GPU, and which plane each point is on. See
// planes.hpp for the method and cloud_op.hpp for the cloud.
//
//   points              (input) float32 [N, 3] or [H, W, 3].
//   normals             (input) float32 of the points' shape, as
//                       NormalEstimationOp emits them; a point with a
//                       non-finite normal is on no plane.
//   labels              uint8, one per point ([H, W] or [N]): plane k is
//                       k + 1; kLabelNoPlane (0) on none, kLabelInvalid (255)
//                       for an invalid point or one without a normal.
//   plane_coefficients  float32 [max_planes, 4] (a, b, c, d), (a, b, c) a unit
//                       normal toward the viewpoint; NaN past the planes found.
//   plane_centroids     float32 [max_planes, 3], the mean of each plane's
//                       inliers; NaN past the planes found.
//   plane_inliers       uint32 [max_planes], each plane's inliers; 0 past the
//                       planes found.
//
// The operator runs once per cloud when the normals with exactly the same
// capture time have arrived. Every output is in device memory and carries the
// cloud's capture time and frame id; they go out in the order above. Nothing
// waits for the GPU.

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
#include <holoscan/normals/planes.hpp>

namespace holoscan::normals {

struct PlaneSegmentationConfig {
  PlaneParams planes;
  // Largest cloud accepted, 1..kMaxPointsLimit points; a larger one fails the
  // frame. It sizes the labels. The default fits 1920x1080.
  std::uint32_t max_points{1920U * 1080U};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const PlaneSegmentationConfig& config);

class PlaneSegmentationOp final : public CloudOp {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit PlaneSegmentationOp(PlaneSegmentationConfig config);
  ~PlaneSegmentationOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> points;
  holoscan::Input<holoscan::Tensor> normals;
  holoscan::Output<holoscan::Tensor> labels;
  holoscan::Output<holoscan::Tensor> plane_coefficients;
  holoscan::Output<holoscan::Tensor> plane_centroids;
  holoscan::Output<holoscan::Tensor> plane_inliers;

 private:
  PlaneSegmentationConfig config_;
  std::unique_ptr<PlaneSegmenter> segmenter_;
};

}  // namespace holoscan::normals
