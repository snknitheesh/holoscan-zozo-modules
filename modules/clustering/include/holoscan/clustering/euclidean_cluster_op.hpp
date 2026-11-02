// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// EuclideanClusterOp: splits a point cloud into objects, the connected
// components of points within a tolerance of each other, on the GPU. See
// cloud_op.hpp for the inputs and EuclideanClusterer in clustering.hpp for the
// method. Outputs, in device memory, emitted in this order:
//
//   labels            int32 cluster id per point, -1 for noise and invalid
//                     points: with OutputLayout::kKeep the points' shape
//                     without its last axis ([N] or [H, W]); with kDense,
//                     [M] for the clustered points grouped by cluster.
//   clusters          float32 [K, kSummaryFields]: one ClusterSummary per
//                     cluster, by id, largest first.
//   clustered_points  float32 (with `emit_points`): with kKeep the input's
//                     shape, unclustered points NaN; with kDense [M, 3], the
//                     clustered points grouped by cluster, in input order
//                     within one. Cluster k's rows start at the sum of the
//                     point counts of the clusters before it.
//   clustered_colors  uint8 RGB of the clustered points' shape (with
//                     `color`), black where a point is not clustered.
//
// The number of clusters is known only once the GPU has found them, so the
// operator waits for its GPU work once per frame (twice with kEuclidean, as
// the grid's size is needed too, and once more with kDense).

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/clustering/cloud_op.hpp>
#include <holoscan/clustering/clustering.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>

namespace holoscan::clustering {

struct EuclideanClusterConfig {
  ClusterParams clustering;
  // Fill the summaries' oriented boxes (PCA), at the cost of one more pass.
  bool oriented_boxes{false};
  // Declare the `clustered_points` output.
  bool emit_points{false};
  // Declare the `colors` input and the `clustered_colors` output; needs
  // `emit_points`.
  bool color{false};
  // Labels and clustered points per input point, or packed by cluster.
  OutputLayout layout{OutputLayout::kKeep};
  // Largest cloud accepted; see CloudSettings.
  std::uint32_t max_points{1920U * 1080U};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const EuclideanClusterConfig& config);

class EuclideanClusterOp final : public CloudOp {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit EuclideanClusterOp(EuclideanClusterConfig config);
  ~EuclideanClusterOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::Tensor> labels;
  holoscan::Output<holoscan::Tensor> clusters;
  holoscan::Output<holoscan::Tensor> clustered_points;
  holoscan::Output<holoscan::Tensor> clustered_colors;

 private:
  EuclideanClusterConfig config_;
  std::unique_ptr<EuclideanClusterer> clusterer_;
};

}  // namespace holoscan::clustering
