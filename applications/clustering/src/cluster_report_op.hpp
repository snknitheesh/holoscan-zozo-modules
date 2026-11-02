// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ClusterReportOp: the demo's sink. For each frame it takes the organized
// cloud, its labels and the cluster summaries, and checks them against the
// synthetic scene's truth: the number of clusters, the adjusted Rand index of
// the labels against the true objects, each object's cluster, its centroid
// and its box, and that the summaries agree with the labels. It completes the
// run after the expected frames.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/clustering/clustering.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace clustering_app {

// Smallest agreement with the truth that passes, and largest errors.
inline constexpr double kMinRandIndex = 0.98;
inline constexpr double kMinIoU = 0.95;
inline constexpr double kMaxCentroidErrorM = 0.01;
inline constexpr double kMaxBoxErrorM = 0.03;

// What one run saw. Written by the reporter during the run, read after it.
struct ClusterReport {
  std::uint32_t frames{};
  std::uint32_t frame_gaps{};
  std::uint64_t clusters{};
  std::uint32_t wrong_counts{};
  std::uint32_t wrong_objects{};
  std::uint32_t inconsistent{};
  double min_rand_index{1.0};
  double min_iou{1.0};
  double max_centroid_error_m{};
  double max_box_error_m{};

  [[nodiscard]] bool passed() const {
    return frames != 0 && frame_gaps == 0 && wrong_counts == 0 && wrong_objects == 0 &&
           inconsistent == 0 && min_rand_index >= kMinRandIndex && clusters != 0;
  }
};

struct ClusterReportConfig {
  std::shared_ptr<ClusterReport> report;
  int width{};
  int height{};
  // The minimum cluster size the clusterer uses: smaller objects are noise.
  std::uint32_t min_cluster_size{};
  std::uint32_t expected_frames{};
  // Counts the frames done, for the source that waits for them.
  std::shared_ptr<std::atomic<std::uint32_t>> finished;
};

class ClusterReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if the report is missing.
  explicit ClusterReportOp(ClusterReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The organized cloud, [H, W, 3].
  holoscan::Input<holoscan::Tensor> points;
  // Its labels, [H, W], and the clusters, [K, kSummaryFields].
  holoscan::Input<holoscan::Tensor> labels;
  holoscan::Input<holoscan::Tensor> clusters;

 private:
  ClusterReportConfig config_;
  std::vector<float> xyz_;
  std::vector<std::int32_t> labels_;
  std::vector<holoscan::clustering::ClusterSummary> summaries_;
  std::uint64_t last_frame_id_{};
  bool finished_{};
};

}  // namespace clustering_app
