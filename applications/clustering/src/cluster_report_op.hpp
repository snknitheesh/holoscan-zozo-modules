// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ClusterReportOp: the demo's sink. For each frame it takes the organized
// point cloud, its labels and the cluster summaries, and checks them against
// the synthetic scene's truth: the number of clusters, the adjusted Rand
// index of the labels against the true objects, each object's cluster, its
// centroid and its box, and that the summaries agree with the labels. It
// completes the run after the last frame.

#pragma once

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
#include <holoscan/messages/point_cloud.hpp>

namespace clustering_app {

// Smallest agreement with the truth that passes, and largest errors.
inline constexpr double kMinRandIndex = 0.98;
inline constexpr double kMinIoU = 0.95;
inline constexpr double kMaxCentroidErrorM = 0.01;
inline constexpr double kMaxBoxErrorM = 0.03;

// What one run saw. Written by the reporter during the run, read after it.
struct ClusterReport {
  std::uint32_t frames{};
  // Frames sent before the first camera info, which give no cloud: those
  // before the first frame reported.
  std::uint32_t before_camera_info{};
  std::uint32_t frame_gaps{};
  std::uint64_t clusters{};
  std::uint32_t wrong_counts{};
  std::uint32_t wrong_objects{};
  std::uint32_t inconsistent{};
  double min_rand_index{1.0};
  double min_iou{1.0};
  double max_centroid_error_m{};
  double max_box_error_m{};

  // Every frame checked and right, the frames from the first camera info on
  // all there. A frame with no object to find is right when it finds none:
  // wrong_counts compares the clusters with the objects the truth has.
  [[nodiscard]] bool passed(std::uint32_t expected_frames,
                            std::uint32_t allowed_before_camera_info) const {
    return frames != 0 && before_camera_info <= allowed_before_camera_info &&
           frames + before_camera_info == expected_frames && frame_gaps == 0 && wrong_counts == 0 &&
           wrong_objects == 0 && inconsistent == 0 && min_rand_index >= kMinRandIndex;
  }
};

struct ClusterReportConfig {
  std::shared_ptr<ClusterReport> report;
  int width{};
  int height{};
  // The minimum cluster size the clusterer uses: smaller objects are noise.
  std::uint32_t min_cluster_size{};
  // The frames sent, with frame ids 1..expected_frames: the run completes
  // after the last.
  std::uint32_t expected_frames{};
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
  holoscan::Input<holoscan::messages::schema::PointCloudT> point_cloud;
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
