// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// FilterReportOp: the demo's sink. For each frame it takes the organized cloud
// after ground removal, the ground plane and the voxels, and checks them
// against the synthetic scene's truth: the plane against the true ground, how
// much ground, objects and flying pixels are left, the voxel count against a
// CPU count of the voxels of the same cloud, the voxels' layout and colors,
// and that every cloud keeps the camera's frame id. It completes the run after
// the expected frames. With a view output it draws the voxels, seen from the
// camera, for the display.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/pcl_filters/cloud_filter_op.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace pcl_filters_app {

// Largest plane error that passes.
inline constexpr double kMaxPlaneAngleDeg = 1.0;
inline constexpr double kMaxPlaneOffsetM = 0.02;
// Largest fractions of ground and flying pixels left, smallest of objects kept.
inline constexpr double kMaxGroundLeft = 0.01;
inline constexpr double kMaxOutliersLeft = 0.10;
inline constexpr double kMinObjectsKept = 0.85;

// What one run saw. Written by the reporter during the run, read after it.
struct FilterReport {
  std::uint32_t frames{};
  std::uint32_t frame_gaps{};
  std::uint64_t ground{};
  std::uint64_t ground_left{};
  std::uint64_t objects{};
  std::uint64_t objects_kept{};
  std::uint64_t outliers{};
  std::uint64_t outliers_left{};
  std::uint64_t voxels{};
  // Frames whose voxel count differs from the CPU's, or whose voxels are not
  // laid out as configured.
  std::uint32_t wrong_voxel_counts{};
  // Voxels without a color: every scene surface has one, so none is black.
  std::uint64_t black_voxels{};
  std::uint32_t wrong_planes{};
  double max_plane_angle_deg{};
  double max_plane_offset_m{};

  [[nodiscard]] double fraction(std::uint64_t part, std::uint64_t whole) const {
    return whole == 0 ? 0.0 : static_cast<double>(part) / static_cast<double>(whole);
  }
  [[nodiscard]] bool passed() const {
    return frames != 0 && frame_gaps == 0 && wrong_planes == 0 && wrong_voxel_counts == 0 &&
           black_voxels == 0 && fraction(ground_left, ground) <= kMaxGroundLeft &&
           fraction(outliers_left, outliers) <= kMaxOutliersLeft &&
           fraction(objects_kept, objects) >= kMinObjectsKept && voxels != 0;
  }
};

struct FilterReportConfig {
  std::shared_ptr<FilterReport> report;
  int width{};
  int height{};
  float voxel_size{};
  // VoxelDownsampleOp's layout: kDense or kPadded.
  holoscan::pcl_filters::OutputLayout voxel_layout{holoscan::pcl_filters::OutputLayout::kDense};
  std::uint32_t expected_frames{};
  // Counts the frames done, for the source that waits for them.
  std::shared_ptr<std::atomic<std::uint32_t>> finished;
  // Declare the `view` output.
  bool view_output{};
};

class FilterReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if the report is missing.
  explicit FilterReportOp(FilterReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The organized cloud after ground removal, [H, W, 3].
  holoscan::Input<holoscan::messages::schema::PointCloudT> ground_cloud;
  // The ground plane, [4].
  holoscan::Input<holoscan::Tensor> plane;
  // The voxels of `ground_cloud`, with their colors.
  holoscan::Input<holoscan::messages::schema::PointCloudT> voxels;
  // With `view_output`: RGB8 [H, W, 3] in host memory, each voxel a 5-pixel
  // square where the camera sees it, the nearest in front.
  holoscan::Output<holoscan::schema::ImageT> view;

 private:
  // Checks the voxels and counts them; draws them when there is a view.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> check_voxels(
      const holoscan::messages::schema::PointCloudT& cloud, holoscan::ExecutionContext& context);
  // Sends the view of the voxels in voxel_xyz_ and voxel_rgb_.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> send_view(
      std::size_t voxel_count, const holoscan::EmitOptions& emit);

  FilterReportConfig config_;
  std::vector<float> xyz_;
  std::vector<float> voxel_xyz_;
  std::vector<std::uint8_t> voxel_rgb_;
  std::uint64_t last_frame_id_{};
  bool finished_{};
};

}  // namespace pcl_filters_app
