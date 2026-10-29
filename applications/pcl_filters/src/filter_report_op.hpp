// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// FilterReportOp: the demo's sink. For each frame it takes the organized cloud
// after ground removal, the ground plane and the voxels, and checks them
// against the synthetic scene's truth: the plane against the true ground, how
// much ground, objects and flying pixels are left, and the voxel count
// against a CPU count of the voxels of the same cloud, and that they have
// colors. It completes the run
// after the expected frames.

#pragma once

#include <atomic>
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
  std::uint32_t expected_frames{};
  // Counts the frames done, for the source that waits for them.
  std::shared_ptr<std::atomic<std::uint32_t>> finished;
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
  holoscan::Input<holoscan::Tensor> points;
  // The ground plane, [4].
  holoscan::Input<holoscan::Tensor> plane;
  // The voxels of `points`, [M, 3], and their colors.
  holoscan::Input<holoscan::Tensor> voxels;
  holoscan::Input<holoscan::Tensor> voxel_colors;

 private:
  FilterReportConfig config_;
  std::vector<float> xyz_;
  std::vector<std::uint8_t> rgb_;
  std::uint64_t last_frame_id_{};
  bool finished_{};
};

}  // namespace pcl_filters_app
