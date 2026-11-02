// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ReportOp: the demo's sink. For each map it takes the elevation, the map
// info and the traversability and checks them against the course's truth:
// the elevation of every known cell away from a height jump (median and 95th
// percentile of the error), the slope on the ramp, the largest step height
// near each of the steps' rises in each row, the unknown mask and the other
// layers, the box's near rim lethal, the free ground cheap, and the window where
// the camera is. It completes the run after the expected frames.

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
#include <holoscan/height_map/grid.hpp>
#include <holoscan/height_map/synthetic_terrain.hpp>

namespace height_map_app {

// What passes, every frame. The elevation's limits are for the mean and fused
// layers; the highest point of a cell lies above the surface by its noise, so
// main() relaxes them for kMax to 2 and 4 cm. The lethal rim's is for the
// fused layer: the others hold only the latest observation, in which a cell
// at the edge of the view may have seen only part of the box's face, so
// main() relaxes it for them to 0.5.
inline constexpr double kMaxMedianErrorM = 0.004;
inline constexpr double kMaxP95ErrorM = 0.02;
inline constexpr double kMaxSlopeErrorDeg = 1.0;
inline constexpr double kMaxStepErrorM = 0.02;
inline constexpr double kMaxFlatCostP95 = 0.3;
inline constexpr double kMinLethalRim = 0.9;
inline constexpr double kMinKnownShare = 0.10;

// What one run saw. Written by the reporter during the run, read after it.
struct Report {
  double max_median_error_m{kMaxMedianErrorM};
  double max_p95_error_m{kMaxP95ErrorM};
  double min_lethal_rim{kMinLethalRim};
  std::uint32_t frames{};
  std::uint32_t frame_gaps{};
  double worst_median_error_m{};
  double worst_p95_error_m{};
  double worst_slope_error_deg{};
  double worst_step_error_m{};
  double worst_flat_cost_p95{};
  double worst_lethal_rim{1.0};
  double least_known_share{1.0};
  std::uint32_t window_wrong{};
  // Cells whose unknown mask disagrees with their cost, or whose slope, step
  // height or roughness is missing or negative where the cost is known.
  std::uint64_t mask_wrong{};

  [[nodiscard]] bool passed() const {
    return frames != 0 && frame_gaps == 0 && worst_median_error_m <= max_median_error_m &&
           worst_p95_error_m <= max_p95_error_m && worst_slope_error_deg <= kMaxSlopeErrorDeg &&
           worst_step_error_m <= kMaxStepErrorM && worst_flat_cost_p95 <= kMaxFlatCostP95 &&
           worst_lethal_rim >= min_lethal_rim && least_known_share >= kMinKnownShare &&
           window_wrong == 0 && mask_wrong == 0;
  }
};

struct ReportConfig {
  std::shared_ptr<Report> report;
  int camera_width{};
  int camera_height{};
  // The map's settings, as HeightMapOp has them.
  holoscan::height_map::MapGeometry geometry;
  std::uint32_t expected_frames{};
  // Counts the frames done, for the camera that waits for them.
  std::shared_ptr<std::atomic<std::uint32_t>> finished;
};

class ReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if the report is missing.
  explicit ReportOp(ReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> elevation;    // float32 [rows, cols]
  holoscan::Input<holoscan::Tensor> map_info;     // float64 [5]
  holoscan::Input<holoscan::Tensor> cost;         // float32 [rows, cols]
  holoscan::Input<holoscan::Tensor> unknown;      // uint8 [rows, cols]
  holoscan::Input<holoscan::Tensor> slope;        // float32 [rows, cols]
  holoscan::Input<holoscan::Tensor> step_height;  // float32 [rows, cols]
  holoscan::Input<holoscan::Tensor> roughness;    // float32 [rows, cols]

 private:
  ReportConfig config_;
  holoscan::height_map::Terrain terrain_;
  std::vector<float> elevation_;
  std::vector<float> cost_;
  std::vector<float> slope_;
  std::vector<float> step_;
  std::vector<float> roughness_;
  std::vector<std::uint8_t> unknown_;
  std::uint64_t last_frame_id_{};
  bool finished_{};
};

}  // namespace height_map_app
