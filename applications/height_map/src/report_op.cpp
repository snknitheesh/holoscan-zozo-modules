// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "report_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "terrain_camera_op.hpp"

namespace height_map_app {
namespace {

using std::chrono_literals::operator""ns;
namespace hm = holoscan::height_map;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "ReportOp: " + message};
}

double percentile(std::vector<double> values, double q) {
  if (values.empty()) {
    return std::nan("");
  }
  std::sort(values.begin(), values.end());
  return values[std::min(values.size() - 1, static_cast<std::size_t>(q * values.size()))];
}

// The steps' rise within `reach` of x, x0 + k * length for k = 0 .. count - 1,
// on the steps away from their sides and their last edge (where the height
// drops back by all of them): k, or -1 for none.
int rise_near(const hm::TerrainFeature& f, double x, double y, double margin, double reach) {
  if (y < f.y0 + margin || y > f.y1 - margin || x > f.x1 - margin) {
    return -1;
  }
  int k = 0;
  for (double edge = f.x0; edge < f.x1 - 1e-6; edge += f.step_length, ++k) {
    if (std::fabs(x - edge) < reach) {
      return k;
    }
  }
  return -1;
}

}  // namespace

ReportOp::ReportOp(ReportConfig config)
    : config_(std::move(config)), terrain_(hm::course_terrain()) {
  if (!config_.report) {
    throw std::invalid_argument("ReportOp needs a report");
  }
}

void ReportOp::start() {
  const std::size_t cells = config_.geometry.cells();
  if (host_ == nullptr &&
      cudaMallocHost(&host_, cells * (5U * sizeof(float) + 1U)) != cudaSuccess) {
    host_ = nullptr;
  }
}

void ReportOp::stop() {
  if (host_ != nullptr) {
    cudaFreeHost(host_);
    host_ = nullptr;
  }
}

void ReportOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: one map in compute() while the next one arrives.
  spec.input(elevation, "elevation").queue_depth(2U);
  spec.input(map_info, "map_info").queue_depth(2U);
  spec.input(cost, "cost").queue_depth(2U);
  spec.input(unknown, "unknown").queue_depth(2U);
  spec.input(slope, "slope").queue_depth(2U);
  spec.input(step_height, "step_height").queue_depth(2U);
  spec.input(roughness, "roughness").queue_depth(2U);
}

holoscan::Contract ReportOp::contract() const {
  holoscan::Contract result;
  result.trigger(
      holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                              elevation, map_info, cost, unknown, slope, step_height, roughness}},
                          .window = 0ns});
  for (const auto* port :
       {&elevation, &map_info, &cost, &unknown, &slope, &step_height, &roughness}) {
    result.input(*port, holoscan::Exact{});
  }
  return result;
}

holoscan::expected<void, holoscan::Error> ReportOp::compute(holoscan::ExecutionContext& context) {
  auto elevation_sample = elevation.receive();
  auto info_sample = map_info.receive();
  auto cost_sample = cost.receive();
  auto unknown_sample = unknown.receive();
  auto slope_sample = slope.receive();
  auto step_sample = step_height.receive();
  auto roughness_sample = roughness.receive();
  for (const auto* failed : {&elevation_sample, &info_sample, &cost_sample, &unknown_sample,
                             &slope_sample, &step_sample, &roughness_sample}) {
    if (!*failed) {
      return holoscan::make_unexpected(failed->error());
    }
  }
  const hm::MapGeometry& expected = config_.geometry;
  const int rows = expected.rows;
  const int cols = expected.cols;
  const std::span<const std::int64_t> shape = elevation_sample->data.shape_span();
  if (shape.size() != 2 || shape[0] != rows || shape[1] != cols) {
    return holoscan::make_unexpected(invalid("expected a [rows, cols] elevation"));
  }
  const std::size_t cells = static_cast<std::size_t>(rows) * cols;
  if (host_ == nullptr) {
    return holoscan::make_unexpected(invalid("no pinned host memory for the layers"));
  }
  float* const heights = static_cast<float*>(host_);
  float* const costs = heights + cells;
  float* const slopes = costs + cells;
  float* const step_heights = slopes + cells;
  float* const roughnesses = step_heights + cells;
  auto* const unknowns = reinterpret_cast<std::uint8_t*>(roughnesses + cells);
  // The layers' copies go on the operator's stream, which waits for their
  // producers; the host waits once, for all of them. (Without a stream each
  // copy completes before it returns.) map_info is in host memory.
  cudaStream_t stream = context.cuda_stream().get();
  for (const auto& [tensor, destination, bytes] :
       {std::tuple{&elevation_sample->data, static_cast<void*>(heights), 4 * cells},
        std::tuple{&cost_sample->data, static_cast<void*>(costs), 4 * cells},
        std::tuple{&slope_sample->data, static_cast<void*>(slopes), 4 * cells},
        std::tuple{&step_sample->data, static_cast<void*>(step_heights), 4 * cells},
        std::tuple{&roughness_sample->data, static_cast<void*>(roughnesses), 4 * cells},
        std::tuple{&unknown_sample->data, static_cast<void*>(unknowns), cells}}) {
    if (auto copied = tensor->copy_to_host_async(destination, bytes, stream); !copied) {
      return copied;
    }
  }
  std::array<double, hm::kMapInfoSize> info{};
  if (auto copied = info_sample->data.copy_to_host(info.data(), sizeof(info)); !copied) {
    return copied;
  }
  if (stream != nullptr) {
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return holoscan::make_unexpected(
          invalid(std::string("cudaStreamSynchronize: ") + cudaGetErrorString(status)));
    }
  }

  Report& report = *config_.report;
  const std::uint64_t frame_id = elevation_sample->metadata.frame_id;
  if (frame_id != last_frame_id_ + 1U) {
    ++report.frame_gaps;
  }
  last_frame_id_ = frame_id;

  // The window: centered on the camera, on the lattice of the configured origin.
  const hm::TerrainCamera camera = camera_for_frame(config_.camera_width, config_.camera_height,
                                                    static_cast<std::uint32_t>(frame_id - 1U));
  const double res = expected.resolution;
  const double origin_x =
      expected.origin_x + static_cast<double>(static_cast<std::int64_t>(std::floor(
                                                  (camera.pose.m[3] - expected.origin_x) / res)) -
                                              cols / 2) *
                              res;
  const double origin_y =
      expected.origin_y + static_cast<double>(static_cast<std::int64_t>(std::floor(
                                                  (camera.pose.m[7] - expected.origin_y) / res)) -
                                              rows / 2) *
                              res;
  if (std::fabs(info[hm::kInfoOriginX] - origin_x) > 1e-6 ||
      std::fabs(info[hm::kInfoOriginY] - origin_y) > 1e-6 || info[hm::kInfoRows] != rows ||
      info[hm::kInfoCols] != cols) {
    ++report.window_wrong;
  }

  std::vector<double> errors;
  std::vector<double> slope_errors;
  // The largest step height in each map row near each rise.
  std::map<std::pair<int, int>, float> rise_steps;
  std::vector<double> flat_costs;
  std::size_t rim = 0;
  std::size_t lethal = 0;
  std::size_t known = 0;
  const hm::TerrainFeature* stairs = nullptr;
  const hm::TerrainFeature* box = nullptr;
  for (int k = 0; k < terrain_.feature_count; ++k) {
    if (terrain_.features[k].kind == hm::FeatureKind::kSteps) {
      stairs = &terrain_.features[k];
    } else if (terrain_.features[k].kind == hm::FeatureKind::kBox) {
      box = &terrain_.features[k];
    }
  }
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const std::size_t i = static_cast<std::size_t>(r) * cols + c;
      const bool layers_agree =
          (unknowns[i] != 0U) == std::isnan(costs[i]) &&
          (unknowns[i] != 0U || (slopes[i] >= 0.0F && step_heights[i] >= 0.0F &&
                                 roughnesses[i] >= 0.0F && costs[i] >= 0.0F && costs[i] <= 1.0F));
      report.mask_wrong += layers_agree ? 0U : 1U;
      const float e = heights[i];
      if (!std::isfinite(e)) {
        continue;
      }
      ++known;
      const double x = info[hm::kInfoOriginX] + (c + 0.5) * res;
      const double y = info[hm::kInfoOriginY] + (r + 0.5) * res;
      const double edge = hm::terrain_edge_distance(terrain_, x, y);
      if (edge > 1.5 * res) {
        errors.push_back(std::fabs(
            e - hm::terrain_height(terrain_, static_cast<float>(x), static_cast<float>(y))));
      }
      const bool cost_known = std::isfinite(costs[i]);
      const double truth_slope = hm::terrain_slope_deg(terrain_, x, y);
      if (cost_known && edge > 3.5 * res) {
        if (truth_slope > 0.0) {
          slope_errors.push_back(std::fabs(slopes[i] - truth_slope));
        } else {
          flat_costs.push_back(costs[i]);
        }
      }
      // A rise inside a cell is seen in full by the cells beside that cell.
      const int rise = stairs != nullptr ? rise_near(*stairs, x, y, 3.0 * res, 2.0 * res) : -1;
      if (cost_known && rise >= 0) {
        float& most = rise_steps[{r, rise}];
        most = std::max(most, step_heights[i]);
      }
      // The box's near rim, which faces the camera: both sides of it must be
      // lethal.
      if (cost_known && box != nullptr && std::fabs(x - box->x0) < res && y >= box->y0 + res &&
          y < box->y1 - res) {
        ++rim;
        lethal += costs[i] == 1.0F ? 1U : 0U;
      }
    }
  }
  const double median = percentile(errors, 0.5);
  const double p95 = percentile(errors, 0.95);
  report.worst_median_error_m =
      std::max(report.worst_median_error_m, errors.empty() ? 1.0 : median);
  report.worst_p95_error_m = std::max(report.worst_p95_error_m, errors.empty() ? 1.0 : p95);
  report.worst_slope_error_deg = std::max(
      report.worst_slope_error_deg, slope_errors.empty() ? 90.0 : percentile(slope_errors, 0.5));
  std::vector<double> step_errors;
  for (const auto& [where, step] : rise_steps) {
    step_errors.push_back(std::fabs(step - stairs->value));
  }
  report.worst_step_error_m =
      std::max(report.worst_step_error_m, step_errors.empty() ? 1.0 : percentile(step_errors, 0.5));
  report.worst_flat_cost_p95 =
      std::max(report.worst_flat_cost_p95, flat_costs.empty() ? 1.0 : percentile(flat_costs, 0.95));
  report.worst_lethal_rim =
      std::min(report.worst_lethal_rim,
               rim == 0 ? 0.0 : static_cast<double>(lethal) / static_cast<double>(rim));
  report.least_known_share =
      std::min(report.least_known_share, static_cast<double>(known) / static_cast<double>(cells));

  ++report.frames;
  if (config_.finished) {
    config_.finished->fetch_add(1U, std::memory_order_release);
  }
  if (!finished_ && report.frames >= config_.expected_frames) {
    finished_ = true;
    context.request_completion();
  }
  return {};
}

}  // namespace height_map_app
