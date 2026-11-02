// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CPU references of the mapper and of the traversability, written from their
// descriptions in height_mapper.hpp and traversability.hpp: plain loops in
// float, the same expressions in the same order, so that, compiled without
// contraction (-ffp-contract=off), the map matches the GPU's bit for bit.
// The traversability uses atan, whose float results may differ from the
// GPU's by an ulp or two, so it is compared within a tolerance.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <holoscan/height_map/grid.hpp>
#include <holoscan/height_map/height_mapper.hpp>
#include <holoscan/height_map/traversability.hpp>

namespace holoscan::height_map::test {

class ReferenceMapper {
 public:
  explicit ReferenceMapper(const HeightMapParams& params)
      : params_(params), geometry_(params.geometry) {
    const std::size_t cells = params.geometry.cells();
    cells_.assign(cells, Cell{});
  }

  void update(const std::vector<float>& xyz, const Transform& pose, std::int64_t now) {
    const HeightMapParams& p = params_;
    const MapGeometry& base = p.geometry;
    const std::size_t cells = base.cells();
    if (p.mode == MapMode::kFollow) {
      const auto cx = static_cast<std::int64_t>(std::floor((pose.m[3] - base.origin_x) /
                                                           static_cast<double>(base.resolution))) -
                      base.cols / 2;
      const auto cy = static_cast<std::int64_t>(std::floor((pose.m[7] - base.origin_y) /
                                                           static_cast<double>(base.resolution))) -
                      base.rows / 2;
      if (placed_ && (cx != corner_x_ || cy != corner_y_)) {
        std::vector<Cell> moved(cells, Cell{});
        for (int r = 0; r < base.rows; ++r) {
          for (int c = 0; c < base.cols; ++c) {
            const std::int64_t sc = c + (cx - corner_x_);
            const std::int64_t sr = r + (cy - corner_y_);
            if (sc >= 0 && sr >= 0 && sc < base.cols && sr < base.rows) {
              moved[static_cast<std::size_t>(r) * base.cols + c] =
                  cells_[static_cast<std::size_t>(sr) * base.cols + static_cast<std::size_t>(sc)];
            }
          }
        }
        cells_ = moved;
      }
      corner_x_ = cx;
      corner_y_ = cy;
      placed_ = true;
      geometry_.origin_x = base.origin_x + static_cast<double>(cx) * base.resolution;
      geometry_.origin_y = base.origin_y + static_cast<double>(cy) * base.resolution;
    }

    // Ageing.
    const bool fused = p.elevation == ElevationMode::kFused;
    const float elapsed = updated_ && now > last_ ? static_cast<float>(now - last_) * 1e-9F : 0.0F;
    const auto max_age_ns =
        static_cast<std::int64_t>(std::llround(static_cast<double>(p.max_age) * 1e9));
    for (Cell& cell : cells_) {
      if (!cell.known) {
        continue;
      }
      if (!p.accumulate || (max_age_ns > 0 && now - cell.stamp > max_age_ns)) {
        cell = Cell{};
        continue;
      }
      if (fused) {
        const float var = cell.fused_var + p.variance_growth * elapsed;
        if (p.max_variance > 0.0F && var > p.max_variance) {
          cell = Cell{};
          continue;
        }
        cell.fused_var = var;
      }
    }
    last_ = now;
    updated_ = true;

    // Binning, in input order.
    float m[12];
    for (int k = 0; k < 12; ++k) {
      m[k] = static_cast<float>(pose.m[k]);
    }
    m[3] = static_cast<float>(pose.m[3] - geometry_.origin_x);
    m[7] = static_cast<float>(pose.m[7] - geometry_.origin_y);
    std::vector<std::vector<std::pair<float, float>>> bins(cells);
    for (std::size_t i = 0; i + 2 < xyz.size(); i += 3) {
      const float x = xyz[i];
      const float y = xyz[i + 1];
      const float z = xyz[i + 2];
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }
      const float range = std::sqrt(x * x + y * y + z * z);
      const float mx = m[0] * x + m[1] * y + m[2] * z + m[3];
      const float my = m[4] * x + m[5] * y + m[6] * z + m[7];
      const float mz = m[8] * x + m[9] * y + m[10] * z + m[11];
      const float u = std::floor(mx / geometry_.resolution);
      const float v = std::floor(my / geometry_.resolution);
      if (!(range >= p.min_range && range <= p.max_range && mz >= p.min_height &&
            mz <= p.max_height && u >= 0.0F && v >= 0.0F &&
            u < static_cast<float>(geometry_.cols) && v < static_cast<float>(geometry_.rows))) {
        continue;
      }
      const float sigma = p.noise_base + p.noise_quadratic * (range * range);
      bins[static_cast<std::size_t>(v) * geometry_.cols + static_cast<std::size_t>(u)].emplace_back(
          mz, 1.0F / (sigma * sigma));
    }

    // The observations.
    for (std::size_t c = 0; c < cells; ++c) {
      const auto& points = bins[c];
      if (points.empty()) {
        continue;
      }
      Cell& cell = cells_[c];
      const float z0 = points[0].first;
      float lo = z0;
      float hi = z0;
      float sum = 0.0F;
      for (const auto& [z, w] : points) {
        lo = std::fmin(lo, z);
        hi = std::fmax(hi, z);
        sum += z - z0;
      }
      const auto count = static_cast<float>(points.size());
      const float mean_offset = sum / count;
      float squares = 0.0F;
      float weights = 0.0F;
      float weighted = 0.0F;
      for (const auto& [z, w] : points) {
        const float d = (z - z0) - mean_offset;
        squares += d * d;
        weights += w;
        weighted += w * (z - z0);
      }
      const float spread = squares / count;
      cell.stamp = now;
      cell.lo = lo;
      cell.hi = hi;
      cell.mean = z0 + mean_offset;
      cell.spread = spread;
      cell.count = count;
      const bool had = cell.known && std::isfinite(cell.fused) && std::isfinite(cell.fused_var);
      cell.known = true;
      if (!fused) {
        continue;
      }
      const float measured = z0 + weighted / weights;
      const float variance = std::fmax(1.0F / weights + spread, p.min_variance);
      if (!had) {
        cell.fused = measured;
        cell.fused_var = variance;
        continue;
      }
      const float innovation = measured - cell.fused;
      const float total = cell.fused_var + variance;
      if (p.mahalanobis_gate > 0.0F &&
          innovation * innovation > (p.mahalanobis_gate * p.mahalanobis_gate) * total) {
        if (innovation > 0.0F) {
          cell.fused = measured;
          cell.fused_var = variance;
        } else {
          cell.fused_var = cell.fused_var + p.rejection_variance;
        }
        continue;
      }
      const float gain = cell.fused_var / total;
      cell.fused = cell.fused + gain * innovation;
      cell.fused_var = (cell.fused_var * variance) / total;
    }
  }

  // The layers, as HeightMapper::write_layers() writes them.
  struct Layers {
    std::vector<float> elevation, variance, count, min_elevation, max_elevation, time;
  };
  [[nodiscard]] Layers layers() const {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Layers out;
    for (const Cell& cell : cells_) {
      if (!cell.known) {
        for (auto* layer : {&out.elevation, &out.variance, &out.count, &out.min_elevation,
                            &out.max_elevation, &out.time}) {
          layer->push_back(nan);
        }
        continue;
      }
      const ElevationMode mode = params_.elevation;
      out.elevation.push_back(mode == ElevationMode::kMax    ? cell.hi
                              : mode == ElevationMode::kMean ? cell.mean
                                                             : cell.fused);
      out.variance.push_back(mode == ElevationMode::kFused ? cell.fused_var : cell.spread);
      out.count.push_back(cell.count);
      out.min_elevation.push_back(cell.lo);
      out.max_elevation.push_back(cell.hi);
      out.time.push_back(static_cast<float>(last_ - cell.stamp) * 1e-9F);
    }
    return out;
  }

  [[nodiscard]] const MapGeometry& geometry() const { return geometry_; }

 private:
  struct Cell {
    bool known{};
    std::int64_t stamp{};
    float fused{std::numeric_limits<float>::quiet_NaN()};
    float fused_var{std::numeric_limits<float>::quiet_NaN()};
    float lo{}, hi{}, mean{}, spread{}, count{};
  };

  HeightMapParams params_;
  MapGeometry geometry_;
  std::vector<Cell> cells_;
  std::int64_t corner_x_{};
  std::int64_t corner_y_{};
  bool placed_{};
  std::int64_t last_{};
  bool updated_{};
};

// The traversability of one cell, as traversability.hpp describes it.
struct ReferenceTraversability {
  std::vector<float> cost, slope, step, roughness;
  std::vector<std::uint8_t> unknown;
};

inline ReferenceTraversability reference_traversability(const std::vector<float>& elevation,
                                                        int rows, int cols, float resolution,
                                                        const TraversabilityParams& p) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  ReferenceTraversability out;
  const int side = 2 * p.window_radius + 1;
  const auto min_valid =
      static_cast<int>(std::ceil(p.min_valid_fraction * static_cast<float>(side * side)));
  const auto at = [&](int r, int c) { return elevation[static_cast<std::size_t>(r) * cols + c]; };
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const float center = at(r, c);
      std::vector<float> xs;
      std::vector<float> ys;
      std::vector<float> zs;
      if (std::isfinite(center)) {
        for (int v = std::max(r - p.window_radius, 0); v <= std::min(r + p.window_radius, rows - 1);
             ++v) {
          for (int u = std::max(c - p.window_radius, 0);
               u <= std::min(c + p.window_radius, cols - 1); ++u) {
            if (std::isfinite(at(v, u))) {
              xs.push_back(static_cast<float>(u - c) * resolution);
              ys.push_back(static_cast<float>(v - r) * resolution);
              zs.push_back(at(v, u) - center);
            }
          }
        }
      }
      const int n = static_cast<int>(xs.size());
      bool known = std::isfinite(center) && n >= min_valid && n >= 3;
      float a = 0.0F;
      float b = 0.0F;
      float mx = 0.0F;
      float my = 0.0F;
      float mz = 0.0F;
      if (known) {
        float sx = 0.0F;
        float sy = 0.0F;
        float sz = 0.0F;
        for (int k = 0; k < n; ++k) {
          sx += xs[k];
          sy += ys[k];
          sz += zs[k];
        }
        mx = sx / static_cast<float>(n);
        my = sy / static_cast<float>(n);
        mz = sz / static_cast<float>(n);
        float cxx = 0.0F;
        float cxy = 0.0F;
        float cyy = 0.0F;
        float cxz = 0.0F;
        float cyz = 0.0F;
        for (int k = 0; k < n; ++k) {
          const float x = xs[k] - mx;
          const float y = ys[k] - my;
          const float z = zs[k] - mz;
          cxx += x * x;
          cxy += x * y;
          cyy += y * y;
          cxz += x * z;
          cyz += y * z;
        }
        const float det = cxx * cyy - cxy * cxy;
        const float scale = cxx + cyy;
        known = det > 1e-6F * (scale * scale);
        if (known) {
          a = (cxz * cyy - cyz * cxy) / det;
          b = (cyz * cxx - cxz * cxy) / det;
        }
      }
      if (!known) {
        out.cost.push_back(nan);
        out.slope.push_back(nan);
        out.step.push_back(nan);
        out.roughness.push_back(nan);
        out.unknown.push_back(1U);
        continue;
      }
      float squares = 0.0F;
      for (int k = 0; k < n; ++k) {
        const float residual = (zs[k] - mz) - (a * (xs[k] - mx) + b * (ys[k] - my));
        squares += residual * residual;
      }
      const float roughness = std::sqrt(squares / static_cast<float>(n));
      const float slope = std::atan(std::sqrt(a * a + b * b)) * 57.29577951308232F;
      float lowest = center;
      float highest = center;
      for (int v = std::max(r - p.step_radius, 0); v <= std::min(r + p.step_radius, rows - 1);
           ++v) {
        for (int u = std::max(c - p.step_radius, 0); u <= std::min(c + p.step_radius, cols - 1);
             ++u) {
          if (std::isfinite(at(v, u))) {
            lowest = std::fmin(lowest, at(v, u));
            highest = std::fmax(highest, at(v, u));
          }
        }
      }
      const float step = highest - lowest;
      const float st = slope / p.max_slope_deg;
      const float tt = step / p.max_step;
      const float rt = roughness / p.max_roughness;
      float cost = 1.0F;
      if (st < 1.0F && tt < 1.0F && rt < 1.0F) {
        cost = (p.slope_weight * st + p.step_weight * tt + p.roughness_weight * rt) /
               (p.slope_weight + p.step_weight + p.roughness_weight);
        cost = std::fmin(std::fmax(cost, 0.0F), 1.0F);
      }
      out.cost.push_back(cost);
      out.slope.push_back(slope);
      out.step.push_back(step);
      out.roughness.push_back(roughness);
      out.unknown.push_back(0U);
    }
  }
  return out;
}

}  // namespace holoscan::height_map::test
