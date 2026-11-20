// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The settings' rules, and whether a transform is rigid: without the SDK or
// CUDA, so that tests/host/ checks them on any machine.

#include <cmath>

#include <holoscan/height_map/grid.hpp>
#include <holoscan/height_map/params.hpp>

namespace holoscan::height_map {
namespace {

bool positive(float value) { return std::isfinite(value) && value > 0.0F; }
bool not_negative(float value) { return std::isfinite(value) && value >= 0.0F; }

}  // namespace

bool valid_params(const HeightMapParams& p, const char** what) {
  const auto fail = [what](const char* message) {
    if (what != nullptr) {
      *what = message;
    }
    return false;
  };
  const MapGeometry& g = p.geometry;
  if (!(std::isfinite(g.resolution) && g.resolution > 0.0F)) {
    return fail("resolution must be positive");
  }
  if (g.rows < 1 || g.cols < 1 || g.rows > kMaxGridSide || g.cols > kMaxGridSide) {
    return fail("rows and cols must be 1..8192");
  }
  if (!std::isfinite(g.origin_x) || !std::isfinite(g.origin_y)) {
    return fail("origin must be finite");
  }
  if (p.mode != MapMode::kFixed && p.mode != MapMode::kFollow) {
    return fail("mode must be kFixed or kFollow");
  }
  if (p.elevation != ElevationMode::kMax && p.elevation != ElevationMode::kMean &&
      p.elevation != ElevationMode::kFused) {
    return fail("elevation must be kMax, kMean or kFused");
  }
  if (std::isnan(p.min_height) || std::isnan(p.max_height) || !(p.min_height <= p.max_height)) {
    return fail("min_height must not exceed max_height");
  }
  if (!(p.min_range >= 0.0F) || std::isnan(p.max_range) || !(p.min_range <= p.max_range)) {
    return fail("min_range must be 0..max_range");
  }
  // A point at the sensor has range 0: only noise_base keeps its standard
  // deviation, and so its weight, finite.
  if (!(std::isfinite(p.noise_base) && p.noise_base > 0.0F) ||
      !(std::isfinite(p.noise_quadratic) && p.noise_quadratic >= 0.0F)) {
    return fail("noise_base must be positive and noise_quadratic not negative, both finite");
  }
  if (!(std::isfinite(p.min_variance) && p.min_variance > 0.0F)) {
    return fail("min_variance must be positive");
  }
  if (!(std::isfinite(p.mahalanobis_gate) && p.mahalanobis_gate >= 0.0F)) {
    return fail("mahalanobis_gate must not be negative");
  }
  if (!(std::isfinite(p.rejection_variance) && p.rejection_variance >= 0.0F)) {
    return fail("rejection_variance must not be negative");
  }
  if (!(std::isfinite(p.max_age) && p.max_age >= 0.0F)) {
    return fail("max_age must not be negative");
  }
  if (!(std::isfinite(p.variance_growth) && p.variance_growth >= 0.0F)) {
    return fail("variance_growth must not be negative");
  }
  if (!(std::isfinite(p.max_variance) && p.max_variance >= 0.0F)) {
    return fail("max_variance must not be negative");
  }
  return true;
}

bool is_rigid(const Transform& t, double tolerance) {
  for (const double value : t.m) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  const double* m = t.m;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const double dot =
          m[4 * a] * m[4 * b] + m[4 * a + 1] * m[4 * b + 1] + m[4 * a + 2] * m[4 * b + 2];
      if (std::fabs(dot - (a == b ? 1.0 : 0.0)) > tolerance) {
        return false;
      }
    }
  }
  const double det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
                     m[2] * (m[4] * m[9] - m[5] * m[8]);
  return det > 0.0;
}

bool valid_params(const TraversabilityParams& p, const char** what) {
  const auto fail = [what](const char* message) {
    if (what != nullptr) {
      *what = message;
    }
    return false;
  };
  if (p.window_radius < 1 || p.window_radius > 10) {
    return fail("window_radius must be 1..10");
  }
  if (!(p.min_valid_fraction >= 0.0F && p.min_valid_fraction <= 1.0F)) {
    return fail("min_valid_fraction must be 0..1");
  }
  if (p.step_radius < 1 || p.step_radius > 10) {
    return fail("step_radius must be 1..10");
  }
  if (!positive(p.max_slope_deg) || !positive(p.max_step) || !positive(p.max_roughness)) {
    return fail("max_slope_deg, max_step and max_roughness must be positive");
  }
  if (!not_negative(p.slope_weight) || !not_negative(p.step_weight) ||
      !not_negative(p.roughness_weight) ||
      !(p.slope_weight + p.step_weight + p.roughness_weight > 0.0F)) {
    return fail("the weights must not be negative, nor all 0");
  }
  return true;
}

}  // namespace holoscan::height_map
