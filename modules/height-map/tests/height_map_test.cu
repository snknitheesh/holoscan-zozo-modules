// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// HeightMapper on clouds of synthetic terrain made at test time: the
// elevation of flat ground, a ramp, steps, a box and a pit, with and without
// noise, per cell against the analytic heights; the counts and the unknown
// cells; the moving window's shift; temporal fusion reducing the variance;
// the clearing and decay policy; the CPU reference on a small grid, bit for
// bit; determinism; a depth camera's view of the demo's course; and empty,
// invalid and out-of-range clouds.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/height_map/height_mapper.hpp>
#include <holoscan/height_map/synthetic_terrain.hpp>

#include "reference.hpp"
#include "terrain_support.hpp"
#include "test_support.hpp"

namespace holoscan::height_map {
namespace {

using test::cell_x;
using test::cell_y;
using test::HostLayers;
using test::percentile;
using test::read_layers;
using test::same_floats;
using test::terrain_cloud;
using test::update;

constexpr std::int64_t kSecond = 1'000'000'000;

// A 4 m square grid of 5 cm cells around the origin.
HeightMapParams square_grid(ElevationMode mode) {
  HeightMapParams p;
  p.geometry =
      MapGeometry{.resolution = 0.05F, .rows = 80, .cols = 80, .origin_x = -2.0, .origin_y = -2.0};
  p.elevation = mode;
  p.noise_base = 0.005F;
  p.noise_quadratic = 0.0F;
  return p;
}

struct NamedTerrain {
  std::string name;
  Terrain terrain;
};

std::vector<NamedTerrain> analytic_terrains() {
  std::vector<NamedTerrain> out;
  out.push_back({"flat", Terrain{.base = 0.12F}});
  Terrain ramped;
  add_feature(ramped, ramp(-1.0F, 1.0F, -1.0F, 1.0F, 20.0F));
  out.push_back({"ramp", ramped});
  Terrain stepped;
  add_feature(stepped, steps(-1.0F, 1.0F, -1.0F, 1.0F, 0.1F, 0.4F));
  out.push_back({"steps", stepped});
  Terrain boxed;
  add_feature(boxed, box(-0.5F, 0.5F, -0.3F, 0.3F, 0.35F));
  out.push_back({"box", boxed});
  Terrain pitted;
  add_feature(pitted, pit(-0.6F, 0.4F, -0.5F, 0.5F, 0.25F));
  out.push_back({"pit", pitted});
  return out;
}

// The elevation of cells whose center is more than `margin` from a height
// jump, against the terrain's height at the center. Returns the absolute
// errors.
std::vector<double> errors(const Terrain& terrain, const MapGeometry& g,
                           const std::vector<float>& elevation, double margin, int* unknown) {
  std::vector<double> out;
  for (int r = 0; r < g.rows; ++r) {
    for (int c = 0; c < g.cols; ++c) {
      const double x = cell_x(g, c);
      const double y = cell_y(g, r);
      if (std::fabs(x) > 1.45 || std::fabs(y) > 1.45 ||
          terrain_edge_distance(terrain, x, y) < margin) {
        continue;
      }
      const float e = elevation[static_cast<std::size_t>(r) * g.cols + c];
      if (!std::isfinite(e)) {
        ++*unknown;
        continue;
      }
      out.push_back(
          std::fabs(e - terrain_height(terrain, static_cast<float>(x), static_cast<float>(y))));
    }
  }
  return out;
}

void check_analytic_terrains() {
  for (const NamedTerrain& t : analytic_terrains()) {
    // With 5 mm noise, 25 points per cell: mean and fused within a few mm.
    const std::vector<float> noisy =
        terrain_cloud(t.terrain, -1.5F, 1.5F, -1.5F, 1.5F, 0.01F, 0.005F, 7U);
    for (const ElevationMode mode : {ElevationMode::kMean, ElevationMode::kFused}) {
      HeightMapper mapper(square_grid(mode));
      update(mapper, noisy, Transform{}, kSecond);
      const HostLayers layers = read_layers(mapper);
      int unknown = 0;
      const std::vector<double> e =
          errors(t.terrain, mapper.geometry(), layers.elevation, 0.04, &unknown);
      const std::string at = t.name + (mode == ElevationMode::kMean ? " mean" : " fused") + ": ";
      std::printf("  %-14s median %.2f mm, p95 %.2f mm, max %.2f mm over %zu cells\n", at.c_str(),
                  1e3 * percentile(e, 0.5), 1e3 * percentile(e, 0.95), 1e3 * percentile(e, 1.0),
                  e.size());
      VERIFY_MSG(e.size() > 2500, at + "cells checked");
      VERIFY_MSG(unknown == 0, at + "every sampled cell is known");
      VERIFY_MSG(percentile(e, 0.5) < 0.0015,
                 at + "median error " + std::to_string(percentile(e, 0.5)));
      VERIFY_MSG(percentile(e, 0.95) < 0.0035,
                 at + "p95 error " + std::to_string(percentile(e, 0.95)));
      VERIFY_MSG(percentile(e, 1.0) < 0.006,
                 at + "max error " + std::to_string(percentile(e, 1.0)));
    }
    // Without noise: the highest point, the lowest and the count, exactly.
    const std::vector<float> clean = terrain_cloud(t.terrain, -1.5F, 1.5F, -1.5F, 1.5F, 0.01F);
    HeightMapper mapper(square_grid(ElevationMode::kMax));
    update(mapper, clean, Transform{}, kSecond);
    const HostLayers layers = read_layers(mapper);
    const MapGeometry g = mapper.geometry();
    int wrong_max = 0;
    int wrong_count = 0;
    int wrong_unknown = 0;
    for (int r = 0; r < g.rows; ++r) {
      for (int c = 0; c < g.cols; ++c) {
        const std::size_t i = static_cast<std::size_t>(r) * g.cols + c;
        const double x = cell_x(g, c);
        const double y = cell_y(g, r);
        const bool sampled = std::fabs(x) < 1.5 && std::fabs(y) < 1.5;
        if (!sampled) {
          wrong_unknown += std::isnan(layers.elevation[i]) && std::isnan(layers.count[i]) &&
                                   std::isnan(layers.variance[i]) &&
                                   std::isnan(layers.min_elevation[i]) && std::isnan(layers.time[i])
                               ? 0
                               : 1;
          continue;
        }
        wrong_count += layers.count[i] == 25.0F ? 0 : 1;
        if (terrain_edge_distance(t.terrain, x, y) < 0.04) {
          continue;
        }
        // The ramp's highest point is its last lattice point in x: 2 cm
        // past the center.
        const auto fx = static_cast<float>(x);
        const auto fy = static_cast<float>(y);
        const float top = terrain_height(t.terrain, fx + 0.02F, fy);
        const float bottom = terrain_height(t.terrain, fx - 0.02F, fy);
        wrong_max += std::fabs(layers.elevation[i] - top) < 1e-5F &&
                             std::fabs(layers.max_elevation[i] - top) < 1e-5F &&
                             std::fabs(layers.min_elevation[i] - bottom) < 1e-5F &&
                             layers.time[i] == 0.0F
                         ? 0
                         : 1;
      }
    }
    VERIFY_MSG(wrong_max == 0,
               t.name + ": highest and lowest points, " + std::to_string(wrong_max) + " wrong");
    VERIFY_MSG(wrong_count == 0, t.name + ": 25 points in every sampled cell, " +
                                     std::to_string(wrong_count) + " wrong");
    VERIFY_MSG(wrong_unknown == 0,
               t.name + ": every other cell unknown, " + std::to_string(wrong_unknown) + " wrong");
  }
}

// A point above max_height or outside the range limits is dropped.
void check_limits() {
  HeightMapParams p = square_grid(ElevationMode::kMax);
  p.max_height = 0.2F;
  p.min_height = -0.1F;
  Terrain boxed;
  add_feature(boxed, box(-0.5F, 0.5F, -0.5F, 0.5F, 0.35F));
  HeightMapper mapper(p);
  update(mapper, terrain_cloud(boxed, -1.0F, 1.0F, -1.0F, 1.0F, 0.01F), Transform{}, kSecond);
  HostLayers layers = read_layers(mapper);
  const MapGeometry g = mapper.geometry();
  int wrong = 0;
  for (int r = 0; r < g.rows; ++r) {
    for (int c = 0; c < g.cols; ++c) {
      const double x = cell_x(g, c);
      const double y = cell_y(g, r);
      const float e = layers.elevation[static_cast<std::size_t>(r) * g.cols + c];
      const bool on_box = std::fabs(x) < 0.5 && std::fabs(y) < 0.5;
      const bool ground = std::fabs(x) < 1.0 && std::fabs(y) < 1.0 && !on_box;
      wrong += (ground ? e == 0.0F : std::isnan(e)) ? 0 : 1;
    }
  }
  VERIFY_MSG(wrong == 0, "the box above max_height is dropped, " + std::to_string(wrong));

  // Range: from the sensor 1 m above the origin, keep 0.5..1.2 m.
  p = square_grid(ElevationMode::kMax);
  p.min_range = 0.5F;
  p.max_range = 1.2F;
  HeightMapper ranged(p);
  const Transform pose = test::translation(0.0, 0.0, 1.0);
  update(ranged,
         test::transformed(terrain_cloud(Terrain{}, -1.5F, 1.5F, -1.5F, 1.5F, 0.01F),
                           test::inverse(pose)),
         pose, kSecond);
  layers = read_layers(ranged);
  wrong = 0;
  for (int r = 0; r < g.rows; ++r) {
    for (int c = 0; c < g.cols; ++c) {
      const double d = std::hypot(cell_x(g, c), cell_y(g, r));
      const float e = layers.elevation[static_cast<std::size_t>(r) * g.cols + c];
      // Cells well inside the 0.66 m ring are known, well outside unknown.
      if (d < 0.6 && std::isnan(e)) {
        ++wrong;
      }
      if (d > 0.72 && std::isfinite(e)) {
        ++wrong;
      }
    }
  }
  VERIFY_MSG(wrong == 0, "points beyond max_range are dropped, " + std::to_string(wrong));
}

// The window follows the sensor: what stays inside keeps its values, what
// enters is unknown.
void check_moving_window() {
  HeightMapParams p;
  p.geometry =
      MapGeometry{.resolution = 0.1F, .rows = 40, .cols = 50, .origin_x = 0.0, .origin_y = 0.0};
  p.mode = MapMode::kFollow;
  p.elevation = ElevationMode::kMean;
  Terrain ramped;
  add_feature(ramped, ramp(-5.0F, 5.0F, -5.0F, 5.0F, 10.0F));
  HeightMapper mapper(p);
  update(mapper, terrain_cloud(ramped, -3.0F, 3.0F, -3.0F, 3.0F, 0.02F), Transform{}, kSecond);
  const HostLayers before = read_layers(mapper);
  const MapGeometry g0 = mapper.geometry();
  VERIFY_MSG(std::fabs(g0.origin_x + 2.5) < 1e-6 && std::fabs(g0.origin_y + 2.0) < 1e-6,
             "the window is centered on the sensor");
  VERIFY(std::none_of(before.elevation.begin(), before.elevation.end(),
                      [](float v) { return std::isnan(v); }));
  // Move by +3.5 cells in x and -2.2 in y, with nothing seen.
  update(mapper, {}, test::translation(0.35, -0.22, 0.0), 2 * kSecond);
  const HostLayers after = read_layers(mapper);
  const MapGeometry g1 = mapper.geometry();
  VERIFY_MSG(std::fabs(g1.origin_x - (g0.origin_x + 0.3)) < 1e-6 &&
                 std::fabs(g1.origin_y - (g0.origin_y - 0.3)) < 1e-6,
             "the window moved by whole cells: 3 in x, -3 in y");
  int carried = 0;
  int fresh = 0;
  int wrong = 0;
  for (int r = 0; r < p.geometry.rows; ++r) {
    for (int c = 0; c < p.geometry.cols; ++c) {
      const std::size_t i = static_cast<std::size_t>(r) * p.geometry.cols + c;
      const int sr = r - 3;
      const int sc = c + 3;
      if (sr >= 0 && sc < p.geometry.cols) {
        const std::size_t j = static_cast<std::size_t>(sr) * p.geometry.cols + sc;
        ++carried;
        wrong += after.elevation[i] == before.elevation[j] && after.count[i] == before.count[j] &&
                         std::fabs(after.time[i] - 1.0F) < 1e-6F
                     ? 0
                     : 1;
      } else {
        ++fresh;
        wrong += std::isnan(after.elevation[i]) && std::isnan(after.count[i]) ? 0 : 1;
      }
    }
  }
  VERIFY_MSG(carried == 37 * 47 && fresh == 40 * 50 - 37 * 47, "cells carried and entered");
  VERIFY_MSG(wrong == 0, "carried cells keep their values, new cells are unknown, " +
                             std::to_string(wrong) + " wrong");
  // A cell carried over is where it was in the map frame: its value is the
  // ramp's height at its center.
  int misplaced = 0;
  for (int r = 0; r < p.geometry.rows; ++r) {
    for (int c = 0; c < p.geometry.cols; ++c) {
      const float e = after.elevation[static_cast<std::size_t>(r) * p.geometry.cols + c];
      const float truth = terrain_height(ramped, static_cast<float>(cell_x(g1, c)),
                                         static_cast<float>(cell_y(g1, r)));
      misplaced += std::isnan(e) || std::fabs(e - truth) < 1e-4F ? 0 : 1;
    }
  }
  VERIFY_MSG(misplaced == 0, "carried cells stay put in the map frame");
  // A jump larger than the window leaves nothing.
  update(mapper, {}, test::translation(20.0, 0.0, 0.0), 3 * kSecond);
  const HostLayers gone = read_layers(mapper);
  VERIFY(std::all_of(gone.elevation.begin(), gone.elevation.end(),
                     [](float v) { return std::isnan(v); }));
}

// Repeated noisy frames of the same ground: the fused variance falls, and the
// estimate improves on a single frame.
void check_fusion() {
  HeightMapParams p = square_grid(ElevationMode::kFused);
  p.noise_base = 0.01F;
  p.mahalanobis_gate = 0.0F;
  const Terrain flat{.base = 0.2F};
  HeightMapper mapper(p);
  std::vector<double> median_variance;
  double first_rms = 0.0;
  double last_rms = 0.0;
  for (std::uint32_t frame = 0; frame < 10; ++frame) {
    // 4 points per cell, 1 cm noise.
    update(mapper, terrain_cloud(flat, -1.0F, 1.0F, -1.0F, 1.0F, 0.025F, 0.01F, 100U + frame),
           Transform{}, (frame + 1) * kSecond / 10);
    const HostLayers layers = read_layers(mapper);
    std::vector<double> variances;
    double squares = 0.0;
    for (std::size_t i = 0; i < layers.elevation.size(); ++i) {
      if (std::isfinite(layers.elevation[i])) {
        variances.push_back(layers.variance[i]);
        squares += (layers.elevation[i] - 0.2) * (layers.elevation[i] - 0.2);
      }
    }
    VERIFY(variances.size() == 40U * 40U);
    median_variance.push_back(percentile(variances, 0.5));
    const double rms = std::sqrt(squares / static_cast<double>(variances.size()));
    (frame == 0 ? first_rms : last_rms) = rms;
  }
  for (std::size_t k = 1; k < median_variance.size(); ++k) {
    VERIFY_MSG(median_variance[k] < median_variance[k - 1],
               "variance falls at frame " + std::to_string(k));
  }
  std::printf("  fusion: median variance %.3g -> %.3g m^2, rms %.2f -> %.2f mm over 10 frames\n",
              median_variance.front(), median_variance.back(), 1e3 * first_rms, 1e3 * last_rms);
  VERIFY_MSG(median_variance.back() < median_variance.front() / 5.0, "variance falls fivefold");
  VERIFY_MSG(last_rms < first_rms / 2.0, "ten frames beat one: rms " + std::to_string(first_rms) +
                                             " -> " + std::to_string(last_rms));
  VERIFY_MSG(last_rms < 0.0025, "fused rms " + std::to_string(last_rms));
}

// The clearing and decay policy, and the gate.
void check_decay() {
  const Terrain flat{.base = 0.1F};
  const std::vector<float> cloud = terrain_cloud(flat, -1.0F, 1.0F, -1.0F, 1.0F, 0.02F);
  const auto known = [](const HostLayers& layers) {
    return std::count_if(layers.elevation.begin(), layers.elevation.end(),
                         [](float v) { return std::isfinite(v); });
  };
  // Without accumulate, the map is the latest cloud's.
  HeightMapParams p = square_grid(ElevationMode::kMean);
  p.accumulate = false;
  HeightMapper single(p);
  update(single, cloud, Transform{}, kSecond);
  VERIFY(known(read_layers(single)) == 1600);
  update(single, {}, Transform{}, 2 * kSecond);
  VERIFY_MSG(known(read_layers(single)) == 0, "without accumulate, nothing stays");

  // max_age.
  p = square_grid(ElevationMode::kMax);
  p.max_age = 0.5F;
  HeightMapper aged(p);
  update(aged, cloud, Transform{}, kSecond);
  update(aged, {}, Transform{}, kSecond + 4 * kSecond / 10);
  HostLayers layers = read_layers(aged);
  VERIFY_MSG(known(layers) == 1600, "younger than max_age stays");
  const auto time = std::find_if(layers.time.begin(), layers.time.end(),
                                 [](float v) { return std::isfinite(v); });
  VERIFY_MSG(time != layers.time.end() && std::fabs(*time - 0.4F) < 1e-6F,
             "time since update is 0.4 s");
  update(aged, {}, Transform{}, kSecond + 6 * kSecond / 10);
  VERIFY_MSG(known(read_layers(aged)) == 0, "older than max_age is cleared");

  // variance_growth and max_variance.
  p = square_grid(ElevationMode::kFused);
  p.variance_growth = 1e-4F;
  p.max_variance = 5e-4F;
  HeightMapper growing(p);
  update(growing, cloud, Transform{}, kSecond);
  const float v0 = read_layers(growing).variance[40 * 80 + 40];
  update(growing, {}, Transform{}, 3 * kSecond);
  const float v1 = read_layers(growing).variance[40 * 80 + 40];
  VERIFY_MSG(std::fabs(v1 - (v0 + 2e-4F)) < 1e-8F, "the variance grows by 2e-4 in 2 s");
  update(growing, {}, Transform{}, 6 * kSecond);
  VERIFY_MSG(known(read_layers(growing)) == 0, "past max_variance, cleared");

  // The gate: a box appears where the ground was.
  Terrain boxed;
  add_feature(boxed, box(-1.0F, 1.0F, -1.0F, 1.0F, 0.3F));
  for (const float gate : {3.0F, 0.0F}) {
    p = square_grid(ElevationMode::kFused);
    p.mahalanobis_gate = gate;
    HeightMapper changing(p);
    for (int k = 0; k < 5; ++k) {
      update(changing, cloud, Transform{}, (k + 1) * kSecond);
    }
    update(changing, terrain_cloud(boxed, -1.0F, 1.0F, -1.0F, 1.0F, 0.02F), Transform{},
           6 * kSecond);
    const float e = read_layers(changing).elevation[40 * 80 + 40];
    if (gate > 0.0F) {
      VERIFY_MSG(std::fabs(e - 0.3F) < 1e-5F,
                 "with the gate the box replaces the ground: " + std::to_string(e));
    } else {
      VERIFY_MSG(e > 0.11F && e < 0.29F, "without it the box is averaged in");
    }
  }

  // A pit appears where the ground was: a lower measurement is dropped at
  // first, and taken once the dropped ones have widened the gate.
  Terrain pitted{.base = 0.1F};
  add_feature(pitted, pit(-1.0F, 1.0F, -1.0F, 1.0F, 0.3F));
  p = square_grid(ElevationMode::kFused);
  p.rejection_variance = 0.004F;
  HeightMapper lowering(p);
  for (int k = 0; k < 5; ++k) {
    update(lowering, cloud, Transform{}, (k + 1) * kSecond);
  }
  const HostLayers settled = read_layers(lowering);
  const std::vector<float> pit_cloud = terrain_cloud(pitted, -1.0F, 1.0F, -1.0F, 1.0F, 0.02F);
  update(lowering, pit_cloud, Transform{}, 6 * kSecond);
  HostLayers lowered = read_layers(lowering);
  const std::size_t center = 40 * 80 + 40;
  VERIFY_MSG(lowered.elevation[center] == settled.elevation[center] &&
                 std::fabs(lowered.variance[center] - (settled.variance[center] + 0.004F)) < 1e-7F,
             "a lower measurement is dropped, and widens the gate");
  for (int k = 0; k < 3; ++k) {
    update(lowering, pit_cloud, Transform{}, (7 + k) * kSecond);
  }
  lowered = read_layers(lowering);
  VERIFY_MSG(std::fabs(lowered.elevation[center] + 0.2F) < 0.002F,
             "lower terrain that persists is taken: " + std::to_string(lowered.elevation[center]));
}

// The GPU against the CPU reference, bit for bit, on a small moving grid:
// rotated sensor frames, noise, invalid points, points outside every limit,
// several frames, each elevation mode and every decay setting.
void check_reference() {
  const Terrain terrain = course_terrain();
  for (const ElevationMode mode :
       {ElevationMode::kMax, ElevationMode::kMean, ElevationMode::kFused}) {
    for (const bool follow : {false, true}) {
      HeightMapParams p;
      p.geometry = MapGeometry{
          .resolution = 0.07F, .rows = 30, .cols = 36, .origin_x = -0.4, .origin_y = -1.1};
      p.mode = follow ? MapMode::kFollow : MapMode::kFixed;
      p.elevation = mode;
      p.min_height = -0.15F;
      p.max_height = 0.33F;
      p.max_range = 3.5F;
      p.min_range = 0.3F;
      p.max_age = 0.25F;
      p.variance_growth = 2e-5F;
      p.max_variance = 0.004F;
      p.mahalanobis_gate = 2.5F;
      HeightMapper mapper(p);
      test::ReferenceMapper reference(p);
      std::mt19937 rng(5U);
      std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
      for (int frame = 0; frame < 8; ++frame) {
        const double yaw = 0.3 * frame;
        const Transform pose = camera_pose(0.9 + 0.13 * frame, -0.2 + 0.05 * frame, 1.0, yaw, 0.6);
        std::vector<float> map_cloud =
            terrain_cloud(terrain, -0.5F + 0.1F * frame, 2.5F, -1.2F, 1.2F, 0.023F, 0.004F,
                          static_cast<std::uint32_t>(frame));
        std::vector<float> cloud = test::transformed(map_cloud, test::inverse(pose));
        for (std::size_t i = 0; i < cloud.size(); i += 3) {
          if (uniform(rng) < 0.03F) {
            cloud[i + static_cast<std::size_t>(frame % 3)] =
                std::numeric_limits<float>::quiet_NaN();
          }
        }
        const std::int64_t stamp = kSecond + frame * (kSecond / 10) + (frame == 5 ? kSecond : 0);
        update(mapper, cloud, pose, stamp);
        reference.update(cloud, pose, stamp);
        const HostLayers got = read_layers(mapper);
        const test::ReferenceMapper::Layers want = reference.layers();
        const std::string at = "mode " + std::to_string(static_cast<int>(mode)) +
                               (follow ? " follow" : " fixed") + " frame " + std::to_string(frame);
        const bool frame_same =
            same_floats(got.elevation, want.elevation) &&
            same_floats(got.variance, want.variance) && same_floats(got.count, want.count) &&
            same_floats(got.min_elevation, want.min_elevation) &&
            same_floats(got.max_elevation, want.max_elevation) && same_floats(got.time, want.time);
        VERIFY_MSG(frame_same, at + ": layers match the reference bit for bit");
        VERIFY_MSG(mapper.geometry().origin_x == reference.geometry().origin_x &&
                       mapper.geometry().origin_y == reference.geometry().origin_y,
                   at + ": the window matches");
        const auto known = std::count_if(got.elevation.begin(), got.elevation.end(),
                                         [](float v) { return std::isfinite(v); });
        VERIFY_MSG(known > 50, at + ": cells known");
      }
    }
  }
}

// The same clouds give the same bits, in one mapper run twice and in a
// fresh one; also with a cloud that crowds many points into few cells.
void check_determinism() {
  const Terrain terrain = course_terrain();
  const std::vector<float> dense = terrain_cloud(terrain, 0.0F, 4.0F, -1.5F, 1.5F, 0.004F, 0.003F);
  for (const ElevationMode mode : {ElevationMode::kMean, ElevationMode::kFused}) {
    HeightMapParams p;
    p.geometry =
        MapGeometry{.resolution = 0.2F, .rows = 30, .cols = 30, .origin_x = -1.0, .origin_y = -3.0};
    p.elevation = mode;
    std::vector<HostLayers> runs;
    for (int run = 0; run < 2; ++run) {
      HeightMapper mapper(p);
      for (int frame = 0; frame < 3; ++frame) {
        update(mapper, dense, test::translation(0.0, 0.0, 0.01 * frame), (frame + 1) * kSecond);
      }
      runs.push_back(read_layers(mapper));
    }
    VERIFY(dense.size() / 3 > 700000U);
    VERIFY_MSG(same_floats(runs[0].elevation, runs[1].elevation) &&
                   same_floats(runs[0].variance, runs[1].variance) &&
                   same_floats(runs[0].min_elevation, runs[1].min_elevation),
               "two mappers give the same bits");
  }
}

// A depth camera over the course, with depth noise, as the demo sees it,
// mapped with its pose.
void check_camera_view() {
  const Terrain terrain = course_terrain();
  TerrainCamera camera;
  camera.width = 320;
  camera.height = 240;
  camera.fx = camera.fy = 225.0F;
  camera.cx = 159.5F;
  camera.cy = 119.5F;
  camera.pose = camera_pose(0.6, 0.0, 1.1, 0.0, 0.7);
  camera.noise_base = 0.001F;
  camera.noise_quadratic = 0.002F;
  camera.seed = 3U;
  const std::size_t pixels = 320U * 240U;
  test::Scratch<float> depth(pixels);
  test::Scratch<float> xyz(3 * pixels);
  VERIFY(render_terrain_depth(terrain, camera, depth.data, xyz.data, nullptr) == cudaSuccess);
  HeightMapParams p;
  p.geometry =
      MapGeometry{.resolution = 0.04F, .rows = 100, .cols = 125, .origin_x = 0.0, .origin_y = -2.0};
  p.noise_base = camera.noise_base;
  p.noise_quadratic = camera.noise_quadratic;
  HeightMapper mapper(p);
  VERIFY(mapper.update(xyz.data, static_cast<std::uint32_t>(pixels), camera.pose, kSecond,
                       nullptr) == cudaSuccess);
  const HostLayers layers = read_layers(mapper);
  const MapGeometry& g = p.geometry;
  std::vector<double> errors_m;
  for (int r = 0; r < g.rows; ++r) {
    for (int c = 0; c < g.cols; ++c) {
      const float e = layers.elevation[static_cast<std::size_t>(r) * g.cols + c];
      const double x = cell_x(g, c);
      const double y = cell_y(g, r);
      if (std::isfinite(e) && terrain_edge_distance(terrain, x, y) > 0.08) {
        errors_m.push_back(
            std::fabs(e - terrain_height(terrain, static_cast<float>(x), static_cast<float>(y))));
      }
    }
  }
  std::printf("  camera view: median %.2f mm, p95 %.2f mm over %zu cells\n",
              1e3 * percentile(errors_m, 0.5), 1e3 * percentile(errors_m, 0.95), errors_m.size());
  VERIFY_MSG(errors_m.size() > 3000, "cells seen: " + std::to_string(errors_m.size()));
  VERIFY_MSG(percentile(errors_m, 0.5) < 0.003,
             "camera median error " + std::to_string(percentile(errors_m, 0.5)));
  VERIFY_MSG(percentile(errors_m, 0.95) < 0.015,
             "camera p95 error " + std::to_string(percentile(errors_m, 0.95)));
}

// Empty, invalid and out-of-range clouds leave every cell unknown.
void check_edge_cases() {
  const auto all_unknown = [](const HostLayers& layers) {
    for (const auto* layer : {&layers.elevation, &layers.variance, &layers.count,
                              &layers.min_elevation, &layers.max_elevation, &layers.time}) {
      const bool unknown =
          std::all_of(layer->begin(), layer->end(), [](float v) { return std::isnan(v); });
      if (!unknown) {
        return false;
      }
    }
    return true;
  };
  const HeightMapParams p = square_grid(ElevationMode::kFused);
  {
    HeightMapper mapper(p);
    VERIFY_MSG(all_unknown(read_layers(mapper)), "before any update, unknown");
    VERIFY(mapper.update(nullptr, 0, Transform{}, kSecond, nullptr) == cudaSuccess);
    VERIFY_MSG(all_unknown(read_layers(mapper)), "empty cloud");
  }
  {
    HeightMapper mapper(p);
    update(mapper, terrain_cloud(Terrain{}, 5.0F, 7.0F, -1.0F, 1.0F, 0.02F), Transform{}, kSecond);
    VERIFY_MSG(all_unknown(read_layers(mapper)), "a cloud outside the grid");
  }
  {
    HeightMapParams clipped = p;
    clipped.max_height = -1.0F;
    clipped.min_height = -2.0F;
    HeightMapper mapper(clipped);
    update(mapper, terrain_cloud(Terrain{}, -1.0F, 1.0F, -1.0F, 1.0F, 0.02F), Transform{}, kSecond);
    VERIFY_MSG(all_unknown(read_layers(mapper)), "a cloud above max_height");
  }
  {
    HeightMapper mapper(p);
    std::vector<float> invalid(3000, std::numeric_limits<float>::quiet_NaN());
    update(mapper, invalid, Transform{}, kSecond);
    VERIFY_MSG(all_unknown(read_layers(mapper)), "an all-NaN cloud");
    const float inf = std::numeric_limits<float>::infinity();
    update(mapper, {inf, 0.0F, 0.0F, 0.0F, -inf, 0.0F}, Transform{}, kSecond);
    VERIFY_MSG(all_unknown(read_layers(mapper)), "infinite points");
  }
  {
    HeightMapper mapper(p);
    VERIFY(mapper.update(nullptr, 5, Transform{}, kSecond, nullptr) == cudaErrorInvalidValue);
  }
  // A single point.
  {
    HeightMapper mapper(p);
    update(mapper, {0.01F, 0.01F, 0.5F}, Transform{}, kSecond);
    const HostLayers layers = read_layers(mapper);
    const std::size_t i = 40 * 80 + 40;
    VERIFY(layers.elevation[i] == 0.5F && layers.count[i] == 1.0F &&
           std::fabs(layers.variance[i] - 2.5e-5F) < 1e-10F);
    VERIFY(std::count_if(layers.elevation.begin(), layers.elevation.end(),
                         [](float v) { return std::isfinite(v); }) == 1);
  }
}

void check_rejected_params() {
  const auto rejects = [](HeightMapParams p, const std::string& what) {
    try {
      HeightMapper mapper(p);
      VERIFY_MSG(false, what + " is rejected");
    } catch (const std::invalid_argument& error) {
      VERIFY_MSG(std::string(error.what()).find(what) != std::string::npos,
                 "'" + std::string(error.what()) + "' names '" + what + "'");
    }
  };
  HeightMapParams p;
  p.geometry.resolution = 0.0F;
  rejects(p, "resolution");
  p = HeightMapParams{};
  p.geometry.rows = 0;
  rejects(p, "rows and cols");
  p = HeightMapParams{};
  p.geometry.cols = 9000;
  rejects(p, "rows and cols");
  p = HeightMapParams{};
  p.min_height = 1.0F;
  p.max_height = 0.0F;
  rejects(p, "min_height");
  p = HeightMapParams{};
  p.max_range = -1.0F;
  rejects(p, "min_range");
  p = HeightMapParams{};
  p.noise_base = 0.0F;
  p.noise_quadratic = 0.0F;
  rejects(p, "noise_base");
  p = HeightMapParams{};
  p.max_age = -1.0F;
  rejects(p, "max_age");
  p = HeightMapParams{};
  p.min_variance = 0.0F;
  rejects(p, "min_variance");
  p = HeightMapParams{};
  p.rejection_variance = -1.0F;
  rejects(p, "rejection_variance");
}

}  // namespace
}  // namespace holoscan::height_map

int main() {
  using holoscan::height_map::test::TestCase;
  namespace hm = holoscan::height_map;
  return holoscan::height_map::test::run({
      TestCase{"analytic terrains", hm::check_analytic_terrains},
      TestCase{"height and range limits", hm::check_limits},
      TestCase{"moving window", hm::check_moving_window},
      TestCase{"temporal fusion", hm::check_fusion},
      TestCase{"clearing, decay and the gate", hm::check_decay},
      TestCase{"CPU reference", hm::check_reference},
      TestCase{"determinism", hm::check_determinism},
      TestCase{"a depth camera over the course", hm::check_camera_view},
      TestCase{"empty, invalid and out-of-range clouds", hm::check_edge_cases},
      TestCase{"rejected settings", hm::check_rejected_params},
  });
}
