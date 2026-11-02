// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Synthetic terrain with known heights, and a depth camera that sees it: for
// tests, benchmarks and demos. A terrain is flat ground at `base` with up to
// kMaxFeatures features on it, each over an axis-aligned rectangle of the
// map's x-y plane, that do not overlap:
//
// - a ramp rises along +x from the rectangle's x0 edge at a constant grade,
//   and drops back to the ground at its x1 edge;
// - steps rise along +x, `value` meters each `step_length` meters, the first
//   at x0;
// - a box stands `value` meters high;
// - a pit is `value` meters deep.
//
// terrain_height() runs on the host and on the GPU; the module compiles the
// renderer without fused multiply-adds, so host code that evaluates the same
// functions must not contract them either (-ffp-contract=off).

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include <holoscan/height_map/grid.hpp>

#if defined(__CUDACC__)
#define HOLOSCAN_HEIGHT_MAP_HOST_DEVICE __host__ __device__
#else
#define HOLOSCAN_HEIGHT_MAP_HOST_DEVICE
#endif

namespace holoscan::height_map {

enum class FeatureKind : std::uint8_t { kRamp, kSteps, kBox, kPit };

struct TerrainFeature {
  FeatureKind kind{FeatureKind::kBox};
  float x0{};
  float x1{};
  float y0{};
  float y1{};
  // Ramp: the grade, rise per meter (tan of the slope). Steps: each step's
  // height. Box: its height. Pit: its depth. Meters.
  float value{};
  // Steps: each step's length along x, meters.
  float step_length{1.0F};
};

inline constexpr int kMaxFeatures = 8;

struct Terrain {
  float base{};
  int feature_count{};
  TerrainFeature features[kMaxFeatures]{};
};

// The terrain's height at (x, y). Inside a rectangle means x0 <= x < x1 and
// y0 <= y < y1.
HOLOSCAN_HEIGHT_MAP_HOST_DEVICE inline float terrain_height(const Terrain& terrain, float x,
                                                            float y) {
  float h = terrain.base;
  for (int k = 0; k < terrain.feature_count; ++k) {
    const TerrainFeature& f = terrain.features[k];
    if (x < f.x0 || x >= f.x1 || y < f.y0 || y >= f.y1) {
      continue;
    }
    switch (f.kind) {
      case FeatureKind::kRamp:
        h += (x - f.x0) * f.value;
        break;
      case FeatureKind::kSteps:
        h += f.value * (1.0F + floorf((x - f.x0) / f.step_length));
        break;
      case FeatureKind::kBox:
        h += f.value;
        break;
      case FeatureKind::kPit:
        h -= f.value;
        break;
    }
  }
  return h;
}

// Builders. A ramp of `slope_deg` degrees.
[[nodiscard]] TerrainFeature ramp(float x0, float x1, float y0, float y1, float slope_deg);
[[nodiscard]] TerrainFeature steps(float x0, float x1, float y0, float y1, float height,
                                   float length);
[[nodiscard]] TerrainFeature box(float x0, float x1, float y0, float y1, float height);
[[nodiscard]] TerrainFeature pit(float x0, float x1, float y0, float y1, float depth);
// @throws std::invalid_argument when the terrain already has kMaxFeatures.
void add_feature(Terrain& terrain, const TerrainFeature& feature);

// The analytic slope at (x, y), degrees: the ramp's inside a ramp, else 0.
[[nodiscard]] double terrain_slope_deg(const Terrain& terrain, double x, double y);
// The distance from (x, y) to the nearest place where the height jumps: an
// edge of a feature's rectangle, or of a step. Infinity on flat ground alone.
[[nodiscard]] double terrain_edge_distance(const Terrain& terrain, double x, double y);

// The demo's course: flat ground with, from x = 1.5 m on, a 15 degree ramp
// (y 0.3..1.1 m, x 1.5..3 m) beside four 8 cm steps (y -0.9..-0.3 m, x
// 1.5..2.7 m), and beyond them a 30 cm box (y -1.9..-1.3 m, x 3.4..3.8 m)
// and a 20 cm deep pit (y -0.4..0.4 m, x 3.4..4.2 m), placed so that a camera
// on the x axis sees each one's near edge.
[[nodiscard]] Terrain course_terrain();

// A pinhole depth camera looking at a terrain.
struct TerrainCamera {
  int width{640};
  int height{480};
  float fx{450.0F};
  float fy{450.0F};
  float cx{319.5F};
  float cy{239.5F};
  // Maps the camera's optical frame (x right, y down, z forward) into the
  // map frame.
  Transform pose;
  // Rays that hit nothing nearer than this many meters along z see nothing.
  float max_depth{12.0F};
  // Gaussian depth noise, standard deviation noise_base + noise_quadratic *
  // depth^2 meters, drawn from `seed` and the pixel; 0 for none.
  float noise_base{};
  float noise_quadratic{};
  std::uint32_t seed{};
};

// A camera at (x, y, z) in the map frame looking along `yaw_rad` (from +x
// toward +y), tilted down by `pitch_down_rad`.
[[nodiscard]] Transform camera_pose(double x, double y, double z, double yaw_rad,
                                    double pitch_down_rad);

// Renders the camera's float32 [height, width] depth image in meters (0 where
// a ray hits nothing within max_depth) into `depth`, and, when `xyz` is not
// null, the organized float32 [height, width, 3] cloud of those depths in
// the optical frame (NaN where there is none). Device memory; the rays are
// marched in steps of at most 1 cm plus 0.5 % of the distance and the hit is
// refined by bisection, so a ray finds every surface thicker than a step.
cudaError_t render_terrain_depth(const Terrain& terrain, const TerrainCamera& camera, float* depth,
                                 float* xyz, cudaStream_t stream);

}  // namespace holoscan::height_map
