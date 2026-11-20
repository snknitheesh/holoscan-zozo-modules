// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The settings' rules, each refusal naming its field; the grid's size; and
// whether a transform is rigid: no SDK, CUDA or GPU.

#include <cmath>
#include <functional>
#include <limits>
#include <string>

#include "holoscan/height_map/grid.hpp"
#include "holoscan/height_map/params.hpp"
#include "test_support.hpp"

namespace holoscan::height_map {
namespace {

constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

// `change` applied to the defaults is refused, and the reason names `field`.
template <typename Params>
void refuses(const std::function<void(Params&)>& change, const std::string& field) {
  Params p;
  change(p);
  const char* what = nullptr;
  const bool valid = valid_params(p, &what);
  VERIFY_MSG(!valid, field + " is refused");
  VERIFY_MSG(what != nullptr && std::string(what).find(field) != std::string::npos,
             "the reason names " + field + ": " + (what != nullptr ? what : "none"));
}

void check_map_params() {
  VERIFY_MSG(valid_params(HeightMapParams{}), "the defaults are valid");
  // valid_params() without a reason.
  HeightMapParams bad;
  bad.geometry.rows = 0;
  VERIFY(!valid_params(bad));
  using P = HeightMapParams;
  refuses<P>([](P& p) { p.geometry.resolution = 0.0F; }, "resolution");
  refuses<P>([](P& p) { p.geometry.resolution = kNan; }, "resolution");
  refuses<P>([](P& p) { p.geometry.rows = 0; }, "rows and cols");
  refuses<P>([](P& p) { p.geometry.cols = kMaxGridSide + 1; }, "rows and cols");
  refuses<P>([](P& p) { p.geometry.origin_x = kInf; }, "origin");
  refuses<P>([](P& p) { p.mode = static_cast<MapMode>(7); }, "mode");
  refuses<P>([](P& p) { p.elevation = static_cast<ElevationMode>(7); }, "elevation");
  refuses<P>(
      [](P& p) {
        p.min_height = 1.0F;
        p.max_height = 0.0F;
      },
      "min_height");
  refuses<P>([](P& p) { p.max_height = kNan; }, "min_height");
  refuses<P>([](P& p) { p.min_range = -1.0F; }, "min_range");
  refuses<P>([](P& p) { p.max_range = kNan; }, "min_range");
  // A point at the sensor needs noise_base for a finite weight.
  refuses<P>([](P& p) { p.noise_base = 0.0F; }, "noise_base");
  refuses<P>([](P& p) { p.noise_base = kInf; }, "noise_base");
  refuses<P>([](P& p) { p.noise_quadratic = -0.1F; }, "noise_quadratic");
  refuses<P>([](P& p) { p.min_variance = 0.0F; }, "min_variance");
  refuses<P>([](P& p) { p.mahalanobis_gate = -1.0F; }, "mahalanobis_gate");
  refuses<P>([](P& p) { p.rejection_variance = -1.0F; }, "rejection_variance");
  refuses<P>([](P& p) { p.max_age = -1.0F; }, "max_age");
  refuses<P>([](P& p) { p.variance_growth = kInf; }, "variance_growth");
  refuses<P>([](P& p) { p.max_variance = -1.0F; }, "max_variance");
  // Limits that are open on one side are fine.
  HeightMapParams open;
  open.min_height = -kInf;
  open.max_range = kInf;
  open.noise_quadratic = 0.0F;
  VERIFY(valid_params(open));
}

void check_traversability_params() {
  VERIFY_MSG(valid_params(TraversabilityParams{}), "the defaults are valid");
  using P = TraversabilityParams;
  refuses<P>([](P& p) { p.window_radius = 0; }, "window_radius");
  refuses<P>([](P& p) { p.window_radius = 11; }, "window_radius");
  refuses<P>([](P& p) { p.min_valid_fraction = 1.5F; }, "min_valid_fraction");
  refuses<P>([](P& p) { p.min_valid_fraction = kNan; }, "min_valid_fraction");
  refuses<P>([](P& p) { p.step_radius = 0; }, "step_radius");
  refuses<P>([](P& p) { p.max_slope_deg = 0.0F; }, "max_slope_deg");
  refuses<P>([](P& p) { p.max_step = kInf; }, "max_step");
  refuses<P>([](P& p) { p.slope_weight = -1.0F; }, "weights");
  refuses<P>(
      [](P& p) {
        p.slope_weight = 0.0F;
        p.step_weight = 0.0F;
        p.roughness_weight = 0.0F;
      },
      "weights");
}

Transform rotation_z(double radians, double tx, double ty, double tz) {
  const double c = std::cos(radians);
  const double s = std::sin(radians);
  return Transform{{c, -s, 0.0, tx, s, c, 0.0, ty, 0.0, 0.0, 1.0, tz}};
}

void check_rigid() {
  VERIFY(is_rigid(Transform{}));
  VERIFY(is_rigid(rotation_z(0.5, 1.0, -2.0, 3.0)));
  Transform scaled = rotation_z(0.5, 0.0, 0.0, 0.0);
  scaled.m[10] = 2.0;
  VERIFY_MSG(!is_rigid(scaled), "a scale");
  Transform mirrored;
  mirrored.m[0] = -1.0;
  VERIFY_MSG(!is_rigid(mirrored), "a reflection");
  Transform sheared;
  sheared.m[1] = 0.5;
  VERIFY_MSG(!is_rigid(sheared), "a shear");
  Transform invalid;
  invalid.m[3] = std::numeric_limits<double>::quiet_NaN();
  VERIFY_MSG(!is_rigid(invalid), "a NaN translation");
  // Within the tolerance, a rotation rounded to float is still rigid.
  Transform rounded = rotation_z(0.3, 0.0, 0.0, 0.0);
  rounded.m[0] += 1e-6;
  VERIFY(is_rigid(rounded));
  VERIFY(!is_rigid(rounded, 1e-8));
}

void check_geometry() {
  const MapGeometry g{.resolution = 0.05F, .rows = 200, .cols = 300};
  VERIFY(g.cells() == 60000U);
  const MapGeometry largest{.rows = kMaxGridSide, .cols = kMaxGridSide};
  VERIFY(largest.cells() == 67108864U);
}

}  // namespace
}  // namespace holoscan::height_map

int main() {
  using holoscan::height_map::test::TestCase;
  namespace hm = holoscan::height_map;
  return holoscan::height_map::test::run({
      TestCase{"map settings", hm::check_map_params},
      TestCase{"traversability settings", hm::check_traversability_params},
      TestCase{"rigid transforms", hm::check_rigid},
      TestCase{"grid", hm::check_geometry},
  });
}
