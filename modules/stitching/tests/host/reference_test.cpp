// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The output geometry and the CPU reference without a GPU: output points and
// their inverse, the tables of simple rigs against hand computation (a region
// of interest across a panorama's edge included), the blend's feathering,
// hard seams and background, and the gain solve recovering known exposures.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/stitching/camera_rig.hpp>
#include <holoscan/stitching/projection.hpp>
#include <holoscan/stitching/remap.hpp>

#include "test_support.hpp"

namespace holoscan::stitching {
namespace {

Camera pinhole(const std::string& name, double yaw, double pitch, std::array<double, 3> position) {
  Camera c;
  c.name = name;
  c.width = 160;
  c.height = 120;
  c.fx = 80.0;
  c.fy = 80.0;
  c.cx = 79.5;
  c.cy = 59.5;
  c.camera_from_rig = mount(position, yaw, pitch);
  return c;
}

void output_points() {
  OutputGeometry cylinder;
  cylinder.projection = Projection::kCylindrical;
  cylinder.width = 720;
  cylinder.height = 200;
  cylinder.panorama = PanoramaParams{.azimuth_center_deg = 30.0,
                                     .horizontal_fov_deg = 360.0,
                                     .elevation_min_deg = -20.0,
                                     .elevation_max_deg = 40.0,
                                     .center = {0.5, -0.2, 1.0},
                                     .radius = 4.0};
  OutputGeometry sphere = cylinder;
  sphere.projection = Projection::kSpherical;
  sphere.panorama.radius = 0.0;
  sphere.panorama.horizontal_fov_deg = 200.0;
  OutputGeometry ground;
  ground.width = 300;
  ground.height = 200;
  ground.birds_eye = BirdsEyeParams{-2.0, 8.0, -5.0, 10.0};
  for (const OutputGeometry& g : {cylinder, sphere, ground}) {
    for (double y = 0.0; y < g.height; y += 13.0) {
      for (double x = 0.0; x < g.width; x += 17.0) {
        OutputPoint p = output_point(g, x, y);
        if (p.at_infinity) {
          // A direction: any point along it from the centre.
          for (int i = 0; i < 3; ++i) p.point[i] += g.panorama.center[i];
        }
        const auto back = output_pixel(g, p.point);
        VERIFY_MSG(back && std::abs(back->x - x) < 1e-9 && std::abs(back->y - y) < 1e-9,
                   std::string(projection_name(g.projection)) + " round trip");
      }
    }
  }
  // The middle column of the cylinder looks along its azimuth, on the cylinder.
  const OutputPoint middle = output_point(cylinder, 359.5, 0.0);
  const double azimuth = std::atan2(middle.point[1] + 0.2, middle.point[0] - 0.5);
  VERIFY(std::abs(azimuth - std::numbers::pi / 6.0) < 1e-12);
  VERIFY(std::abs(std::hypot(middle.point[0] - 0.5, middle.point[1] + 0.2) - 4.0) < 1e-12);
  VERIFY(!middle.at_infinity && output_point(sphere, 1.0, 1.0).at_infinity);
  // The top edge of the sphere is its top elevation; ground: up is forward,
  // right is the rig's right.
  const OutputPoint top = output_point(sphere, 100.0, -0.5);
  VERIFY(std::abs(std::asin(top.point[2]) - 40.0 * std::numbers::pi / 180.0) < 1e-12);
  const OutputPoint corner = output_point(ground, -0.5, -0.5);
  VERIFY(corner.point[0] == 8.0 && corner.point[1] == 10.0 && corner.point[2] == 0.0);
  VERIFY(wraps(cylinder) && !wraps(sphere) && !wraps(ground));

  // The bird's-eye helper: the area over the resolution, rounded up, grown.
  const OutputGeometry view = birds_eye_view(BirdsEyeParams{-1.0, 3.0, -2.05, 2.0}, 0.1);
  VERIFY(view.width == 41 && view.height == 40);
  VERIFY(std::abs(view.birds_eye.y_max - view.birds_eye.y_min - 4.1) < 1e-12);
  VERIFY(test::throws<std::invalid_argument>(
      [] { (void)birds_eye_view(BirdsEyeParams{0.0, 1.0, 0.0, 1.0}, 0.0); }));
  VERIFY(test::throws<std::invalid_argument>(
      [] { (void)birds_eye_view(BirdsEyeParams{0.0, 100.0, 0.0, 1.0}, 0.001); }));
  OutputGeometry bad = cylinder;
  bad.panorama.elevation_max_deg = 90.0;
  VERIFY(test::throws<std::invalid_argument>([&] { validate(bad); }));
  bad = cylinder;
  bad.panorama.horizontal_fov_deg = 361.0;
  VERIFY(test::throws<std::invalid_argument>([&] { validate(bad); }));
  bad = cylinder;
  bad.panorama.radius = -1.0;
  VERIFY(test::throws<std::invalid_argument>([&] { validate(bad); }));
  bad = ground;
  bad.width = 0;
  VERIFY(test::throws<std::invalid_argument>([&] { validate(bad); }));
}

void tables_by_hand() {
  // A camera 2 m up looking straight down sees the ground below it; each
  // ground point lands where the pinhole puts it.
  CameraRig rig;
  rig.cameras.push_back(pinhole("down", 0.0, -90.0, {0.0, 0.0, 2.0}));
  OutputGeometry g;
  g.width = 100;
  g.height = 100;
  g.birds_eye = BirdsEyeParams{-10.0, 10.0, -10.0, 10.0};
  const RemapTables tables = build_remap_tables(rig, g);
  const CameraTable& t = tables.cameras[0];
  // It sees 2 m * 79.5 / 80 ahead and behind, 2 m * 59.5 / 80 to the sides,
  // plus half a pixel: about +-2 m by +-1.5 m, so 20 x 16 pixels at 0.2 m.
  VERIFY(t.width > 0 && t.width <= 20 && t.height > 0 && t.height <= 22);
  VERIFY(t.uv.size() == 2U * t.width * t.height && t.margin.size() == 1U * t.width * t.height);
  for (int dy = 0; dy < t.height; ++dy) {
    for (int dx = 0; dx < t.width; ++dx) {
      const std::size_t i = static_cast<std::size_t>(dy) * t.width + dx;
      if (t.margin[i] <= 0.0F) continue;
      const OutputPoint p = output_point(g, t.x0 + dx, t.y0 + dy);
      // Looking down, image right is the rig's -y and image down is -x.
      const double u = 79.5 - 80.0 * p.point[1] / 2.0;
      const double v = 59.5 - 80.0 * p.point[0] / 2.0;
      VERIFY(std::abs(t.uv[2 * i] - u) < 1e-3 && std::abs(t.uv[2 * i + 1] - v) < 1e-3);
      const double margin = std::min({u + 0.5, 159.5 - u, v + 0.5, 119.5 - v});
      VERIFY(std::abs(t.margin[i] - margin) < 1e-3);
    }
  }
  // A camera looking backwards covers both edges of a 360-degree panorama: its
  // region runs across the right edge.
  CameraRig back;
  back.cameras.push_back(pinhole("back", 180.0, 0.0, {0.0, 0.0, 1.0}));
  OutputGeometry pano;
  pano.projection = Projection::kCylindrical;
  pano.width = 360;
  pano.height = 60;
  pano.panorama.center = {0.0, 0.0, 1.0};
  const RemapTables wrapped = build_remap_tables(back, pano);
  const CameraTable& w = wrapped.cameras[0];
  // 2 * atan(80 / 80) = 90 degrees: 90 columns around column 0 and 359.
  VERIFY(wrapped.wraps && w.x0 > 300 && w.x0 + w.width > 360 && w.width >= 89 && w.width <= 91);
  // Looking forward, it sits in the middle.
  CameraRig front;
  front.cameras.push_back(pinhole("front", 0.0, 0.0, {0.0, 0.0, 1.0}));
  const RemapTables front_tables = build_remap_tables(front, pano);
  const CameraTable& f = front_tables.cameras[0];
  VERIFY_MSG(f.x0 >= 134 && f.x0 + f.width <= 226,
             std::to_string(f.x0) + " + " + std::to_string(f.width));
  // Looking up at a bird's-eye view, it sees nothing.
  CameraRig up;
  up.cameras.push_back(pinhole("up", 0.0, 90.0, {0.0, 0.0, 1.0}));
  const RemapTables up_tables = build_remap_tables(up, g);
  const CameraTable& u = up_tables.cameras[0];
  VERIFY(u.width == 0 && u.height == 0 && u.uv.empty());
}

// A solid image of `value` for each camera.
struct SolidImages {
  std::vector<std::vector<std::uint8_t>> pixels;
  std::vector<HostImage> images;
};

SolidImages solid(const CameraRig& rig, const std::vector<std::array<std::uint8_t, 3>>& colours) {
  SolidImages s;
  for (std::size_t c = 0; c < rig.cameras.size(); ++c) {
    const Camera& cam = rig.cameras[c];
    std::vector<std::uint8_t> p(static_cast<std::size_t>(cam.width) * cam.height * 3U);
    for (std::size_t i = 0; i < p.size(); ++i) p[i] = colours[c][i % 3];
    s.pixels.push_back(std::move(p));
  }
  for (std::size_t c = 0; c < rig.cameras.size(); ++c) {
    const Camera& cam = rig.cameras[c];
    s.images.push_back(HostImage{s.pixels[c].data(), cam.width, cam.height,
                                 static_cast<std::size_t>(cam.width) * 3U, PixelFormat::kRgb8});
  }
  return s;
}

void blending() {
  // Two cameras 60 degrees apart with 90-degree views overlap by 30 degrees.
  CameraRig rig;
  rig.cameras.push_back(pinhole("a", 30.0, 0.0, {0.0, 0.0, 1.0}));
  rig.cameras.push_back(pinhole("b", -30.0, 0.0, {0.0, 0.0, 1.0}));
  OutputGeometry pano;
  pano.projection = Projection::kCylindrical;
  pano.width = 360;
  pano.height = 40;
  pano.panorama.center = {0.0, 0.0, 1.0};
  pano.panorama.elevation_min_deg = -10.0;
  pano.panorama.elevation_max_deg = 10.0;
  const RemapTables tables = build_remap_tables(rig, pano);
  const SolidImages images = solid(rig, {{{100, 100, 100}}, {{200, 200, 200}}});
  const auto row_of = [&](const BlendSettings& settings) {
    const int ch = channels(settings.output_format);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(pano.width) * pano.height * ch);
    reference_stitch(tables, images.images, settings, {}, out.data(),
                     static_cast<std::size_t>(pano.width) * ch);
    std::vector<int> row;
    for (int x = 0; x < pano.width; ++x) {
      row.push_back(out[(static_cast<std::size_t>(20) * pano.width + x) * ch]);
    }
    return std::pair{row, out};
  };
  // Azimuth 0 (between the cameras) is column 180; camera a (left) covers
  // azimuth -15..75, columns 105..195, camera b 165..255.
  const auto [feather, feather_out] = row_of(BlendSettings{});
  VERIFY(feather[120] == 100 && feather[240] == 200);
  VERIFY(feather[180] > 140 && feather[180] < 160);  // Halfway, both at equal margin.
  for (int x = 166; x < 195; ++x) VERIFY(feather[x] <= feather[x + 1]);  // A ramp.
  VERIFY(feather[20] == 0 && feather_out[(20 * 360 + 20) * 4 + 3] == 0);
  VERIFY(feather_out[(20 * 360 + 180) * 4 + 3] == 255);
  BlendSettings hard;
  hard.mode = BlendMode::kHardSeam;
  hard.output_format = PixelFormat::kRgb8;
  hard.background = {9, 8, 7};
  const auto [seam, seam_out] = row_of(hard);
  // The seam at the middle, the background where neither sees.
  VERIFY(seam[178] == 100 && seam[181] == 200 && seam[20] == 9 && seam_out[20 * 3 + 2] == 7);
  // A capped weight reaches its cap 2 pixels from each camera's edge: the
  // ramps are short, and between them both count equally.
  BlendSettings narrow;
  narrow.feather_width = 2.0F;
  const auto [capped, unused] = row_of(narrow);
  VERIFY(capped[120] == 100 && capped[240] == 200);
  VERIFY(capped[172] == 150 && capped[180] == 150 && capped[188] == 150);
  // Gains scale each camera.
  std::vector<std::uint8_t> out(static_cast<std::size_t>(pano.width) * pano.height * 4U);
  const std::vector<float> gains{1.5F, 0.5F};
  reference_stitch(tables, images.images, BlendSettings{}, gains, out.data(),
                   static_cast<std::size_t>(pano.width) * 4U);
  VERIFY(out[(20 * 360 + 120) * 4] == 150 && out[(20 * 360 + 240) * 4] == 100);
  // Mismatched images are refused.
  std::vector<HostImage> one{images.images[0]};
  VERIFY(test::throws<std::invalid_argument>(
      [&] { reference_stitch(tables, one, BlendSettings{}, {}, out.data(), pano.width * 4U); }));
}

void gains() {
  // Three cameras in a row whose overlaps show the same scene, at exposures
  // 1, 0.7 and 1.3: camera i's overlap mean is 120 * exposure_i.
  const std::vector<double> exposure{1.0, 0.7, 1.3};
  OverlapStats stats;
  stats.sum.assign(9, 0U);
  stats.count.assign(9, 0U);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j || std::abs(i - j) > 1) continue;
      stats.count[i * 3 + j] = 1000U;
      stats.sum[i * 3 + j] = static_cast<std::uint64_t>(std::llround(120.0 * exposure[i] * 1000.0));
    }
  }
  const std::vector<float> g = reference_gains(stats, BlendSettings{}, {});
  double lo = 1e9;
  double hi = 0.0;
  for (int i = 0; i < 3; ++i) {
    lo = std::min(lo, g[i] * exposure[i]);
    hi = std::max(hi, g[i] * exposure[i]);
  }
  VERIFY_MSG((hi - lo) / lo < 0.01, "spread " + std::to_string((hi - lo) / lo));
  // Brown and Lowe's sigma_g = 0.1 holds the gains nearer to 1.
  BlendSettings strict;
  strict.exposure_sigma_gain = 0.1F;
  const std::vector<float> held = reference_gains(stats, strict, {});
  VERIFY(std::abs(held[1] - 1.0F) < std::abs(g[1] - 1.0F));
  // Smoothing: a quarter of the way from the previous gains.
  BlendSettings smooth;
  smooth.exposure_smoothing = 0.75F;
  const std::vector<float> previous{1.0F, 1.0F, 1.0F};
  const std::vector<float> s = reference_gains(stats, smooth, previous);
  for (int i = 0; i < 3; ++i) VERIFY(std::abs(s[i] - (0.25F * g[i] + 0.75F)) < 1e-5F);
  // No overlap (or too few samples): gain 1.
  OverlapStats lonely;
  lonely.sum.assign(4, 0U);
  lonely.count.assign(4, 0U);
  lonely.count[1] = 5U;
  lonely.sum[1] = 500U;
  for (const float gain : reference_gains(lonely, BlendSettings{}, {})) VERIFY(gain == 1.0F);
  OverlapStats odd;
  odd.sum.assign(5, 0U);
  odd.count.assign(5, 0U);
  VERIFY(test::throws<std::invalid_argument>([&] { (void)reference_gains(odd, {}, {}); }));
  BlendSettings bad;
  bad.exposure_smoothing = 1.0F;
  VERIFY(test::throws<std::invalid_argument>([&] { validate(bad); }));
}

void names() {
  for (const auto p : {Projection::kCylindrical, Projection::kSpherical, Projection::kBirdsEye}) {
    VERIFY(parse_projection(projection_name(p)) == p);
  }
  for (const auto m : {BlendMode::kFeather, BlendMode::kHardSeam}) {
    VERIFY(parse_blend_mode(blend_mode_name(m)) == m);
  }
  for (const auto f : {PixelFormat::kMono8, PixelFormat::kRgb8, PixelFormat::kRgba8,
                       PixelFormat::kBgr8, PixelFormat::kBgra8}) {
    VERIFY(parse_format(format_name(f)) == f);
  }
  VERIFY(!parse_projection("fisheye") && !parse_blend_mode("multiband") && !parse_format("nv12"));
}

}  // namespace
}  // namespace holoscan::stitching

int main() {
  namespace st = holoscan::stitching;
  return st::test::run({
      {"output_points", st::output_points},
      {"tables_by_hand", st::tables_by_hand},
      {"blending", st::blending},
      {"gains", st::gains},
      {"names", st::names},
  });
}
