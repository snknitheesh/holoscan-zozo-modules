// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/mono_depth/synthetic_scene.hpp>

namespace holoscan::mono_depth {
namespace {

using Vec3 = std::array<float, 3>;

float dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

Vec3 normalized(Vec3 v) {
  const float length = std::sqrt(dot(v, v));
  return {v[0] / length, v[1] / length, v[2] / length};
}

// The direction the light travels: from the upper left, behind the camera.
const Vec3 kLight = normalized({0.5F, 1.0F, 0.6F});

std::uint8_t to_byte(float value) {
  return static_cast<std::uint8_t>(std::clamp(std::lround(value * 255.0F), 0L, 255L));
}

// A ray's nearest hit: its depth along z, region and color.
struct Hit {
  float z{};
  SceneRegion region{SceneRegion::kWall};
  Vec3 color{};
};

// Diffuse light on a surface with normal `n`, with some ambient.
float lambert(const Vec3& n) { return 0.35F + 0.65F * std::max(0.0F, -dot(n, kLight)); }

Hit trace(const DepthScene& s, const Vec3& ray) {
  // The wall is always behind everything else.
  Hit hit;
  hit.z = s.wall_z;
  hit.region = SceneRegion::kWall;
  {
    const float x = ray[0] * s.wall_z;
    const float y = ray[1] * s.wall_z;
    // Pale plaster with a darker band at eye level, and tiles of 0.5 m.
    const bool band = y > -0.4F && y < -0.1F;
    const bool seam =
        std::fabs(std::remainder(x, 0.5F)) < 0.012F || std::fabs(std::remainder(y, 0.5F)) < 0.012F;
    const float shade = lambert({0.0F, 0.0F, -1.0F});
    const Vec3 base = band ? Vec3{0.55F, 0.62F, 0.72F} : Vec3{0.86F, 0.84F, 0.78F};
    const float seam_factor = seam ? 0.7F : 1.0F;
    hit.color = {base[0] * shade * seam_factor, base[1] * shade * seam_factor,
                 base[2] * shade * seam_factor};
  }
  // The floor, below the horizon.
  if (ray[1] > 0.0F) {
    const float z = s.floor_y / ray[1];
    if (z < hit.z) {
      const float x = ray[0] * z;
      const bool dark =
          (static_cast<int>(std::floor(x / 0.5F)) + static_cast<int>(std::floor(z / 0.5F))) % 2 !=
          0;
      const float shade = lambert({0.0F, -1.0F, 0.0F});
      const Vec3 base = dark ? Vec3{0.25F, 0.22F, 0.2F} : Vec3{0.78F, 0.74F, 0.66F};
      // The sphere's contact shadow.
      const float dx = x - s.sphere_center[0];
      const float dz = z - s.sphere_center[2];
      const float distance = std::sqrt(dx * dx + dz * dz) / s.sphere_radius;
      const float shadow = distance < 1.5F ? 0.55F + 0.45F * distance / 1.5F : 1.0F;
      hit = {z,
             SceneRegion::kFloor,
             {base[0] * shade * shadow, base[1] * shade * shadow, base[2] * shade * shadow}};
    }
  }
  // The panel, a plane at panel_z.
  {
    const float x = ray[0] * s.panel_z;
    const float y = ray[1] * s.panel_z;
    if (s.panel_z < hit.z && x >= s.panel_left && x <= s.panel_right && y >= s.panel_top &&
        y <= s.floor_y) {
      // Blue boards 0.2 m wide with a light frame.
      const bool frame =
          x - s.panel_left < 0.05F || s.panel_right - x < 0.05F || y - s.panel_top < 0.05F;
      const bool groove = std::fabs(std::remainder(x - s.panel_left, 0.2F)) < 0.008F;
      const float shade = lambert({0.0F, 0.0F, -1.0F});
      const Vec3 base = frame    ? Vec3{0.9F, 0.88F, 0.8F}
                        : groove ? Vec3{0.1F, 0.15F, 0.3F}
                                 : Vec3{0.2F, 0.35F, 0.7F};
      hit = {s.panel_z, SceneRegion::kPanel, {base[0] * shade, base[1] * shade, base[2] * shade}};
    }
  }
  // The sphere: |t * ray - c| = r for the smallest t > 0.
  {
    const Vec3& c = s.sphere_center;
    const float a = dot(ray, ray);
    const float b = -2.0F * dot(ray, c);
    const float k = dot(c, c) - s.sphere_radius * s.sphere_radius;
    const float discriminant = b * b - 4.0F * a * k;
    if (discriminant >= 0.0F) {
      const float t = (-b - std::sqrt(discriminant)) / (2.0F * a);
      if (t > 0.0F && t < hit.z) {  // ray[2] is 1, so t is the depth
        const Vec3 p{ray[0] * t, ray[1] * t, ray[2] * t};
        const Vec3 n = normalized({p[0] - c[0], p[1] - c[1], p[2] - c[2]});
        const Vec3 view = normalized({-p[0], -p[1], -p[2]});
        // The light, which travels along kLight, reflected about the normal.
        const float along = 2.0F * dot(n, kLight);
        const Vec3 reflected{kLight[0] - along * n[0], kLight[1] - along * n[1],
                             kLight[2] - along * n[2]};
        const float specular = std::pow(std::max(0.0F, dot(reflected, view)), 24.0F) * 0.6F;
        const float shade = lambert(n);
        hit = {t,
               SceneRegion::kSphere,
               {0.9F * shade + specular, 0.45F * shade + specular, 0.15F * shade + specular}};
      }
    }
  }
  return hit;
}

// Ranks of `values`, ties by their average rank.
std::vector<double> ranks(std::span<const float> values) {
  std::vector<std::size_t> order(values.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(),
            [&values](std::size_t a, std::size_t b) { return values[a] < values[b]; });
  std::vector<double> result(values.size());
  for (std::size_t i = 0; i < order.size();) {
    std::size_t j = i;
    while (j + 1 < order.size() && values[order[j + 1]] == values[order[i]]) {
      ++j;
    }
    const double rank = (static_cast<double>(i) + static_cast<double>(j)) / 2.0;
    for (std::size_t k = i; k <= j; ++k) {
      result[order[k]] = rank;
    }
    i = j + 1;
  }
  return result;
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}

}  // namespace

DepthScene make_depth_scene(int width, int height, std::uint32_t frame) {
  if (width < kMinSceneWidth || height < kMinSceneHeight || width > kMaxSceneSide ||
      height > kMaxSceneSide) {
    throw std::invalid_argument("synthetic images must be " + std::to_string(kMinSceneWidth) + "x" +
                                std::to_string(kMinSceneHeight) + " to " +
                                std::to_string(kMaxSceneSide) + "x" +
                                std::to_string(kMaxSceneSide));
  }
  DepthScene s;
  s.width = width;
  s.height = height;
  // A horizontal field of view of about 67 degrees.
  s.focal = 0.75F * static_cast<float>(width);
  s.cx = 0.5F * static_cast<float>(width);
  s.cy = 0.5F * static_cast<float>(height);
  s.floor_y = 1.2F;
  s.wall_z = 7.0F;
  s.panel_left = -1.6F;
  s.panel_right = -0.45F;
  s.panel_top = -0.5F;
  s.panel_z = 1.8F;
  s.sphere_radius = 0.55F;
  // The sphere rolls around on the floor, well between the panel and the wall.
  const float phase = 0.15F * static_cast<float>(frame % 1000U);
  s.sphere_center = {0.9F + 0.4F * std::sin(phase), s.floor_y - s.sphere_radius,
                     4.4F + 0.3F * std::cos(phase)};
  return s;
}

void render_depth_scene(const DepthScene& scene, PixelFormat format, std::uint8_t* pixels,
                        std::size_t pitch, float* depth, SceneRegion* regions) {
  const int n = channels(format);
  const bool bgr = blue_first(format);
  for (int v = 0; v < scene.height; ++v) {
    std::uint8_t* row = pixels + static_cast<std::size_t>(v) * pitch;
    for (int u = 0; u < scene.width; ++u) {
      const Vec3 ray{(static_cast<float>(u) + 0.5F - scene.cx) / scene.focal,
                     (static_cast<float>(v) + 0.5F - scene.cy) / scene.focal, 1.0F};
      const Hit hit = trace(scene, ray);
      std::uint8_t* pixel = row + static_cast<std::size_t>(u) * static_cast<std::size_t>(n);
      pixel[bgr ? 2 : 0] = to_byte(hit.color[0]);
      pixel[1] = to_byte(hit.color[1]);
      pixel[bgr ? 0 : 2] = to_byte(hit.color[2]);
      if (n == 4) {
        pixel[3] = 255;
      }
      const std::size_t index = static_cast<std::size_t>(v) * scene.width + u;
      if (depth != nullptr) {
        depth[index] = hit.z;
      }
      if (regions != nullptr) {
        regions[index] = hit.region;
      }
    }
  }
}

double spearman(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size() || a.size() < 2) {
    return 0.0;
  }
  const std::vector<double> ra = ranks(a);
  const std::vector<double> rb = ranks(b);
  const double mean = (static_cast<double>(a.size()) - 1.0) / 2.0;
  double covariance = 0.0;
  double variance_a = 0.0;
  double variance_b = 0.0;
  for (std::size_t i = 0; i < ra.size(); ++i) {
    covariance += (ra[i] - mean) * (rb[i] - mean);
    variance_a += (ra[i] - mean) * (ra[i] - mean);
    variance_b += (rb[i] - mean) * (rb[i] - mean);
  }
  if (variance_a == 0.0 || variance_b == 0.0) {
    return 0.0;
  }
  return covariance / std::sqrt(variance_a * variance_b);
}

OrderingResult check_ordering(const DepthScene& scene, std::span<const float> inverse_depth,
                              int stride) {
  const std::size_t pixels = static_cast<std::size_t>(scene.width) * scene.height;
  if (inverse_depth.size() != pixels || stride < 1) {
    throw std::invalid_argument("the inverse depth must hold the scene's image");
  }
  std::vector<std::uint8_t> image(pixels * 3U);
  std::vector<float> depth(pixels);
  std::vector<SceneRegion> regions(pixels);
  render_depth_scene(scene, PixelFormat::kRgb8, image.data(),
                     static_cast<std::size_t>(scene.width) * 3U, depth.data(), regions.data());
  std::vector<float> predicted;
  std::vector<float> truth;
  std::array<std::vector<double>, 4> by_region;
  for (int v = stride / 2; v < scene.height; v += stride) {
    for (int u = stride / 2; u < scene.width; u += stride) {
      const std::size_t index = static_cast<std::size_t>(v) * scene.width + u;
      predicted.push_back(inverse_depth[index]);
      truth.push_back(1.0F / depth[index]);
      by_region[static_cast<std::size_t>(regions[index])].push_back(inverse_depth[index]);
    }
  }
  OrderingResult result;
  result.spearman = spearman(predicted, truth);
  for (std::size_t r = 0; r < by_region.size(); ++r) {
    result.median[r] = median(by_region[r]);
  }
  const auto m = [&result](SceneRegion region) {
    return result.median[static_cast<std::size_t>(region)];
  };
  result.ordered = !by_region[static_cast<std::size_t>(SceneRegion::kPanel)].empty() &&
                   !by_region[static_cast<std::size_t>(SceneRegion::kSphere)].empty() &&
                   m(SceneRegion::kPanel) > m(SceneRegion::kSphere) &&
                   m(SceneRegion::kSphere) > m(SceneRegion::kWall);
  return result;
}

}  // namespace holoscan::mono_depth
