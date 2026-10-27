// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/apriltag/synthetic_scene.hpp"

// The AprilTag 3 library's C API.
extern "C" {
#include <apriltag.h>

#include <common/image_u8.h>
}

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "homography.hpp"
#include "tag_family.hpp"

namespace holoscan::apriltag {
namespace {

using Matrix3 = std::array<double, 9>;

Matrix3 multiply(const Matrix3& a, const Matrix3& b) {
  Matrix3 c{};
  for (int r = 0; r < 3; ++r) {
    for (int k = 0; k < 3; ++k) {
      c[r * 3 + k] = a[r * 3] * b[k] + a[r * 3 + 1] * b[3 + k] + a[r * 3 + 2] * b[6 + k];
    }
  }
  return c;
}

Matrix3 rotation_x(double angle) {
  const double c = std::cos(angle), s = std::sin(angle);
  return {1, 0, 0, 0, c, -s, 0, s, c};
}
Matrix3 rotation_y(double angle) {
  const double c = std::cos(angle), s = std::sin(angle);
  return {c, 0, s, 0, 1, 0, -s, 0, c};
}
Matrix3 rotation_z(double angle) {
  const double c = std::cos(angle), s = std::sin(angle);
  return {c, -s, 0, s, c, 0, 0, 0, 1};
}

// A deterministic per-pixel value in [-1, 1].
double noise(int x, int y) {
  std::uint32_t h =
      static_cast<std::uint32_t>(x) * 374761393U + static_cast<std::uint32_t>(y) * 668265263U;
  h = (h ^ (h >> 13U)) * 1274126177U;
  h ^= h >> 16U;
  return static_cast<double>(h & 0xFFFFU) / 32767.5 - 1.0;
}

struct ImageDeleter {
  void operator()(image_u8_t* image) const noexcept { image_u8_destroy(image); }
};

}  // namespace

SyntheticScene make_scene(const std::string& family, std::uint32_t frame, int width, int height) {
  if (width < 320 || height < 240) {
    throw std::invalid_argument("synthetic scenes need at least 320x240 pixels");
  }
  const TagFamily tag_family(family);
  SyntheticScene scene;
  scene.width = width;
  scene.height = height;
  scene.family = family;
  scene.tag_size = 0.12;
  const double focal = 0.9 * width;
  scene.intrinsics = {focal, focal, (width - 1) / 2.0, (height - 1) / 2.0};

  // Four distinct ids spread over the family.
  const auto ncodes = static_cast<int>(tag_family.get()->ncodes);
  // Tag centers as fractions of the image size from its center, inside the
  // frame at every pose.
  constexpr std::array<std::array<double, 2>, 4> kAnchors{
      {{-0.22, -0.24}, {0.22, -0.24}, {-0.22, 0.24}, {0.22, 0.24}}};
  const double t = frame * 0.05;
  for (int i = 0; i < 4; ++i) {
    const double phase = t + i * std::numbers::pi / 2.0;
    const double depth = 1.1 + 0.2 * std::sin(0.7 * phase);
    const double yaw = 0.6 * std::sin(phase);
    const double pitch = 0.45 * std::cos(1.3 * phase);
    // Each tag turned a different number of quarter turns in plane, so every
    // decoding orientation is exercised.
    const double roll = i * std::numbers::pi / 2.0 + 0.5 * std::sin(0.5 * phase + i);
    SceneTag tag;
    tag.id = i * (ncodes / 4) + 1;
    tag.pose.rotation = multiply(rotation_y(yaw), multiply(rotation_x(pitch), rotation_z(roll)));
    tag.pose.translation = {kAnchors[i][0] * width / focal * depth,
                            kAnchors[i][1] * height / focal * depth, depth};
    scene.tags.push_back(tag);
  }
  return scene;
}

std::array<Point2, 4> project_corners(const SceneTag& tag, const CameraIntrinsics& intrinsics,
                                      double tag_size) {
  const auto& r = tag.pose.rotation;
  const auto& p = tag.pose.translation;
  std::array<Point2, 4> corners{};
  for (int i = 0; i < 4; ++i) {
    const double x = ideal_x(i) * tag_size / 2.0;
    const double y = ideal_y(i) * tag_size / 2.0;
    const double cx = r[0] * x + r[1] * y + p[0];
    const double cy = r[3] * x + r[4] * y + p[1];
    const double cz = r[6] * x + r[7] * y + p[2];
    corners[i] = {intrinsics.fx * cx / cz + intrinsics.cx, intrinsics.fy * cy / cz + intrinsics.cy};
  }
  return corners;
}

std::vector<std::uint8_t> render_scene(const SyntheticScene& scene, int channels) {
  if (channels != 1 && channels != 3) {
    throw std::invalid_argument("render_scene supports 1 or 3 channels");
  }
  const TagFamily family(scene.family);
  const int border = family.get()->width_at_border;
  const int total = family.get()->total_width;
  std::vector<std::unique_ptr<image_u8_t, ImageDeleter>> bitmaps;
  for (const SceneTag& tag : scene.tags) {
    bitmaps.emplace_back(apriltag_to_image(family.get(), static_cast<std::uint32_t>(tag.id)));
  }
  // Tag cells are tag_size / border meters; the printed tag gets one extra cell
  // of white paper around its full width.
  const double cell = scene.tag_size / border;
  const double paper_half = (total / 2.0 + 1.0) * cell;
  const auto& k = scene.intrinsics;

  // Pixels each tag's paper can cover, so the rest skip supersampling.
  struct Box {
    int x0, y0, x1, y1;
  };
  std::vector<Box> boxes;
  for (const SceneTag& tag : scene.tags) {
    const auto corners = project_corners(tag, k, 2.0 * paper_half);
    double x0 = HUGE_VAL, y0 = HUGE_VAL, x1 = -HUGE_VAL, y1 = -HUGE_VAL;
    for (const auto& c : corners) {
      x0 = std::min(x0, c[0]);
      y0 = std::min(y0, c[1]);
      x1 = std::max(x1, c[0]);
      y1 = std::max(y1, c[1]);
    }
    boxes.push_back({static_cast<int>(std::floor(x0)) - 2, static_cast<int>(std::floor(y0)) - 2,
                     static_cast<int>(std::ceil(x1)) + 2, static_cast<int>(std::ceil(y1)) + 2});
  }

  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(scene.width) * scene.height * channels);
  constexpr int kSamples = 3;
  for (int v = 0; v < scene.height; ++v) {
    for (int u = 0; u < scene.width; ++u) {
      const double background = 95.0 + 70.0 * u / scene.width + 25.0 * v / scene.height;
      const bool covered = std::any_of(boxes.begin(), boxes.end(), [u, v](const Box& b) {
        return u >= b.x0 && u <= b.x1 && v >= b.y0 && v <= b.y1;
      });
      const int samples = covered ? kSamples : 1;
      double sum = 0.0;
      for (int sy = 0; sy < samples; ++sy) {
        for (int sx = 0; sx < samples; ++sx) {
          const double px = u + (sx + 0.5) / samples - 0.5;
          const double py = v + (sy + 0.5) / samples - 0.5;
          double value = background;
          double nearest = HUGE_VAL;
          for (std::size_t i = 0; covered && i < scene.tags.size(); ++i) {
            const Box& b = boxes[i];
            if (u < b.x0 || u > b.x1 || v < b.y0 || v > b.y1) continue;
            const std::array<double, 3> ray{(px - k.cx) / k.fx, (py - k.cy) / k.fy, 1.0};
            const auto& r = scene.tags[i].pose.rotation;
            const auto& t = scene.tags[i].pose.translation;
            // The tag plane through t with normal R's z column.
            const double n_dot_t = r[2] * t[0] + r[5] * t[1] + r[8] * t[2];
            const double n_dot_ray = r[2] * ray[0] + r[5] * ray[1] + r[8] * ray[2];
            if (std::abs(n_dot_ray) < 1e-12) continue;
            const double s = n_dot_t / n_dot_ray;
            if (s <= 0.0 || s >= nearest) continue;
            const double dx = s * ray[0] - t[0];
            const double dy = s * ray[1] - t[1];
            const double dz = s * ray[2] - t[2];
            // Tag-frame coordinates: R^T (P - t).
            const double x = r[0] * dx + r[3] * dy + r[6] * dz;
            const double y = r[1] * dx + r[4] * dy + r[7] * dz;
            if (std::abs(x) > paper_half || std::abs(y) > paper_half) continue;
            nearest = s;
            const int col = static_cast<int>(std::floor(x / cell + total / 2.0));
            const int row = static_cast<int>(std::floor(y / cell + total / 2.0));
            const image_u8_t* bitmap = bitmaps[i].get();
            const bool inside = col >= 0 && row >= 0 && col < total && row < total;
            const bool white = !inside || bitmap->buf[row * bitmap->stride + col] > 127;
            value = white ? 225.0 : 25.0;
          }
          sum += value;
        }
      }
      const double shade = sum / (samples * samples) + 4.0 * noise(u, v);
      const auto level = static_cast<std::uint8_t>(std::clamp(std::lround(shade), 0L, 255L));
      std::uint8_t* out =
          pixels.data() + (static_cast<std::size_t>(v) * scene.width + u) * channels;
      for (int c = 0; c < channels; ++c) {
        out[c] = level;
      }
    }
  }
  return pixels;
}

TruthCounts match_truth(const SyntheticScene& scene, std::span<const Detection> detections,
                        const std::function<void(const TruthMatch&)>& on_match) {
  std::map<int, const SceneTag*> expected;
  for (const SceneTag& tag : scene.tags) expected[tag.id] = &tag;
  TruthCounts counts;
  for (const Detection& detection : detections) {
    const auto it = expected.find(detection.id);
    if (it == expected.end()) {
      ++counts.extra;
      continue;
    }
    const SceneTag& tag = *it->second;
    const auto corners = project_corners(tag, scene.intrinsics, scene.tag_size);
    TruthMatch match{detection, tag, {}, std::nullopt};
    for (int c = 0; c < 4; ++c) {
      match.corner_errors[c] = std::hypot(detection.corners[c][0] - corners[c][0],
                                          detection.corners[c][1] - corners[c][1]);
    }
    if (detection.pose) {
      const auto& t = tag.pose.translation;
      const auto& found = detection.pose->translation;
      match.translation_error = std::hypot(found[0] - t[0], found[1] - t[1], found[2] - t[2]) /
                                std::hypot(t[0], t[1], t[2]);
    }
    on_match(match);
    expected.erase(it);
  }
  counts.missed = static_cast<int>(expected.size());
  return counts;
}

}  // namespace holoscan::apriltag
