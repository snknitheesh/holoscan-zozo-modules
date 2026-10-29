// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/fiducials/synthetic_scene.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/fiducials/dictionary.hpp>
#include <holoscan/fiducials/pose.hpp>

namespace holoscan::fiducials {
namespace {

using Matrix3 = std::array<double, 9>;

constexpr double kWhite = 225.0;
constexpr double kBlack = 25.0;

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

void check_size(int width, int height) {
  if (width < 320 || height < 240) {
    throw std::invalid_argument("synthetic scenes need at least 320x240 pixels");
  }
}

SyntheticScene base_scene(std::string dictionary, int width, int height) {
  check_size(width, height);
  SyntheticScene scene;
  scene.width = width;
  scene.height = height;
  scene.dictionary = std::move(dictionary);
  const double focal = 0.9 * width;
  scene.intrinsics = {focal, focal, (width - 1) / 2.0, (height - 1) / 2.0};
  scene.noise_sigma = 2.0;
  return scene;
}

// A planar target: where a ray from the camera meets it, and what is printed there.
struct Target {
  Pose pose;
  // The printed value at (x, y) of the target's frame; negative off the target.
  std::function<double(double x, double y)> paint;
  std::array<Point3, 4> extent;  // Its outline, for the pixels it can cover.
};

// Projects an outline and returns the box of pixels it covers, grown by 2.
std::array<int, 4> pixel_box(const SyntheticScene& scene, const Target& target) {
  double x0 = HUGE_VAL, y0 = HUGE_VAL, x1 = -HUGE_VAL, y1 = -HUGE_VAL;
  for (const Point3& p : target.extent) {
    const Point2 q = project(scene.intrinsics, target.pose, p);
    x0 = std::min(x0, q[0]);
    y0 = std::min(y0, q[1]);
    x1 = std::max(x1, q[0]);
    y1 = std::max(y1, q[1]);
  }
  return {static_cast<int>(std::floor(x0)) - 2, static_cast<int>(std::floor(y0)) - 2,
          static_cast<int>(std::ceil(x1)) + 2, static_cast<int>(std::ceil(y1)) + 2};
}

std::vector<Target> targets(const SyntheticScene& scene) {
  std::vector<Target> result;
  if (!scene.markers.empty()) {
    const Dictionary dictionary(scene.dictionary);
    const int side = dictionary.marker_size() + 2;
    for (const SceneMarker& marker : scene.markers) {
      const double cell = marker.length / side;
      const double half = 0.5 * marker.length;
      const double paper = half + cell;  // One cell of white around the border.
      auto cells = dictionary.cells(marker.id);
      result.push_back(
          {marker.pose,
           [cells = std::move(cells), cell, half, paper, side](double x, double y) -> double {
             if (std::abs(x) > paper || std::abs(y) > paper) return -1.0;
             const int col = static_cast<int>(std::floor((x + half) / cell));
             const int row = static_cast<int>(std::floor((y + half) / cell));
             if (col < 0 || row < 0 || col >= side || row >= side) return kWhite;
             return cells[static_cast<std::size_t>(row) * side + col] ? kWhite : kBlack;
           },
           {{{-paper, -paper, 0}, {paper, -paper, 0}, {paper, paper, 0}, {-paper, paper, 0}}}});
    }
  }
  if (scene.board) {
    const CharucoBoard board(scene.board->config);
    const Dictionary dictionary(scene.board->config.dictionary);
    const int side = dictionary.marker_size() + 2;
    std::vector<std::vector<std::uint8_t>> cells;
    for (int i = 0; i < board.marker_count(); ++i) {
      cells.push_back(dictionary.cells(scene.board->config.first_marker_id + i));
    }
    const auto& c = scene.board->config;
    const double width = c.squares_x * c.square_length;
    const double height = c.squares_y * c.square_length;
    const double margin = 0.5 * c.square_length;
    const double inset = 0.5 * (c.square_length - c.marker_length);
    const double cell = c.marker_length / side;
    // Each square's marker index, -1 for a black square.
    std::vector<int> square_marker(static_cast<std::size_t>(c.squares_x) * c.squares_y, -1);
    for (int i = 0; i < board.marker_count(); ++i) {
      const auto [x, y] = board.marker_square(i);
      square_marker[static_cast<std::size_t>(y) * c.squares_x + x] = i;
    }
    result.push_back(
        {scene.board->pose,
         [=, cells = std::move(cells), square_marker = std::move(square_marker)](
             double x, double y) -> double {
           if (x < -margin || y < -margin || x > width + margin || y > height + margin) return -1.0;
           if (x < 0.0 || y < 0.0 || x >= width || y >= height) return kWhite;
           const int sx = std::min(static_cast<int>(x / c.square_length), c.squares_x - 1);
           const int sy = std::min(static_cast<int>(y / c.square_length), c.squares_y - 1);
           const int marker = square_marker[static_cast<std::size_t>(sy) * c.squares_x + sx];
           if (marker < 0) return kBlack;
           const int col = static_cast<int>(std::floor((x - sx * c.square_length - inset) / cell));
           const int row = static_cast<int>(std::floor((y - sy * c.square_length - inset) / cell));
           if (col < 0 || row < 0 || col >= side || row >= side) return kWhite;
           return cells[marker][static_cast<std::size_t>(row) * side + col] ? kWhite : kBlack;
         },
         {{{-margin, -margin, 0},
           {width + margin, -margin, 0},
           {width + margin, height + margin, 0},
           {-margin, height + margin, 0}}}});
  }
  return result;
}

// The value the camera sees along the ray through pixel (u, v), or `background`.
double trace(const SyntheticScene& scene, const std::vector<Target>& all,
             const std::vector<std::array<int, 4>>& boxes, double u, double v, int pu, int pv,
             double background) {
  const auto& k = scene.intrinsics;
  const std::array<double, 3> ray{(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0};
  double nearest = HUGE_VAL;
  double value = background;
  for (std::size_t i = 0; i < all.size(); ++i) {
    const auto& b = boxes[i];
    if (pu < b[0] || pu > b[2] || pv < b[1] || pv > b[3]) continue;
    const auto& r = all[i].pose.rotation;
    const auto& t = all[i].pose.translation;
    const double n_dot_t = r[2] * t[0] + r[5] * t[1] + r[8] * t[2];
    const double n_dot_ray = r[2] * ray[0] + r[5] * ray[1] + r[8] * ray[2];
    if (std::abs(n_dot_ray) < 1e-12) continue;
    const double s = n_dot_t / n_dot_ray;
    if (s <= 0.0 || s >= nearest) continue;
    const double dx = s * ray[0] - t[0];
    const double dy = s * ray[1] - t[1];
    const double dz = s * ray[2] - t[2];
    const double painted =
        all[i].paint(r[0] * dx + r[3] * dy + r[6] * dz, r[1] * dx + r[4] * dy + r[7] * dz);
    if (painted < 0.0) continue;
    nearest = s;
    value = painted;
  }
  return value;
}

bool inside(const std::array<Point2, 4>& q, double x, double y) {
  bool negative = false;
  bool positive = false;
  for (int i = 0; i < 4; ++i) {
    const Point2& a = q[i];
    const Point2& b = q[(i + 1) % 4];
    const double side = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
    negative = negative || side < 0.0;
    positive = positive || side > 0.0;
  }
  return !(negative && positive);
}

// Separable Gaussian blur of a float image, edges clamped.
void blur(std::vector<double>& image, int width, int height, double sigma) {
  const int radius = static_cast<int>(std::ceil(3.0 * sigma));
  std::vector<double> kernel(2 * radius + 1);
  double total = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    kernel[i + radius] = std::exp(-0.5 * i * i / (sigma * sigma));
    total += kernel[i + radius];
  }
  for (double& w : kernel) w /= total;
  std::vector<double> temp(image.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double sum = 0.0;
      for (int i = -radius; i <= radius; ++i) {
        sum += kernel[i + radius] *
               image[static_cast<std::size_t>(y) * width + std::clamp(x + i, 0, width - 1)];
      }
      temp[static_cast<std::size_t>(y) * width + x] = sum;
    }
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double sum = 0.0;
      for (int i = -radius; i <= radius; ++i) {
        sum += kernel[i + radius] *
               temp[static_cast<std::size_t>(std::clamp(y + i, 0, height - 1)) * width + x];
      }
      image[static_cast<std::size_t>(y) * width + x] = sum;
    }
  }
}

}  // namespace

SyntheticScene make_marker_scene(const std::string& dictionary, std::uint32_t frame, int width,
                                 int height) {
  SyntheticScene scene = base_scene(dictionary, width, height);
  const int count = Dictionary(dictionary).count();
  const double focal = scene.intrinsics.fx;
  // Marker centers as fractions of the image size from its center, inside the frame at every
  // pose.
  constexpr std::array<std::array<double, 2>, 6> kAnchors{
      {{-0.30, -0.22}, {0.0, -0.22}, {0.30, -0.22}, {-0.30, 0.22}, {0.0, 0.22}, {0.30, 0.22}}};
  const double t = frame * 0.05;
  for (int i = 0; i < 6; ++i) {
    const double phase = t + i * std::numbers::pi / 3.0;
    const double depth = 1.0 + 0.2 * std::sin(0.7 * phase);
    const double yaw = 0.5 * std::sin(phase);
    const double pitch = 0.45 * std::cos(1.3 * phase);
    const double roll = i * std::numbers::pi / 2.0 + 0.5 * std::sin(0.5 * phase + i);
    SceneMarker marker;
    marker.id = (i * count) / 6 + 1;
    marker.length = 0.10;
    marker.pose.rotation = multiply(rotation_y(yaw), multiply(rotation_x(pitch), rotation_z(roll)));
    marker.pose.translation = {kAnchors[i][0] * width / focal * depth,
                               kAnchors[i][1] * height / focal * depth, depth};
    scene.markers.push_back(marker);
  }
  return scene;
}

SyntheticScene make_board_scene(const CharucoBoardConfig& config, std::uint32_t frame, int width,
                                int height) {
  const CharucoBoard board(config);  // Validates.
  SyntheticScene scene = base_scene(config.dictionary, width, height);
  const double t = frame * 0.05;
  const double board_w = config.squares_x * config.square_length;
  const double board_h = config.squares_y * config.square_length;
  // About half the image's height, from the board's larger side.
  const double depth = scene.intrinsics.fx * std::max(board_w, board_h) / (0.55 * height) *
                       (1.0 + 0.1 * std::sin(0.9 * t));
  const Pose pose{.rotation = multiply(rotation_y(0.6 * std::sin(t)),
                                       multiply(rotation_x(0.45 * std::cos(1.3 * t)),
                                                rotation_z(0.4 * std::sin(0.7 * t)))),
                  .translation = {0.0, 0.0, depth}};
  // Center the board on the optical axis, then move it a little.
  const auto& r = pose.rotation;
  const double cx = 0.5 * board_w;
  const double cy = 0.5 * board_h;
  SceneBoard placed{config, pose};
  const double shift_x = 0.08 * width / scene.intrinsics.fx * depth * std::sin(0.4 * t);
  const double shift_y = 0.06 * height / scene.intrinsics.fy * depth * std::cos(0.5 * t);
  placed.pose.translation = {-(r[0] * cx + r[1] * cy) + shift_x, -(r[3] * cx + r[4] * cy) + shift_y,
                             depth - (r[6] * cx + r[7] * cy)};
  scene.board = placed;
  return scene;
}

std::vector<std::uint8_t> render_scene(const SyntheticScene& scene, int channels) {
  if (channels != 1 && channels != 3) {
    throw std::invalid_argument("render_scene supports 1 or 3 channels");
  }
  const std::vector<Target> all = targets(scene);
  std::vector<std::array<int, 4>> boxes;
  for (const Target& target : all) boxes.push_back(pixel_box(scene, target));
  const int width = scene.width;
  const int height = scene.height;
  std::vector<double> image(static_cast<std::size_t>(width) * height);
  constexpr int kSamples = 8;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const double background = 95.0 + 70.0 * u / width + 25.0 * v / height;
      // The pixel's center and corners; where they agree the pixel is not on an edge.
      const double center = trace(scene, all, boxes, u, v, u, v, background);
      bool edge = false;
      for (const auto& [dx, dy] : {std::pair{-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5}, {0.5, 0.5}}) {
        edge = edge || trace(scene, all, boxes, u + dx, v + dy, u, v, background) != center;
      }
      double value = center;
      if (edge) {
        double sum = 0.0;
        for (int sy = 0; sy < kSamples; ++sy) {
          for (int sx = 0; sx < kSamples; ++sx) {
            sum += trace(scene, all, boxes, u + (sx + 0.5) / kSamples - 0.5,
                         v + (sy + 0.5) / kSamples - 0.5, u, v, background);
          }
        }
        value = sum / (kSamples * kSamples);
      }
      for (const Occluder& occluder : scene.occluders) {
        if (inside(occluder.corners, u, v)) value = occluder.gray;
      }
      image[static_cast<std::size_t>(v) * width + u] = value;
    }
  }
  if (scene.blur_sigma > 0.0) blur(image, width, height, scene.blur_sigma);
  std::mt19937 random(scene.seed);
  std::normal_distribution<double> noise(0.0, scene.noise_sigma > 0.0 ? scene.noise_sigma : 1.0);
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * channels);
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      double value = image[static_cast<std::size_t>(v) * width + u];
      if (scene.noise_sigma > 0.0) value += noise(random);
      const auto level = static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
      const int x = scene.mirrored ? width - 1 - u : u;
      std::uint8_t* out = pixels.data() + (static_cast<std::size_t>(v) * width + x) * channels;
      for (int c = 0; c < channels; ++c) out[c] = level;
    }
  }
  return pixels;
}

std::array<Point2, 4> marker_truth(const SyntheticScene& scene, const SceneMarker& marker) {
  const auto corners = marker_object_corners(marker.length);
  std::array<Point2, 4> result;
  for (int i = 0; i < 4; ++i) result[i] = project(scene.intrinsics, marker.pose, corners[i]);
  return result;
}

std::vector<CharucoCorner> board_truth(const SyntheticScene& scene) {
  std::vector<CharucoCorner> result;
  if (!scene.board) return result;
  const CharucoBoard board(scene.board->config);
  for (int id = 0; id < board.corner_count(); ++id) {
    const Point3 position = board.corner(id);
    const Point2 pixel = project(scene.intrinsics, scene.board->pose, position);
    if (pixel[0] < 0.0 || pixel[1] < 0.0 || pixel[0] > scene.width - 1.0 ||
        pixel[1] > scene.height - 1.0) {
      continue;
    }
    const bool hidden =
        std::any_of(scene.occluders.begin(), scene.occluders.end(),
                    [&](const Occluder& o) { return inside(o.corners, pixel[0], pixel[1]); });
    if (!hidden) result.push_back({.id = id, .image = pixel, .board = position});
  }
  return result;
}

double rotation_error_deg(const std::array<double, 9>& a, const std::array<double, 9>& b) {
  double trace = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) trace += a[k * 3 + i] * b[k * 3 + i];
  }
  return std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / std::numbers::pi;
}

TruthCounts match_markers(const SyntheticScene& scene, std::span<const Marker> found,
                          const std::function<void(const MarkerMatch&)>& on_match) {
  std::map<int, const SceneMarker*> expected;
  for (const SceneMarker& marker : scene.markers) expected[marker.id] = &marker;
  TruthCounts counts;
  for (const Marker& marker : found) {
    const auto it = expected.find(marker.id);
    if (it == expected.end()) {
      ++counts.extra;
      continue;
    }
    const SceneMarker& truth = *it->second;
    const auto corners = marker_truth(scene, truth);
    MarkerMatch match{marker, truth, {}, std::nullopt, std::nullopt};
    for (int c = 0; c < 4; ++c) {
      match.corner_errors[c] =
          std::hypot(marker.corners[c][0] - corners[c][0], marker.corners[c][1] - corners[c][1]);
    }
    if (marker.pose) {
      const auto& t = truth.pose.translation;
      const auto& p = marker.pose->translation;
      match.translation_error =
          std::hypot(p[0] - t[0], p[1] - t[1], p[2] - t[2]) / std::hypot(t[0], t[1], t[2]);
      match.rotation_error_deg = rotation_error_deg(marker.pose->rotation, truth.pose.rotation);
    }
    on_match(match);
    expected.erase(it);
  }
  counts.missed = static_cast<int>(expected.size());
  return counts;
}

}  // namespace holoscan::fiducials
