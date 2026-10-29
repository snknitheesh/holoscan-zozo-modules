// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/fiducials/charuco.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/fiducials/dictionary.hpp>
#include <holoscan/fiducials/pose.hpp>

#include "detection_core.hpp"

namespace holoscan::fiducials {
namespace {

// Twice the signed area of (o, a, b).
double cross(const Point2& o, const Point2& a, const Point2& b) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

// The area of the convex hull of `points` (Andrew's monotone chain).
double hull_area(std::vector<Point2> points) {
  if (points.size() < 3) return 0.0;
  std::sort(points.begin(), points.end());
  std::vector<Point2> hull(2 * points.size());
  std::size_t k = 0;
  for (const Point2& p : points) {
    while (k >= 2 && cross(hull[k - 2], hull[k - 1], p) <= 0.0) --k;
    hull[k++] = p;
  }
  for (std::size_t i = points.size() - 1, lower = k + 1; i-- > 0;) {
    while (k >= lower && cross(hull[k - 2], hull[k - 1], points[i]) <= 0.0) --k;
    hull[k++] = points[i];
  }
  double twice = 0.0;
  for (std::size_t i = 0; i + 1 < k; ++i) {
    twice += hull[i][0] * hull[i + 1][1] - hull[i + 1][0] * hull[i][1];
  }
  return 0.5 * std::abs(twice);
}

// Whether the points span the plane: some point is at least `spread` from the line through
// the two farthest apart.
bool spans_plane(std::span<const Point2> points, double spread) {
  if (points.size() < 3) return false;
  std::size_t a = 0;
  std::size_t b = 0;
  double longest = -1.0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (std::size_t j = i + 1; j < points.size(); ++j) {
      const double d = std::hypot(points[j][0] - points[i][0], points[j][1] - points[i][1]);
      if (d > longest) {
        longest = d;
        a = i;
        b = j;
      }
    }
  }
  if (!(longest > 0.0)) return false;
  for (const Point2& p : points) {
    if (std::abs(cross(points[a], points[b], p)) / longest >= spread) return true;
  }
  return false;
}

}  // namespace

CharucoBoard::CharucoBoard(CharucoBoardConfig config) : config_(std::move(config)) {
  const CharucoBoardConfig& c = config_;
  if (c.squares_x < 2 || c.squares_x > 64 || c.squares_y < 2 || c.squares_y > 64) {
    throw std::invalid_argument("squares_x and squares_y must be in 2..64");
  }
  if (static_cast<std::size_t>(corner_count()) > kMaxCharucoCorners) {
    throw std::invalid_argument("the board has more than " + std::to_string(kMaxCharucoCorners) +
                                " chessboard corners");
  }
  if (!(std::isfinite(c.square_length) && c.square_length > 0.0)) {
    throw std::invalid_argument("square_length must be positive");
  }
  if (!(std::isfinite(c.marker_length) && c.marker_length > 0.0 &&
        c.marker_length < c.square_length)) {
    throw std::invalid_argument("marker_length must be positive and less than square_length");
  }
  const Dictionary dictionary(c.dictionary);
  for (int y = 0; y < c.squares_y; ++y) {
    for (int x = 0; x < c.squares_x; ++x) {
      if (has_marker(x, y)) squares_.push_back({x, y});
    }
  }
  if (squares_.size() > kMaxMarkersPerFrame) {
    throw std::invalid_argument("the board has more than " + std::to_string(kMaxMarkersPerFrame) +
                                " markers");
  }
  if (c.first_marker_id < 0 || c.first_marker_id + marker_count() > dictionary.count()) {
    throw std::invalid_argument("the board needs ids " + std::to_string(c.first_marker_id) +
                                " to " + std::to_string(c.first_marker_id + marker_count() - 1) +
                                ", and " + c.dictionary + " has 0 to " +
                                std::to_string(dictionary.count() - 1));
  }
  // Each square's marker index, then each corner's two markers.
  std::vector<int> square_marker(static_cast<std::size_t>(c.squares_x) * c.squares_y, -1);
  for (int i = 0; i < marker_count(); ++i) {
    square_marker[squares_[i][1] * c.squares_x + squares_[i][0]] = i;
  }
  for (int y = 0; y + 1 < c.squares_y; ++y) {
    for (int x = 0; x + 1 < c.squares_x; ++x) {
      std::array<int, 2> pair{-1, -1};
      int n = 0;
      for (const auto& [dx, dy] : {std::pair{0, 0}, {1, 0}, {0, 1}, {1, 1}}) {
        const int marker = square_marker[(y + dy) * c.squares_x + x + dx];
        if (marker >= 0) pair[n++] = marker;
      }
      neighbors_.push_back(pair);
    }
  }
}

int CharucoBoard::corner_count() const noexcept {
  return (config_.squares_x - 1) * (config_.squares_y - 1);
}

std::optional<int> CharucoBoard::marker_index(int id) const noexcept {
  const int index = id - config_.first_marker_id;
  if (index < 0 || index >= marker_count()) return std::nullopt;
  return index;
}

std::array<int, 2> CharucoBoard::marker_square(int index) const { return squares_.at(index); }

std::array<Point3, 4> CharucoBoard::marker_corners(int index) const {
  const auto [x, y] = squares_.at(index);
  const double s = config_.square_length;
  const double m = config_.marker_length;
  const double x0 = x * s + 0.5 * (s - m);
  const double y0 = y * s + 0.5 * (s - m);
  return {{{x0, y0, 0.0}, {x0 + m, y0, 0.0}, {x0 + m, y0 + m, 0.0}, {x0, y0 + m, 0.0}}};
}

Point3 CharucoBoard::corner(int id) const {
  if (id < 0 || id >= corner_count())
    throw std::out_of_range("no chessboard corner " + std::to_string(id));
  const int per_row = config_.squares_x - 1;
  return {(id % per_row + 1) * config_.square_length, (id / per_row + 1) * config_.square_length,
          0.0};
}

std::span<const int> CharucoBoard::corner_markers(int id) const { return neighbors_.at(id); }

bool CharucoBoard::has_marker(int x, int y) const noexcept {
  const bool legacy = config_.legacy_pattern && config_.squares_y % 2 == 0;
  return (x + y) % 2 == (legacy ? 0 : 1);
}

CharucoObservation observe_charuco(const CharucoBoard& board, std::span<const Marker> markers,
                                   const SaddleRefiner& refine, int width, int height,
                                   const CharucoOptions& options,
                                   const std::optional<CameraIntrinsics>& intrinsics) {
  if (options.min_markers < 1 || options.min_markers > 2) {
    throw std::invalid_argument("min_markers must be 1 or 2");
  }
  CharucoObservation result;
  // The board's markers; of two with one id, the one with fewer corrected bits.
  std::vector<const Marker*> found(board.marker_count(), nullptr);
  for (const Marker& marker : markers) {
    const auto index = board.marker_index(marker.id);
    if (!index) continue;
    const Marker*& slot = found[*index];
    if (slot == nullptr || marker.hamming < slot->hamming) slot = &marker;
  }
  for (const Marker* marker : found) {
    if (marker != nullptr) result.markers.push_back(*marker);
  }

  // Each corner predicted by the homography of each of its markers that was found.
  const double marker_length = board.config().marker_length;
  std::vector<SaddleSeed> seeds;
  std::vector<int> seed_ids;
  for (int id = 0; id < board.corner_count(); ++id) {
    const Point3 position = board.corner(id);
    double sum_x = 0.0;
    double sum_y = 0.0;
    double nearest = HUGE_VAL;  // Pixels from the prediction to the nearest marker corner.
    int used = 0;
    for (const int index : board.corner_markers(id)) {
      const Marker* marker = found[index];
      if (marker == nullptr) continue;
      double quad[4][2];
      for (int i = 0; i < 4; ++i) {
        quad[i][0] = marker->corners[i][0];
        quad[i][1] = marker->corners[i][1];
      }
      double h[9];
      if (!core::square_homography(quad, marker_length, h)) continue;
      const Point3 origin = board.marker_corners(index)[0];
      const double u = position[0] - origin[0];
      const double v = position[1] - origin[1];
      const double w = h[6] * u + h[7] * v + h[8];
      sum_x += (h[0] * u + h[1] * v + h[2]) / w;
      sum_y += (h[3] * u + h[4] * v + h[5]) / w;
      ++used;
    }
    if (used < options.min_markers) continue;
    const Point2 predicted{sum_x / used, sum_y / used};
    for (const int index : board.corner_markers(id)) {
      if (found[index] == nullptr) continue;
      for (const Point2& c : found[index]->corners) {
        nearest = std::min(nearest, std::hypot(c[0] - predicted[0], c[1] - predicted[1]));
      }
    }
    // The square window must stay clear of the markers, which sit as far in from the corner
    // along both of the board's axes: about nearest / sqrt(2) pixels.
    const int half_window = std::clamp(static_cast<int>(0.75 * nearest / std::sqrt(2.0)), 2, 12);
    if (predicted[0] < half_window + 1 || predicted[1] < half_window + 1 ||
        predicted[0] > width - half_window - 2 || predicted[1] > height - half_window - 2) {
      continue;  // Its window would leave the image.
    }
    seeds.push_back({predicted, half_window});
    seed_ids.push_back(id);
  }
  const auto refined = seeds.empty() ? std::vector<std::optional<Point2>>{} : refine(seeds);
  for (std::size_t i = 0; i < seeds.size() && i < refined.size(); ++i) {
    if (!refined[i]) continue;
    result.corners.push_back(
        {.id = seed_ids[i], .image = *refined[i], .board = board.corner(seed_ids[i])});
  }

  std::vector<Point2> board_xy;
  std::vector<Point2> image;
  for (const CharucoCorner& corner : result.corners) {
    board_xy.push_back({corner.board[0], corner.board[1]});
    image.push_back(corner.image);
  }
  if (intrinsics && board_xy.size() >= 4 &&
      spans_plane(board_xy, 0.5 * board.config().square_length)) {
    result.pose = solve_planar_pose(board_xy, image, *intrinsics);
  }
  result.quality = CharucoQuality{
      .corners_found = static_cast<int>(result.corners.size()),
      .corners_total = board.corner_count(),
      .markers_found = static_cast<int>(result.markers.size()),
      .markers_total = board.marker_count(),
      .coverage = hull_area(image) / (static_cast<double>(width) * height),
  };
  return result;
}

}  // namespace holoscan::fiducials
