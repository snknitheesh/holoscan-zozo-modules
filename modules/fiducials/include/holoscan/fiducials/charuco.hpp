// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A ChArUco board's geometry, as OpenCV 4.6 and later lays it out, and turning the markers
// found on it into chessboard-corner observations: what a camera calibration needs.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/fiducials/types.hpp>

namespace holoscan::fiducials {

// Most chessboard corners a board may have, (squares_x - 1) * (squares_y - 1).
inline constexpr std::size_t kMaxCharucoCorners = 1024;

struct CharucoBoardConfig {
  int squares_x{5};            // Squares along the board's x (its width), 2..64.
  int squares_y{7};            // Squares along its y (its height), 2..64.
  double square_length{0.04};  // Meters.
  double marker_length{0.03};  // Meters across a marker's black border; less than a square.
  std::string dictionary{"DICT_4X4_50"};
  // OpenCV's layout before 4.6 for an even number of rows: the top-left square holds a marker.
  bool legacy_pattern{false};
  int first_marker_id{0};  // Markers take the ids from this one up, row by row.
};

// The board frame: origin at the board's top-left corner as printed, x along the squares_x
// squares, y along the squares_y squares (down), z into the board. Chessboard corner k, of the
// (squares_x - 1) * (squares_y - 1) inner ones, is ((k % (squares_x - 1) + 1) * square_length,
// (k / (squares_x - 1) + 1) * square_length, 0), as in OpenCV. Square (x, y) holds a marker
// when x + y is odd (even, with legacy_pattern and an even squares_y).
class CharucoBoard {
 public:
  // @throws std::invalid_argument for an unusable configuration, such as a dictionary with too
  //         few markers.
  explicit CharucoBoard(CharucoBoardConfig config);

  [[nodiscard]] const CharucoBoardConfig& config() const noexcept { return config_; }
  [[nodiscard]] int marker_count() const noexcept { return static_cast<int>(squares_.size()); }
  [[nodiscard]] int corner_count() const noexcept;

  // The index (0 .. marker_count() - 1) of the marker with `id`, or nullopt off the board.
  [[nodiscard]] std::optional<int> marker_index(int id) const noexcept;
  // The square (x, y) marker `index` sits in.
  [[nodiscard]] std::array<int, 2> marker_square(int index) const;
  // Marker `index`'s corners in the board frame, in Marker::corners' order.
  [[nodiscard]] std::array<Point3, 4> marker_corners(int index) const;
  // Chessboard corner `id`'s position in the board frame.
  [[nodiscard]] Point3 corner(int id) const;
  // The markers (indices) in the squares that meet at chessboard corner `id`: always two.
  [[nodiscard]] std::span<const int> corner_markers(int id) const;
  // Whether square (x, y) holds a marker.
  [[nodiscard]] bool has_marker(int x, int y) const noexcept;

 private:
  CharucoBoardConfig config_;
  std::vector<std::array<int, 2>> squares_;    // Per marker index.
  std::vector<std::array<int, 2>> neighbors_;  // Per corner: its two markers.
};

// One chessboard corner seen in an image.
struct CharucoCorner {
  int id{};
  Point2 image{};  // Pixels, pixel centers at integers, sub-pixel.
  Point3 board{};  // In the board frame, meters.
};

// How much of the board, and of the image, one frame's view covers.
struct CharucoQuality {
  int corners_found{};
  int corners_total{};
  int markers_found{};
  int markers_total{};
  // The convex hull of the corners found, as a fraction of the image's area.
  double coverage{};
};

// One frame's view of a board: the observation a calibration takes.
struct CharucoObservation {
  std::vector<Marker> markers;         // The board's markers found, by id.
  std::vector<CharucoCorner> corners;  // By id.
  std::optional<Pose> pose;            // The board frame in the camera frame.
  CharucoQuality quality;
};

struct CharucoOptions {
  // A corner is reported only when this many of its two markers were found, 1..2 (OpenCV's
  // minMarkers).
  int min_markers{2};
};

// Where each seed's chessboard corner is; nullopt where there is none.
using Saddles = std::vector<std::optional<Point2>>;
// Moves each seed to the chessboard corner within its window.
using SaddleRefiner = std::function<Saddles(std::span<const SaddleSeed>)>;

// The observation of `board` in a width x height image whose markers are `markers`: each
// corner predicted from its markers, refined by `refine`, and with intrinsics, the board's pose.
[[nodiscard]] CharucoObservation observe_charuco(const CharucoBoard& board,
                                                 std::span<const Marker> markers,
                                                 const SaddleRefiner& refine, int width, int height,
                                                 const CharucoOptions& options,
                                                 const std::optional<CameraIntrinsics>& intrinsics);

}  // namespace holoscan::fiducials
