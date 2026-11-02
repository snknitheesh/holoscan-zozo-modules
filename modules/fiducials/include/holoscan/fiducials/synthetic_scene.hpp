// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Deterministic scenes of markers and ChArUco boards at known poses, rendered without external
// data, with blur, noise, occluders and mirroring, and their ground truth: for tests and demos.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/fiducials/charuco.hpp>
#include <holoscan/fiducials/types.hpp>

namespace holoscan::fiducials {

// A printed marker: `length` meters across its black border, with one cell of white around it.
struct SceneMarker {
  int id{};
  double length{};
  Pose pose;  // Its frame, centered on it; reprojection_error unused.
};

// A printed ChArUco board, with half a square of white around it.
struct SceneBoard {
  CharucoBoardConfig config;
  Pose pose;  // Its frame (CharucoBoard's); reprojection_error unused.
};

// A flat gray quadrilateral in front of everything, in pixels.
struct Occluder {
  std::array<Point2, 4> corners;
  std::uint8_t gray{128};
};

struct SyntheticScene {
  int width{};
  int height{};
  CameraIntrinsics intrinsics;  // No distortion.
  std::string dictionary;
  std::vector<SceneMarker> markers;
  std::optional<SceneBoard> board;
  std::vector<Occluder> occluders;
  double blur_sigma{};    // Gaussian blur, pixels.
  double noise_sigma{};   // Gaussian noise, gray levels.
  std::uint32_t seed{1};  // For the noise.
  bool mirrored{};        // Flipped left to right after rendering; the truth is not.
};

// Six markers of `dictionary` moving smoothly with `frame`: tilted up to 40 degrees, turned to
// every quarter in plane (plus up to 30 degrees), 0.8 to 1.2 m away; ids spread over the
// dictionary. Light noise, no blur.
// @throws std::invalid_argument for an unknown dictionary or a size below 320x240.
[[nodiscard]] SyntheticScene make_marker_scene(const std::string& dictionary, std::uint32_t frame,
                                               int width, int height);

// `board` moving smoothly with `frame`, whole in view: tilted up to 35 degrees, turned up to 25
// degrees in plane, filling about half the image's height. Light noise, no blur.
// @throws std::invalid_argument for an unusable board or a size below 320x240.
[[nodiscard]] SyntheticScene make_board_scene(const CharucoBoardConfig& board, std::uint32_t frame,
                                              int width, int height);

// The scene as gray (1 channel) or RGB (3 channels) pixels, rows packed, over a lit, noisy
// background; edges antialiased by supersampling.
// @throws std::invalid_argument for another channel count.
[[nodiscard]] std::vector<std::uint8_t> render_scene(const SyntheticScene& scene, int channels);

// Where a marker's corners appear, in Marker::corners' order (before mirroring).
[[nodiscard]] std::array<Point2, 4> marker_truth(const SyntheticScene& scene,
                                                 const SceneMarker& marker);

// Where each of the board's chessboard corners appears (before mirroring), by id; corners
// outside the image or under an occluder are left out.
[[nodiscard]] std::vector<CharucoCorner> board_truth(const SyntheticScene& scene);

// One detected marker matched to the scene's: its corner errors in pixels and, with a pose, its
// translation error as a fraction of its distance and its rotation error in degrees.
struct MarkerMatch {
  const Marker& found;
  const SceneMarker& truth;
  std::array<double, 4> corner_errors;
  std::optional<double> translation_error;
  std::optional<double> rotation_error_deg;
};

struct TruthCounts {
  int extra{};   // Detections of no scene marker; a second report of one marker counts too.
  int missed{};  // Scene markers not detected.
};

// Matches `found` to the scene's markers by id, calling `on_match` for each match.
TruthCounts match_markers(const SyntheticScene& scene, std::span<const Marker> found,
                          const std::function<void(const MarkerMatch&)>& on_match);

// The angle between two rotations, in degrees.
[[nodiscard]] double rotation_error_deg(const std::array<double, 9>& a,
                                        const std::array<double, 9>& b);

}  // namespace holoscan::fiducials
