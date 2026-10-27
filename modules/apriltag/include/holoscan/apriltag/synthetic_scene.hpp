// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A deterministic scene of tags at known poses, rendered without external data,
// and the scorer that checks detections against it.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/apriltag/detection.hpp>
#include <holoscan/apriltag/pose.hpp>

namespace holoscan::apriltag {

struct SceneTag {
  int id{};
  TagPose pose;  // `error` unused.
};

struct SyntheticScene {
  int width{};
  int height{};
  CameraIntrinsics intrinsics;
  std::string family;
  double tag_size{};  // Meters across the black border.
  std::vector<SceneTag> tags;
};

// Four tags moving smoothly with `frame`: they tilt by up to 35 degrees, sit at
// 0, 90, 180 and 270 degrees in plane (plus up to 30 degrees either way), and
// move between about 0.9 and 1.3 m from the camera.
// @throws std::invalid_argument for an unknown family or a size below 320x240.
[[nodiscard]] SyntheticScene make_scene(const std::string& family, std::uint32_t frame, int width,
                                        int height);

// The scene as gray (1 channel) or RGB (3 channels) pixels, rows packed, over
// a lit, slightly noisy background. Edges are antialiased by supersampling.
[[nodiscard]] std::vector<std::uint8_t> render_scene(const SyntheticScene& scene, int channels);

// Where a tag's corners project, in Detection::corners' order.
[[nodiscard]] std::array<Point2, 4> project_corners(const SceneTag& tag,
                                                    const CameraIntrinsics& intrinsics,
                                                    double tag_size);

// One detection of a scene tag: its corners' errors in pixels and, with a pose,
// its translation error as a fraction of the tag's distance.
struct TruthMatch {
  const Detection& detection;
  const SceneTag& tag;
  std::array<double, 4> corner_errors;
  std::optional<double> translation_error;
};

struct TruthCounts {
  int extra{};   // Detections of no scene tag; a second report of one tag counts too.
  int missed{};  // Scene tags not detected.
};

// Matches `detections` to `scene`'s tags by id, calling `on_match` for each match.
TruthCounts match_truth(const SyntheticScene& scene, std::span<const Detection> detections,
                        const std::function<void(const TruthMatch&)>& on_match);

}  // namespace holoscan::apriltag
