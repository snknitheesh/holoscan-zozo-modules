// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What the SDK-free tests share: running the cpu detector on a synthetic scene, and summing up
// how its markers compare with the scene's.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/fiducials/charuco.hpp>
#include <holoscan/fiducials/pose.hpp>
#include <holoscan/fiducials/synthetic_scene.hpp>

#include "cpu_detector.hpp"
#include "test_support.hpp"

namespace holoscan::fiducials::test {

inline GrayImageView host_view(const std::vector<std::uint8_t>& gray, int width, int height) {
  return {gray.data(), width, height, static_cast<std::size_t>(width), MemoryLocation::kHost};
}

// How a scene's markers were found, over any number of frames.
struct MarkerStats {
  int expected{};
  int missed{};
  int extra{};
  int corners{};
  double sum_corner_px{};
  double max_corner_px{};
  double max_translation{};  // Fraction of the distance.
  double max_rotation_deg{};

  void add(const SyntheticScene& scene, const std::vector<Marker>& found) {
    expected += static_cast<int>(scene.markers.size());
    const TruthCounts counts = match_markers(scene, found, [this](const MarkerMatch& match) {
      for (const double e : match.corner_errors) {
        ++corners;
        sum_corner_px += e;
        max_corner_px = std::max(max_corner_px, e);
      }
      if (match.translation_error) {
        max_translation = std::max(max_translation, *match.translation_error);
        max_rotation_deg = std::max(max_rotation_deg, *match.rotation_error_deg);
      }
    });
    missed += counts.missed;
    extra += counts.extra;
  }
  [[nodiscard]] double mean_corner_px() const { return corners ? sum_corner_px / corners : 0.0; }

  void print(const std::string& what) const {
    std::cout << "  " << what << ": " << expected - missed << '/' << expected << " found, extra "
              << extra << ", corner px mean " << mean_corner_px() << " max " << max_corner_px
              << ", translation max " << 100.0 * max_translation << "%, rotation max "
              << max_rotation_deg << " deg\n";
  }
};

// The cpu detector's markers in `scene`, with poses, with `options` (dictionary included).
inline std::vector<Marker> detect_scene(const SyntheticScene& scene,
                                        const DetectorOptions& options) {
  CpuDetector detector(options, scene.width, scene.height);
  const auto gray = render_scene(scene, 1);
  std::vector<Marker> markers = detector.detect(host_view(gray, scene.width, scene.height)).markers;
  const double length = scene.markers.empty() ? 0.1 : scene.markers.front().length;
  for (Marker& marker : markers) marker.pose = marker_pose(marker, scene.intrinsics, length);
  return markers;
}

// The same with the default options and the scene's dictionary.
inline std::vector<Marker> detect_scene(const SyntheticScene& scene) {
  DetectorOptions options;
  options.dictionary = scene.dictionary;
  return detect_scene(scene, options);
}

// The cpu detector's view of `scene`'s board.
inline CharucoObservation observe_scene(const SyntheticScene& scene, const CharucoBoard& board,
                                        CharucoOptions charuco = {}, DetectorOptions options = {}) {
  options.dictionary = board.config().dictionary;
  CpuDetector detector(options, scene.width, scene.height);
  const auto gray = render_scene(scene, 1);
  const GrayImageView view = host_view(gray, scene.width, scene.height);
  const FrameMarkers found = detector.detect(view);
  return observe_charuco(
      board, found.markers,
      [&](std::span<const SaddleSeed> seeds) { return detector.refine_saddles(view, seeds); },
      scene.width, scene.height, charuco, scene.intrinsics);
}

// How a board's corners compare with the scene's.
struct CornerStats {
  int expected{};
  int found{};
  int wrong{};  // Found with the wrong board position, or more than 0.5 px from the truth.
  double sum_px{};
  double max_px{};
  double max_translation{};
  double max_rotation_deg{};
  int poses{};

  void add(const SyntheticScene& scene, const CharucoObservation& view) {
    const std::vector<CharucoCorner> truth = board_truth(scene);
    expected += static_cast<int>(truth.size());
    for (const CharucoCorner& corner : view.corners) {
      const auto it = std::find_if(truth.begin(), truth.end(),
                                   [&](const CharucoCorner& t) { return t.id == corner.id; });
      if (it == truth.end()) {
        ++wrong;
        continue;
      }
      const double e = std::hypot(corner.image[0] - it->image[0], corner.image[1] - it->image[1]);
      ++found;
      sum_px += e;
      max_px = std::max(max_px, e);
      if (e > 0.5 || corner.board != it->board) ++wrong;
    }
    if (view.pose) {
      ++poses;
      const auto& t = scene.board->pose.translation;
      const auto& p = view.pose->translation;
      max_translation =
          std::max(max_translation, std::hypot(p[0] - t[0], p[1] - t[1], p[2] - t[2]) /
                                        std::hypot(t[0], t[1], t[2]));
      max_rotation_deg = std::max(
          max_rotation_deg, rotation_error_deg(view.pose->rotation, scene.board->pose.rotation));
    }
  }
  [[nodiscard]] double mean_px() const { return found ? sum_px / found : 0.0; }

  void print(const std::string& what) const {
    std::cout << "  " << what << ": " << found << '/' << expected << " corners, wrong " << wrong
              << ", px mean " << mean_px() << " max " << max_px << ", pose translation max "
              << 100.0 * max_translation << "%, rotation max " << max_rotation_deg << " deg\n";
  }
};

}  // namespace holoscan::fiducials::test
