// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The cpu detector on synthetic markers with ground truth: every dictionary, every quarter turn
// and tilts to 40 degrees; corners and poses; blur, noise and partial occlusion; false positives
// on clutter; markers at and across the image's border; a mirrored image; the wrong dictionary;
// and the options.

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/fiducials/dictionary.hpp>

#include "host_support.hpp"

namespace holoscan::fiducials {
namespace {

using test::MarkerStats;

constexpr int kWidth = 1280;
constexpr int kHeight = 720;

void check_dictionaries() {
  for (const char* name :
       {"DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250", "DICT_7X7_1000", "DICT_ARUCO_ORIGINAL",
        "DICT_ARUCO_MIP_36h12", "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10",
        "DICT_APRILTAG_36h11"}) {
    MarkerStats stats;
    for (std::uint32_t frame = 0; frame < 6; ++frame) {
      const SyntheticScene scene = make_marker_scene(name, frame * 7, kWidth, kHeight);
      stats.add(scene, test::detect_scene(scene));
    }
    stats.print(name);
    VERIFY_MSG(stats.missed == 0 && stats.extra == 0, std::string(name) + ": every marker, no other");
    VERIFY_MSG(stats.mean_corner_px() < 0.06 && stats.max_corner_px < 0.3,
               std::string(name) + ": sub-pixel corners");
    VERIFY_MSG(stats.max_translation < 0.005 && stats.max_rotation_deg < 1.0,
               std::string(name) + ": poses");
  }
}

void check_blur_and_noise() {
  for (const auto& [blur, noise] : {std::array{1.0, 6.0}, std::array{2.0, 4.0}}) {
    MarkerStats stats;
    for (std::uint32_t frame = 0; frame < 6; ++frame) {
      SyntheticScene scene = make_marker_scene("DICT_6X6_250", frame * 11, kWidth, kHeight);
      scene.blur_sigma = blur;
      scene.noise_sigma = noise;
      scene.seed = frame + 3;
      stats.add(scene, test::detect_scene(scene));
    }
    stats.print("blur " + std::to_string(blur) + " noise " + std::to_string(noise));
    VERIFY(stats.missed == 0 && stats.extra == 0);
    VERIFY(stats.mean_corner_px() < 0.15 && stats.max_corner_px < 0.5);
    VERIFY(stats.max_translation < 0.01 && stats.max_rotation_deg < 2.0);
  }
}

// A quad over part of a marker: across one of its corners, or across its middle.
Occluder cover(const SyntheticScene& scene, const SceneMarker& marker, bool corner) {
  const auto c = marker_truth(scene, marker);
  const auto mix = [&](const Point2& a, const Point2& b, double t) {
    return Point2{a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])};
  };
  const Point2 center{0.25 * (c[0][0] + c[1][0] + c[2][0] + c[3][0]),
                      0.25 * (c[0][1] + c[1][1] + c[2][1] + c[3][1])};
  if (corner) {
    // The top-left corner and about a third of the marker around it, and beyond it.
    const Point2 out = mix(center, c[0], 1.3);
    return Occluder{{out, mix(c[0], c[1], 0.45), mix(c[0], c[2], 0.45), mix(c[0], c[3], 0.45)},
                    150};
  }
  // A band across the middle, clear of the border.
  return Occluder{{mix(c[0], c[3], 0.35), mix(c[1], c[2], 0.35), mix(c[1], c[2], 0.6),
                   mix(c[0], c[3], 0.6)},
                  128};
}

// A partly covered marker is missed or found with its own id, never taken for another; the
// others are found.
void check_partial_occlusion() {
  for (const char* name : {"DICT_4X4_50", "DICT_6X6_250", "DICT_7X7_50"}) {
    int found_covered = 0;
    int extra = 0;
    int missed_clear = 0;
    for (std::uint32_t frame = 0; frame < 6; ++frame) {
      SyntheticScene scene = make_marker_scene(name, frame * 5, kWidth, kHeight);
      scene.occluders.push_back(cover(scene, scene.markers[0], true));
      scene.occluders.push_back(cover(scene, scene.markers[1], false));
      const std::vector<Marker> found = test::detect_scene(scene);
      std::set<int> ids;
      for (const Marker& m : found) ids.insert(m.id);
      for (std::size_t i = 0; i < scene.markers.size(); ++i) {
        const bool seen = ids.count(scene.markers[i].id) != 0;
        if (i < 2) {
          found_covered += seen;
        } else {
          missed_clear += !seen;
        }
      }
      for (const Marker& m : found) {
        bool known = false;
        for (const SceneMarker& s : scene.markers) known = known || s.id == m.id;
        extra += !known;
      }
    }
    std::cout << "  " << name << ": covered markers found " << found_covered << "/12, clear missed "
              << missed_clear << ", wrong ids " << extra << '\n';
    VERIFY_MSG(extra == 0, std::string(name) + ": no covered marker taken for another");
    VERIFY_MSG(missed_clear == 0, std::string(name) + ": the uncovered markers are found");
  }
}

// Clutter with no markers: dark and bright quads, checkered patches, stripes, blur, noise.
std::vector<std::uint8_t> clutter(std::uint32_t seed) {
  std::mt19937 random(seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  SyntheticScene scene;
  scene.width = kWidth;
  scene.height = kHeight;
  scene.intrinsics = {1000, 1000, 640, 360};
  scene.noise_sigma = 3.0;
  scene.blur_sigma = 0.7;
  scene.seed = seed;
  for (int i = 0; i < 120; ++i) {
    const double cx = unit(random) * kWidth;
    const double cy = unit(random) * kHeight;
    const double r = 8.0 + unit(random) * 70.0;
    const double a = unit(random) * 6.283;
    Occluder quad;
    for (int k = 0; k < 4; ++k) {
      const double angle = a + k * 1.5708 + 0.3 * (unit(random) - 0.5);
      const double radius = r * (0.6 + 0.4 * unit(random));
      quad.corners[k] = {cx + radius * std::cos(angle), cy + radius * std::sin(angle)};
    }
    quad.gray = static_cast<std::uint8_t>(unit(random) * 255.0);
    scene.occluders.push_back(quad);
  }
  std::vector<std::uint8_t> gray = render_scene(scene, 1);
  // Checkered patches and stripes. (Not framed in black: a framed pattern can be a marker, as
  // stripes are in DICT_ARUCO_ORIGINAL.)
  for (int i = 0; i < 40; ++i) {
    const int cell = 3 + static_cast<int>(unit(random) * 12);
    const int cells = 2 + static_cast<int>(unit(random) * 8);
    const int x0 = static_cast<int>(unit(random) * (kWidth - cell * cells));
    const int y0 = static_cast<int>(unit(random) * (kHeight - cell * cells));
    const bool stripes = unit(random) < 0.3;
    for (int y = 0; y < cell * cells; ++y) {
      for (int x = 0; x < cell * cells; ++x) {
        const bool dark = stripes ? (x / cell) % 2 == 0 : (x / cell + y / cell) % 2 == 0;
        gray[static_cast<std::size_t>(y0 + y) * kWidth + x0 + x] = dark ? 30 : 220;
      }
    }
  }
  return gray;
}

void check_false_positives() {
  int images = 0;
  int detections = 0;
  for (const char* name : {"DICT_4X4_50", "DICT_4X4_1000", "DICT_5X5_1000", "DICT_6X6_1000",
                           "DICT_7X7_1000", "DICT_ARUCO_ORIGINAL", "DICT_ARUCO_MIP_36h12",
                           "DICT_APRILTAG_16h5", "DICT_APRILTAG_36h11"}) {
    DetectorOptions options;
    options.dictionary = name;
    CpuDetector detector(options, kWidth, kHeight);
    for (std::uint32_t seed = 1; seed <= 6; ++seed) {
      const auto gray = clutter(seed);
      const auto found = detector.detect(test::host_view(gray, kWidth, kHeight)).markers;
      ++images;
      detections += static_cast<int>(found.size());
      for (const Marker& m : found) {
        std::cout << "  false positive: " << name << " id " << m.id << " at " << m.corners[0][0]
                  << ", " << m.corners[0][1] << " (clutter " << seed << ")\n";
      }
    }
  }
  std::cout << "  " << detections << " false positives in " << images << " clutter images\n";
  VERIFY(detections == 0);
}

// A frontal marker `length` meters across, 1 m away, centered at pixel (u, v).
SceneMarker frontal(const SyntheticScene& scene, int id, double u, double v) {
  SceneMarker marker{.id = id, .length = 0.1, .pose = {}};
  marker.pose.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  marker.pose.translation = {(u - scene.intrinsics.cx) / scene.intrinsics.fx,
                             (v - scene.intrinsics.cy) / scene.intrinsics.fy, 1.0};
  return marker;
}

// Markers close to the image's edges are found; one the edge cuts is not.
void check_border() {
  SyntheticScene scene = make_marker_scene("DICT_5X5_100", 0, kWidth, kHeight);
  scene.markers.clear();
  const double half = 0.5 * scene.intrinsics.fx * 0.1;  // Pixels from a marker's center to a side.
  scene.markers.push_back(frontal(scene, 3, 3.0 + half, 300));                 // 3 px from left.
  scene.markers.push_back(frontal(scene, 4, 640, kHeight - 1 - 3.0 - half));  // 3 px from bottom.
  scene.markers.push_back(frontal(scene, 5, kWidth - 1, 300));                 // Cut in half.
  scene.markers.push_back(frontal(scene, 6, 400, -0.3 * half));                // Mostly cut.
  const std::vector<Marker> found = test::detect_scene(scene);
  std::set<int> ids;
  for (const Marker& m : found) ids.insert(m.id);
  VERIFY_MSG(ids == (std::set<int>{3, 4}), "the markers inside the image, and only those");
  MarkerStats stats;
  SyntheticScene inside = scene;
  inside.markers.resize(2);
  stats.add(inside, found);
  stats.print("at the border");
  VERIFY(stats.max_corner_px < 0.3);
}

// The dictionary marker a mirrored marker reads as, if any: the marker's cells flipped left to
// right, looked up in every turn within the detector's correction.
std::optional<int> mirrored_id(const Dictionary& d, int id, int max_hamming) {
  const int n = d.marker_size();
  const std::uint64_t code = d.code(id);
  std::uint64_t mirrored = 0;
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      const std::uint64_t bit = (code >> (n * n - 1 - (row * n + (n - 1 - col)))) & 1U;
      mirrored |= bit << (n * n - 1 - (row * n + col));
    }
  }
  int best = n * n + 1;
  int best_id = -1;
  for (int other = 0; other < d.count(); ++other) {
    std::uint64_t c = d.code(other);
    for (int r = 0; r < 4; ++r) {
      const int distance = __builtin_popcountll(c ^ mirrored);
      if (distance < best) {
        best = distance;
        best_id = other;
      }
      c = rotate_code(c, n);
    }
  }
  if (best > max_hamming) return std::nullopt;
  return best_id;
}

// In a mirrored image a marker reads as its cells flipped: found only as the dictionary marker
// those match, if any, never as itself.
void check_mirrored() {
  for (const char* name : {"DICT_4X4_50", "DICT_5X5_1000", "DICT_6X6_250", "DICT_APRILTAG_36h11"}) {
    const Dictionary d(name);
    const int max_hamming = static_cast<int>(d.max_correction_bits() * 0.6);
    int matches = 0;
    for (std::uint32_t frame = 0; frame < 4; ++frame) {
      SyntheticScene scene = make_marker_scene(name, frame * 9, kWidth, kHeight);
      scene.mirrored = true;
      std::multiset<int> expected;
      for (const SceneMarker& m : scene.markers) {
        if (const auto id = mirrored_id(d, m.id, max_hamming)) expected.insert(*id);
      }
      std::multiset<int> found;
      for (const Marker& m : test::detect_scene(scene)) found.insert(m.id);
      matches += found == expected;
      if (found != expected) {
        std::cout << "  " << name << " frame " << frame << ": mirrored found " << found.size()
                  << ", expected " << expected.size() << '\n';
      }
    }
    VERIFY_MSG(matches == 4, std::string(name) + ": a mirrored image reads as predicted");
  }
}

// Markers of another dictionary are not reported.
void check_wrong_dictionary() {
  for (const auto& [scene_dictionary, detector_dictionary] :
       {std::array<const char*, 2>{"DICT_6X6_250", "DICT_4X4_50"},
        {"DICT_4X4_50", "DICT_6X6_250"},
        {"DICT_APRILTAG_36h11", "DICT_6X6_1000"},
        {"DICT_5X5_100", "DICT_ARUCO_ORIGINAL"},
        {"DICT_7X7_250", "DICT_5X5_1000"}}) {
    int found = 0;
    for (std::uint32_t frame = 0; frame < 4; ++frame) {
      const SyntheticScene scene = make_marker_scene(scene_dictionary, frame * 13, kWidth, kHeight);
      DetectorOptions options;
      options.dictionary = detector_dictionary;
      found += static_cast<int>(test::detect_scene(scene, options).size());
    }
    VERIFY_MSG(found == 0, std::string(scene_dictionary) + " markers read with " +
                               detector_dictionary + ": " + std::to_string(found) + " found");
  }
}

void check_options() {
  const SyntheticScene scene = make_marker_scene("DICT_6X6_250", 4, kWidth, kHeight);
  const DetectorOptions defaults{.dictionary = scene.dictionary};
  // Without refinement the corners are where the border's outermost dark pixels are.
  {
    DetectorOptions options = defaults;
    options.refine_corners = false;
    MarkerStats stats;
    stats.add(scene, test::detect_scene(scene, options));
    stats.print("unrefined corners");
    VERIFY(stats.missed == 0 && stats.extra == 0 && stats.max_corner_px < 1.5);
  }
  // Without correction, only exact codes; larger tiles and markers still found.
  {
    DetectorOptions options = defaults;
    options.error_correction_rate = 0.0;
    options.tile_size = 16;
    options.min_marker_px = 40;
    MarkerStats stats;
    const std::vector<Marker> found = test::detect_scene(scene, options);
    stats.add(scene, found);
    VERIFY(stats.missed == 0 && stats.extra == 0);
    for (const Marker& m : found) VERIFY(m.hamming == 0);
  }
  // Markers smaller than min_marker_px are not reported.
  {
    DetectorOptions options = defaults;
    options.min_marker_px = 400;
    VERIFY(test::detect_scene(scene, options).empty());
  }
  for (const auto& bad : std::initializer_list<void (*)(DetectorOptions&)>{
           [](DetectorOptions& o) { o.dictionary = "DICT_8X8_50"; },
           [](DetectorOptions& o) { o.error_correction_rate = 1.5; },
           [](DetectorOptions& o) { o.tile_size = 1; },
           [](DetectorOptions& o) { o.min_contrast = 0; },
           [](DetectorOptions& o) { o.min_marker_px = 2; }}) {
    DetectorOptions options;
    bad(options);
    VERIFY(test::throws<std::invalid_argument>([&] { validate_options(options); }));
  }
  VERIFY(test::throws<std::invalid_argument>([] { CpuDetector d(DetectorOptions{}, 8, 720); }));
  CpuDetector detector(DetectorOptions{}, 640, 480);
  std::vector<std::uint8_t> other(320 * 240);
  VERIFY(test::throws<std::invalid_argument>(
      [&] { (void)detector.detect(test::host_view(other, 320, 240)); }));
}

void report_timing() {
  const SyntheticScene scene = make_marker_scene("DICT_4X4_50", 0, kWidth, kHeight);
  const auto gray = render_scene(scene, 1);
  CpuDetector detector(DetectorOptions{}, kWidth, kHeight);
  (void)detector.detect(test::host_view(gray, kWidth, kHeight));
  constexpr int kRuns = 10;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kRuns; ++i) (void)detector.detect(test::host_view(gray, kWidth, kHeight));
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  std::cout << "  cpu detection at 1280x720, six markers: " << ms / kRuns << " ms\n";
}

}  // namespace
}  // namespace holoscan::fiducials

int main() {
  namespace f = holoscan::fiducials;
  return f::test::run({
      {"every dictionary, corners and poses", f::check_dictionaries},
      {"blur and noise", f::check_blur_and_noise},
      {"partial occlusion", f::check_partial_occlusion},
      {"no false positives on clutter", f::check_false_positives},
      {"markers at the image border", f::check_border},
      {"a mirrored image", f::check_mirrored},
      {"the wrong dictionary", f::check_wrong_dictionary},
      {"options", f::check_options},
      {"timing", f::report_timing},
  });
}
