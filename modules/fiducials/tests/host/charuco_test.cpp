// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ChArUco boards on the cpu detector against ground truth: the board's geometry against
// OpenCV's, every corner's id, board position and sub-pixel image position, the board's pose
// and the quality summary; the legacy layout and first marker id; blur, noise, occlusion and the
// image's border; min_markers; and bad configurations.

#include <algorithm>
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "host_support.hpp"

namespace holoscan::fiducials {
namespace {

using test::CornerStats;

constexpr int kWidth = 1280;
constexpr int kHeight = 720;

std::vector<std::array<int, 2>> marker_squares(const CharucoBoard& board) {
  std::vector<std::array<int, 2>> squares;
  for (int i = 0; i < board.marker_count(); ++i) squares.push_back(board.marker_square(i));
  return squares;
}

// As OpenCV 4.12's CharucoBoard lays them out.
void check_geometry() {
  CharucoBoardConfig config{
      .squares_x = 5, .squares_y = 4, .square_length = 0.04, .marker_length = 0.03};
  const CharucoBoard board(config);
  VERIFY(board.marker_count() == 10 && board.corner_count() == 12);
  VERIFY((marker_squares(board) ==
          std::vector<std::array<int, 2>>{
              {1, 0}, {3, 0}, {0, 1}, {2, 1}, {4, 1}, {1, 2}, {3, 2}, {0, 3}, {2, 3}, {4, 3}}));
  const auto m0 = board.marker_corners(0);
  VERIFY(std::abs(m0[0][0] - 0.045) < 1e-12 && std::abs(m0[0][1] - 0.005) < 1e-12);
  VERIFY(std::abs(m0[2][0] - 0.075) < 1e-12 && std::abs(m0[2][1] - 0.035) < 1e-12);
  VERIFY(board.corner(0) == (Point3{0.04, 0.04, 0.0}));
  VERIFY(board.corner(5) == (Point3{0.08, 0.08, 0.0}));
  // Corner 0 lies between squares (0, 0) black, (1, 0) and (0, 1) markers 0 and 2, (1, 1) black.
  const auto pair = board.corner_markers(0);
  VERIFY(pair.size() == 2 && pair[0] == 0 && pair[1] == 2);
  config.legacy_pattern = true;
  VERIFY((marker_squares(CharucoBoard(config)) ==
          std::vector<std::array<int, 2>>{
              {0, 0}, {2, 0}, {4, 0}, {1, 1}, {3, 1}, {0, 2}, {2, 2}, {4, 2}, {1, 3}, {3, 3}}));
  // With an odd number of rows the legacy layout is the same as today's.
  config.squares_y = 3;
  const auto legacy_odd = marker_squares(CharucoBoard(config));
  config.legacy_pattern = false;
  VERIFY(legacy_odd == marker_squares(CharucoBoard(config)));
  config.first_marker_id = 10;
  const CharucoBoard offset(config);
  VERIFY(!offset.marker_index(9) && offset.marker_index(10) == 0 && offset.marker_index(16) == 6 &&
         !offset.marker_index(17));
}

void check_board(const CharucoBoardConfig& config, const std::string& what, double max_mean_px) {
  const CharucoBoard board(config);
  CornerStats stats;
  for (std::uint32_t frame = 0; frame < 6; ++frame) {
    const SyntheticScene scene = make_board_scene(config, frame * 9, kWidth, kHeight);
    const CharucoObservation view = test::observe_scene(scene, board);
    stats.add(scene, view);
    VERIFY(view.quality.corners_total == board.corner_count());
    VERIFY(view.quality.markers_total == board.marker_count());
    VERIFY(view.quality.markers_found == board.marker_count());
    VERIFY(view.quality.corners_found == static_cast<int>(view.corners.size()));
    VERIFY(view.quality.coverage > 0.02 && view.quality.coverage < 0.5);
    VERIFY(std::is_sorted(view.corners.begin(), view.corners.end(),
                          [](const auto& a, const auto& b) { return a.id < b.id; }));
    VERIFY(view.pose && view.pose->reprojection_error < 0.2);
  }
  stats.print(what);
  VERIFY_MSG(stats.found == stats.expected && stats.wrong == 0, what + ": every corner, right");
  VERIFY_MSG(stats.mean_px() < max_mean_px && stats.max_px < 0.3, what + ": sub-pixel corners");
  VERIFY_MSG(stats.max_translation < 0.003 && stats.max_rotation_deg < 0.3, what + ": poses");
}

void check_boards() {
  check_board({}, "5x7 DICT_4X4_50", 0.06);
  check_board({.squares_x = 8,
               .squares_y = 6,
               .square_length = 0.03,
               .marker_length = 0.022,
               .dictionary = "DICT_5X5_100"},
              "8x6 DICT_5X5_100", 0.06);
  check_board({.squares_x = 7,
               .squares_y = 5,
               .square_length = 0.04,
               .marker_length = 0.03,
               .dictionary = "DICT_6X6_250",
               .legacy_pattern = false,
               .first_marker_id = 30},
              "7x5 DICT_6X6_250 from id 30", 0.06);
  check_board({.squares_x = 6,
               .squares_y = 8,
               .square_length = 0.035,
               .marker_length = 0.026,
               .dictionary = "DICT_4X4_100",
               .legacy_pattern = true},
              "6x8 DICT_4X4_100 legacy", 0.06);
}

// A board's markers with other ids, or in the other layout, give no corners.
void check_other_boards() {
  const CharucoBoardConfig printed{.first_marker_id = 20};
  const SyntheticScene scene = make_board_scene(printed, 3, kWidth, kHeight);
  const CharucoObservation view = test::observe_scene(scene, CharucoBoard(CharucoBoardConfig{}));
  VERIFY(view.corners.empty() && view.markers.empty() && !view.pose);
  VERIFY(view.quality.coverage == 0.0);

  CharucoBoardConfig legacy{.squares_x = 6,
                            .squares_y = 8,
                            .square_length = 0.035,
                            .marker_length = 0.026,
                            .legacy_pattern = true};
  const SyntheticScene legacy_scene = make_board_scene(legacy, 3, kWidth, kHeight);
  legacy.legacy_pattern = false;
  CornerStats stats;
  stats.add(legacy_scene, test::observe_scene(legacy_scene, CharucoBoard(legacy)));
  VERIFY_MSG(stats.wrong == 0, "a legacy board read as a current one gives no wrong corners");
}

void check_blur_and_noise() {
  const CharucoBoardConfig config{.squares_x = 7, .squares_y = 5};
  const CharucoBoard board(config);
  CornerStats stats;
  for (std::uint32_t frame = 0; frame < 6; ++frame) {
    SyntheticScene scene = make_board_scene(config, frame * 7, kWidth, kHeight);
    scene.blur_sigma = 1.0;
    scene.noise_sigma = 5.0;
    scene.seed = frame + 11;
    stats.add(scene, test::observe_scene(scene, board));
  }
  stats.print("blur 1 noise 5");
  VERIFY(stats.found == stats.expected && stats.wrong == 0);
  VERIFY(stats.mean_px() < 0.1 && stats.max_px < 0.4);
  VERIFY(stats.max_translation < 0.005 && stats.max_rotation_deg < 0.5);
}

// Corners under an occluder are missing, the rest found, none wrong; with min_markers 1, also
// the corners next to a covered marker.
void check_occlusion() {
  const CharucoBoardConfig config{};
  const CharucoBoard board(config);
  for (const int min_markers : {2, 1}) {
    CornerStats stats;
    int found = 0;
    for (std::uint32_t frame = 0; frame < 4; ++frame) {
      SyntheticScene scene = make_board_scene(config, frame * 10, kWidth, kHeight);
      const auto truth = board_truth(scene);
      // A band across the board's middle, about a sixth of its height.
      const Point2 a = truth[(board.corner_count() / 2) - 2].image;
      const Point2 b = truth[(board.corner_count() / 2) + 1].image;
      scene.occluders.push_back(Occluder{{Point2{0, a[1] - 25}, Point2{kWidth, b[1] - 25},
                                          Point2{kWidth, b[1] + 25}, Point2{0, a[1] + 25}},
                                         140});
      const CharucoObservation view =
          test::observe_scene(scene, board, CharucoOptions{.min_markers = min_markers});
      stats.add(scene, view);
      found += static_cast<int>(view.corners.size());
    }
    stats.print("occluded, min_markers " + std::to_string(min_markers));
    VERIFY_MSG(stats.wrong == 0, "no wrong corner under occlusion");
    VERIFY_MSG(found > stats.expected / 2, "the uncovered corners are found");
  }
}

// A board running off the image: the corners inside it are found, none wrong.
void check_cut_board() {
  const CharucoBoardConfig config{
      .squares_x = 8, .squares_y = 6, .square_length = 0.04, .marker_length = 0.03};
  const CharucoBoard board(config);
  SyntheticScene scene = make_board_scene(config, 2, kWidth, kHeight);
  scene.board->pose.translation[0] += 0.5 * scene.board->pose.translation[2];  // To the right.
  CornerStats stats;
  const CharucoObservation view = test::observe_scene(scene, board);
  stats.add(scene, view);
  stats.print("cut by the image's edge");
  VERIFY(stats.wrong == 0);
  VERIFY(stats.found > 0 && stats.found < board.corner_count());
}

void check_bad_configs() {
  for (const CharucoBoardConfig& bad :
       {CharucoBoardConfig{.squares_x = 1}, CharucoBoardConfig{.squares_y = 65},
        CharucoBoardConfig{.square_length = 0.0},
        CharucoBoardConfig{.square_length = 0.04, .marker_length = 0.04},
        CharucoBoardConfig{.dictionary = "DICT_4X4_51"},
        CharucoBoardConfig{.squares_x = 12, .squares_y = 12, .dictionary = "DICT_4X4_50"},
        CharucoBoardConfig{.first_marker_id = 40}, CharucoBoardConfig{.first_marker_id = -1}}) {
    VERIFY(test::throws<std::invalid_argument>([&] { CharucoBoard board(bad); }));
  }
  const CharucoBoard board(CharucoBoardConfig{});
  VERIFY(test::throws<std::invalid_argument>([&] {
    (void)observe_charuco(
        board, {}, [](std::span<const SaddleSeed>) { return std::vector<std::optional<Point2>>{}; },
        640, 480, CharucoOptions{.min_markers = 3}, std::nullopt);
  }));
}

}  // namespace
}  // namespace holoscan::fiducials

int main() {
  namespace f = holoscan::fiducials;
  return f::test::run({
      {"board geometry as OpenCV's", f::check_geometry},
      {"boards: corners, poses, quality", f::check_boards},
      {"other boards give no corners", f::check_other_boards},
      {"blur and noise", f::check_blur_and_noise},
      {"occlusion and min_markers", f::check_occlusion},
      {"a board cut by the image's edge", f::check_cut_board},
      {"bad configurations", f::check_bad_configs},
  });
}
