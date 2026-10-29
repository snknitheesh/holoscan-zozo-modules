// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The predefined dictionaries: every name, size and correction, codes against values from
// OpenCV, the smaller ArUco dictionaries as prefixes of the 1000-marker ones, no two markers
// alike in any rotation, rotation itself, the printed cells, and the errors.

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/fiducials/dictionary.hpp>

#include "test_support.hpp"

namespace holoscan::fiducials {
namespace {

void check_names() {
  const std::vector<std::string_view> names = dictionary_names();
  VERIFY(names.size() == 22);
  struct Expected {
    const char* name;
    int size;
    int count;
    int correction;
  };
  // Sizes and counts as OpenCV defines them; corrections are OpenCV's maxCorrectionBits, and
  // (d - 1) / 2 for the AprilTag families' distance d.
  for (const Expected& e : std::initializer_list<Expected>{{"DICT_4X4_50", 4, 50, 1},
                                                           {"DICT_4X4_1000", 4, 1000, 0},
                                                           {"DICT_5X5_100", 5, 100, 3},
                                                           {"DICT_6X6_250", 6, 250, 5},
                                                           {"DICT_7X7_1000", 7, 1000, 6},
                                                           {"DICT_ARUCO_ORIGINAL", 5, 1024, 0},
                                                           {"DICT_ARUCO_MIP_36h12", 6, 250, 12},
                                                           {"DICT_APRILTAG_16h5", 4, 30, 2},
                                                           {"DICT_APRILTAG_25h9", 5, 35, 4},
                                                           {"DICT_APRILTAG_36h10", 6, 2320, 4},
                                                           {"DICT_APRILTAG_36h11", 6, 587, 5}}) {
    const Dictionary d(e.name);
    VERIFY_MSG(d.name() == e.name && d.marker_size() == e.size && d.count() == e.count &&
                   d.max_correction_bits() == e.correction,
               std::string(e.name) + " has the expected size, count and correction");
  }
  VERIFY(test::throws<std::invalid_argument>([] { Dictionary d("DICT_4X4_51"); }));
  VERIFY(test::throws<std::out_of_range>([] { (void)Dictionary("DICT_4X4_50").code(50); }));
  VERIFY(test::throws<std::out_of_range>([] { (void)Dictionary("DICT_4X4_50").cells(-1); }));
}

// Codes read row by row, white = 1: ArUco markers as OpenCV 4.12 draws them, AprilTag tags as
// the AprilTag library 3.4.5 draws them (OpenCV turns those a half turn).
void check_known_codes() {
  VERIFY(Dictionary("DICT_4X4_50").code(0) == 0xb532);
  VERIFY(Dictionary("DICT_4X4_50").code(1) == 0x0f9a);
  VERIFY(Dictionary("DICT_APRILTAG_16h5").code(0) == 0x231b);
  VERIFY(Dictionary("DICT_APRILTAG_36h11").code(0) == 0xd5d628584ULL);
}

void check_prefixes() {
  for (const char* n : {"4X4", "5X5", "6X6", "7X7"}) {
    const Dictionary all(std::string("DICT_") + n + "_1000");
    for (const int count : {50, 100, 250}) {
      const Dictionary part(std::string("DICT_") + n + "_" + std::to_string(count));
      bool same = true;
      for (int id = 0; id < count; ++id) same = same && part.code(id) == all.code(id);
      VERIFY_MSG(same, std::string(part.name()) + " is the start of " + std::string(all.name()));
    }
  }
}

void check_rotations_and_uniqueness() {
  for (const std::string_view name : dictionary_names()) {
    const Dictionary d(name);
    const int n = d.marker_size();
    std::set<std::uint64_t> others;  // Every turn of every marker before this one.
    bool distinct = true;
    bool turns_back = true;
    std::vector<int> symmetric;  // Markers that look the same turned.
    for (int id = 0; id < d.count(); ++id) {
      std::uint64_t code = d.code(id);
      const std::uint64_t first = code;
      std::set<std::uint64_t> turns;
      for (int r = 0; r < 4; ++r) {
        distinct = distinct && others.count(code) == 0;
        turns.insert(code);
        code = rotate_code(code, n);
      }
      turns_back = turns_back && code == first;
      if (turns.size() != 4) symmetric.push_back(id);
      others.insert(turns.begin(), turns.end());
    }
    VERIFY_MSG(distinct, std::string(name) + ": no marker looks like another in any turn");
    VERIFY_MSG(turns_back, std::string(name) + ": four quarter turns are no turn");
    // DICT_ARUCO_ORIGINAL's last marker is three white columns: the same turned a half turn.
    const std::vector<int> expected =
        name == "DICT_ARUCO_ORIGINAL" ? std::vector<int>{1023} : std::vector<int>{};
    VERIFY_MSG(symmetric == expected, std::string(name) + ": the symmetric markers");
  }
  // One quarter turn clockwise: the top row becomes the right column.
  // 1000      0001
  // 0000  ->  0000
  // 0000      0000
  // 0000      0000
  VERIFY(rotate_code(0x8000, 4) == 0x1000);
}

void check_cells() {
  const Dictionary d("DICT_5X5_100");
  const std::vector<std::uint8_t> cells = d.cells(7);
  VERIFY(cells.size() == 49);
  bool border = true;
  for (int i = 0; i < 7; ++i) {
    border = border && cells[i] == 0 && cells[42 + i] == 0 && cells[i * 7] == 0 &&
             cells[i * 7 + 6] == 0;
  }
  VERIFY_MSG(border, "the border is black");
  std::uint64_t code = 0;
  for (int row = 1; row <= 5; ++row) {
    for (int col = 1; col <= 5; ++col) code = (code << 1U) | cells[row * 7 + col];
  }
  VERIFY(code == d.code(7));
}

}  // namespace
}  // namespace holoscan::fiducials

int main() {
  using holoscan::fiducials::test::TestCase;
  return holoscan::fiducials::test::run({
      {"dictionary names, sizes and corrections", holoscan::fiducials::check_names},
      {"dictionary codes match OpenCV and AprilTag", holoscan::fiducials::check_known_codes},
      {"dictionary prefixes", holoscan::fiducials::check_prefixes},
      {"dictionary rotations and uniqueness", holoscan::fiducials::check_rotations_and_uniqueness},
      {"dictionary printed cells", holoscan::fiducials::check_cells},
  });
}
