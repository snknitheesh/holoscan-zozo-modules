// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The predefined marker dictionaries: OpenCV's ArUco dictionaries and the AprilTag families
// OpenCV carries, with OpenCV's names and ids. ArUco markers have OpenCV's bits; AprilTag tags
// are as the AprilTag library draws them, which OpenCV turns a half turn, so that their corners
// and poses agree with holoscan-apriltag's.

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace holoscan::fiducials {

struct DictionaryTable;

class Dictionary {
 public:
  // @throws std::invalid_argument for a name dictionary_names() does not list.
  explicit Dictionary(std::string_view name);

  [[nodiscard]] std::string_view name() const noexcept;
  // Inner cells per side, 4..7; a printed marker has a one-cell black border around them.
  [[nodiscard]] int marker_size() const noexcept;
  // Markers, with ids 0 .. count() - 1.
  [[nodiscard]] int count() const noexcept;
  // Bit errors the dictionary can correct: OpenCV's maxCorrectionBits, and for the AprilTag
  // families (d - 1) / 2 for their smallest distance d between two codes.
  [[nodiscard]] int max_correction_bits() const noexcept;
  // Marker `id`'s inner cells row by row from the top-left one, the first in the most
  // significant of marker_size()^2 bits, 1 for white.
  // @throws std::out_of_range for an id outside 0 .. count() - 1.
  [[nodiscard]] std::uint64_t code(int id) const;
  // Marker `id` as printed, its black border included: (marker_size() + 2)^2 cells row by row,
  // 1 for white.
  // @throws std::out_of_range for an id outside 0 .. count() - 1.
  [[nodiscard]] std::vector<std::uint8_t> cells(int id) const;

 private:
  const DictionaryTable* table_;
};

// Every predefined dictionary's name, such as "DICT_4X4_50" or "DICT_APRILTAG_36h11".
[[nodiscard]] std::vector<std::string_view> dictionary_names();

// `code`, a marker_size x marker_size grid as Dictionary::code() packs it, turned a quarter
// turn clockwise.
[[nodiscard]] std::uint64_t rotate_code(std::uint64_t code, int marker_size) noexcept;

}  // namespace holoscan::fiducials
