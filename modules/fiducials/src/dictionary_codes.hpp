// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The predefined dictionaries' tables, generated into dictionary_codes.cpp.

#pragma once

#include <array>
#include <cstdint>

namespace holoscan::fiducials {

struct DictionaryTable {
  const char* name;         // OpenCV's name, such as "DICT_4X4_50".
  int marker_size;          // Inner cells per side.
  int count;                // Markers, ids 0 .. count - 1.
  int max_correction_bits;  // Bit errors the dictionary can correct at most.
  const std::uint64_t* codes;
};

[[nodiscard]] const std::array<DictionaryTable, 22>& dictionary_tables();

}  // namespace holoscan::fiducials
