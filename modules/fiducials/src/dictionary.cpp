// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/fiducials/dictionary.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "dictionary_codes.hpp"

namespace holoscan::fiducials {

Dictionary::Dictionary(std::string_view name) : table_(nullptr) {
  for (const DictionaryTable& table : dictionary_tables()) {
    if (name == table.name) {
      table_ = &table;
      return;
    }
  }
  throw std::invalid_argument("unknown dictionary '" + std::string(name) +
                              "'; use a name such as DICT_4X4_50 or DICT_APRILTAG_36h11");
}

std::string_view Dictionary::name() const noexcept { return table_->name; }
int Dictionary::marker_size() const noexcept { return table_->marker_size; }
int Dictionary::count() const noexcept { return table_->count; }
int Dictionary::max_correction_bits() const noexcept { return table_->max_correction_bits; }

std::uint64_t Dictionary::code(int id) const {
  if (id < 0 || id >= table_->count) {
    throw std::out_of_range(std::string(table_->name) + " has no marker " + std::to_string(id));
  }
  return table_->codes[id];
}

std::vector<std::uint8_t> Dictionary::cells(int id) const {
  const std::uint64_t bits = code(id);
  const int n = table_->marker_size;
  const int side = n + 2;
  std::vector<std::uint8_t> result(static_cast<std::size_t>(side) * side, 0);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      const int bit = n * n - 1 - (row * n + col);
      result[static_cast<std::size_t>(row + 1) * side + col + 1] = (bits >> bit) & 1U;
    }
  }
  return result;
}

std::vector<std::string_view> dictionary_names() {
  std::vector<std::string_view> names;
  for (const DictionaryTable& table : dictionary_tables()) names.emplace_back(table.name);
  return names;
}

std::uint64_t rotate_code(std::uint64_t code, int marker_size) noexcept {
  const int n = marker_size;
  std::uint64_t result = 0;
  // Turned a quarter clockwise, the cell at (row, col) comes from (n - 1 - col, row).
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      const int from = (n - 1 - col) * n + row;
      const std::uint64_t bit = (code >> (n * n - 1 - from)) & 1U;
      result |= bit << (n * n - 1 - (row * n + col));
    }
  }
  return result;
}

}  // namespace holoscan::fiducials
