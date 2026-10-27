// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Owns one AprilTag 3 family (its code book and bit layout) by name.

#pragma once

#include <cstdint>
#include <string_view>

struct apriltag_family;

namespace holoscan::apriltag {

struct FamilyEntry;

// Whether `name` is a family AprilTag 3 ships (also the set VPI supports).
[[nodiscard]] bool is_known_family(std::string_view name) noexcept;

// Codes plus every variant within `max_hamming` bits: what a decoder must index.
// @throws std::invalid_argument for an unknown family.
[[nodiscard]] std::uint64_t decode_table_entries(std::string_view family, int max_hamming);

class TagFamily {
 public:
  // @throws std::invalid_argument if `name` is not a known family.
  explicit TagFamily(std::string_view name);
  ~TagFamily();
  TagFamily(const TagFamily&) = delete;
  TagFamily& operator=(const TagFamily&) = delete;

  [[nodiscard]] apriltag_family* get() const noexcept { return family_; }
  [[nodiscard]] std::string_view name() const noexcept;

 private:
  const FamilyEntry* entry_;
  apriltag_family* family_;
};

}  // namespace holoscan::apriltag
