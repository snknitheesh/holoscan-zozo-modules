// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "tag_family.hpp"

#include <apriltag.h>
#include <tag16h5.h>
#include <tag25h9.h>
#include <tag36h10.h>
#include <tag36h11.h>
#include <tagCircle21h7.h>
#include <tagCircle49h12.h>
#include <tagCustom48h12.h>
#include <tagStandard41h12.h>
#include <tagStandard52h13.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace holoscan::apriltag {

struct FamilyEntry {
  std::string_view name;
  apriltag_family_t* (*create)();
  void (*destroy)(apriltag_family_t*);
};

namespace {

constexpr std::array<FamilyEntry, 9> kFamilies{{
    {"tag16h5", tag16h5_create, tag16h5_destroy},
    {"tag25h9", tag25h9_create, tag25h9_destroy},
    {"tag36h10", tag36h10_create, tag36h10_destroy},
    {"tag36h11", tag36h11_create, tag36h11_destroy},
    {"tagCircle21h7", tagCircle21h7_create, tagCircle21h7_destroy},
    {"tagCircle49h12", tagCircle49h12_create, tagCircle49h12_destroy},
    {"tagCustom48h12", tagCustom48h12_create, tagCustom48h12_destroy},
    {"tagStandard41h12", tagStandard41h12_create, tagStandard41h12_destroy},
    {"tagStandard52h13", tagStandard52h13_create, tagStandard52h13_destroy},
}};

const FamilyEntry* find_family(std::string_view name) noexcept {
  const auto* it = std::find_if(kFamilies.begin(), kFamilies.end(),
                                [name](const FamilyEntry& entry) { return entry.name == name; });
  return it == kFamilies.end() ? nullptr : &*it;
}

}  // namespace

bool is_known_family(std::string_view name) noexcept { return find_family(name) != nullptr; }

std::uint64_t decode_table_entries(std::string_view family, int max_hamming) {
  const TagFamily tags(family);
  const std::uint64_t codes = tags.get()->ncodes;
  const std::uint64_t bits = tags.get()->nbits;
  std::uint64_t variants = 1;
  if (max_hamming >= 1) variants += bits;
  if (max_hamming >= 2) variants += bits * (bits - 1) / 2;
  return codes * variants;
}

TagFamily::TagFamily(std::string_view name) : entry_(find_family(name)) {
  if (entry_ == nullptr) {
    throw std::invalid_argument("unknown AprilTag family '" + std::string(name) + "'");
  }
  family_ = entry_->create();
  if (family_ == nullptr) {
    throw std::runtime_error("failed to create AprilTag family '" + std::string(name) + "'");
  }
}

TagFamily::~TagFamily() { entry_->destroy(family_); }

std::string_view TagFamily::name() const noexcept { return entry_->name; }

}  // namespace holoscan::apriltag
