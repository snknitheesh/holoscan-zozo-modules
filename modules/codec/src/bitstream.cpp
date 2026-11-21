// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/codec/bitstream.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace holoscan::codec {
namespace {

// H.264 nal_unit_type values (ITU-T H.264 table 7-1).
constexpr int kH264NonIdrSlice = 1;
constexpr int kH264IdrSlice = 5;
constexpr int kH264Sei = 6;
constexpr int kH264Sps = 7;
constexpr int kH264Pps = 8;

// H.265 nal_unit_type values (ITU-T H.265 table 7-1).
constexpr int kH265LastVclType = 31;
constexpr int kH265BlaWLp = 16;
constexpr int kH265IdrWRadl = 19;
constexpr int kH265IdrNLp = 20;
constexpr int kH265CraNut = 21;
constexpr int kH265Vps = 32;
constexpr int kH265Sps = 33;
constexpr int kH265Pps = 34;

// Where the start code at or after `from` begins and how long it is; {size, 0} when there is none.
struct StartCode {
  std::size_t at{};
  std::size_t length{};
};

StartCode find_start_code(std::span<const std::uint8_t> stream, std::size_t from) {
  for (std::size_t i = from; i + 2 < stream.size(); ++i) {
    if (stream[i] != 0 || stream[i + 1] != 0) {
      continue;
    }
    if (stream[i + 2] == 1) {
      // A zero before 00 00 01 belongs to a four-byte start code.
      if (i > from && stream[i - 1] == 0) {
        return {i - 1, 4};
      }
      return {i, 3};
    }
  }
  return {stream.size(), 0};
}

int nal_type(Codec codec, std::uint8_t first_byte) {
  return codec == Codec::kH264 ? first_byte & 0x1F : (first_byte >> 1) & 0x3F;
}

// Whether an H.264 SEI NAL unit (`nal`, its header byte first) holds a recovery point message
// (payloadType 6, ITU-T H.264 D.1.8).
bool has_recovery_point(std::span<const std::uint8_t> nal) {
  constexpr int kRecoveryPoint = 6;
  // The payload without its emulation prevention bytes (00 00 03 -> 00 00).
  std::vector<std::uint8_t> rbsp;
  rbsp.reserve(nal.size());
  int zeros = 0;
  for (std::size_t i = 1; i < nal.size(); ++i) {
    if (zeros >= 2 && nal[i] == 3) {
      zeros = 0;
      continue;
    }
    zeros = nal[i] == 0 ? zeros + 1 : 0;
    rbsp.push_back(nal[i]);
  }
  std::size_t at = 0;
  // Messages until the rbsp trailing bits (0x80).
  while (at < rbsp.size() && rbsp[at] != 0x80) {
    std::size_t type = 0;
    while (at < rbsp.size() && rbsp[at] == 0xFF) type += rbsp[at++];
    if (at >= rbsp.size()) return false;
    type += rbsp[at++];
    std::size_t size = 0;
    while (at < rbsp.size() && rbsp[at] == 0xFF) size += rbsp[at++];
    if (at >= rbsp.size()) return false;
    size += rbsp[at++];
    if (type == kRecoveryPoint) return true;
    at += size;
  }
  return false;
}

bool is_parameter_set(Codec codec, int type) {
  return codec == Codec::kH264 ? type == kH264Sps || type == kH264Pps
                               : type == kH265Vps || type == kH265Sps || type == kH265Pps;
}

}  // namespace

std::vector<NalUnit> split_nal_units(Codec codec, std::span<const std::uint8_t> stream) {
  std::vector<NalUnit> units;
  StartCode code = find_start_code(stream, 0);
  while (code.length != 0) {
    const std::size_t begin = code.at + code.length;
    const StartCode next = find_start_code(stream, begin);
    // Trailing zero bytes before the next start code are not part of the NAL unit.
    std::size_t end = next.at;
    while (end > begin && next.length != 0 && stream[end - 1] == 0) {
      --end;
    }
    if (end > begin) {
      units.push_back(
          NalUnit{.offset = begin, .size = end - begin, .type = nal_type(codec, stream[begin])});
    }
    code = next;
  }
  return units;
}

AccessUnitInfo analyze_access_unit(Codec codec, std::span<const std::uint8_t> au) {
  AccessUnitInfo info;
  for (const NalUnit& unit : split_nal_units(codec, au)) {
    ++info.nal_units;
    const int type = unit.type;
    if (codec == Codec::kH264) {
      info.sps = info.sps || type == kH264Sps;
      info.pps = info.pps || type == kH264Pps;
      if (type >= kH264NonIdrSlice && type <= kH264IdrSlice) {
        ++info.slices;
      }
      if (type == kH264IdrSlice) {
        info.idr = true;
        info.random_access = true;
      }
      if (type == kH264Sei && has_recovery_point(au.subspan(unit.offset, unit.size))) {
        info.recovery_point = true;
        info.random_access = true;
      }
    } else {
      info.vps = info.vps || type == kH265Vps;
      info.sps = info.sps || type == kH265Sps;
      info.pps = info.pps || type == kH265Pps;
      if (type <= kH265LastVclType) {
        ++info.slices;
      }
      if (type == kH265IdrWRadl || type == kH265IdrNLp) {
        info.idr = true;
      }
      if (type >= kH265BlaWLp && type <= kH265CraNut) {
        info.random_access = true;
      }
    }
  }
  return info;
}

bool is_keyframe(Codec codec, std::span<const std::uint8_t> au) {
  const AccessUnitInfo info = analyze_access_unit(codec, au);
  return info.random_access && info.sps && info.pps && (codec == Codec::kH264 || info.vps);
}

std::vector<std::uint8_t> parameter_sets(Codec codec, std::span<const std::uint8_t> au) {
  static constexpr std::uint8_t kStartCode[] = {0, 0, 0, 1};
  std::vector<std::uint8_t> sets;
  for (const NalUnit& unit : split_nal_units(codec, au)) {
    if (is_parameter_set(codec, unit.type)) {
      sets.insert(sets.end(), std::begin(kStartCode), std::end(kStartCode));
      sets.insert(sets.end(), au.begin() + static_cast<std::ptrdiff_t>(unit.offset),
                  au.begin() + static_cast<std::ptrdiff_t>(unit.offset + unit.size));
    }
  }
  return sets;
}

}  // namespace holoscan::codec
