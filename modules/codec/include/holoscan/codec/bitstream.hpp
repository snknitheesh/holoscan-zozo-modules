// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Reading H.264 and H.265 Annex-B byte streams: NAL units by their start codes, their types, and
// what an access unit holds. No SDK, CUDA or GPU.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <holoscan/codec/codec_types.hpp>

namespace holoscan::codec {

// One NAL unit of an Annex-B stream: its bytes after the start code, up to the next start code.
struct NalUnit {
  std::size_t offset{};  // Of the first byte after the start code.
  std::size_t size{};
  // nal_unit_type: 5 bits for H.264, 6 bits for H.265.
  int type{};
};

// The NAL units of `stream`, in order. Bytes before the first start code (00 00 01 or 00 00 00 01)
// are not a NAL unit and are skipped; an empty or start-code-free stream has none.
[[nodiscard]] std::vector<NalUnit> split_nal_units(Codec codec,
                                                   std::span<const std::uint8_t> stream);

// What one access unit (one encoded frame) contains.
struct AccessUnitInfo {
  std::size_t nal_units{};
  bool vps{};  // H.265 only.
  bool sps{};
  bool pps{};
  // An IDR picture (H.264 type 5; H.265 IDR_W_RADL or IDR_N_LP), where decoding may start.
  bool idr{};
  // An intra random access point: an IDR, or for H.265 also a CRA or BLA picture.
  bool random_access{};
  // Slices of any picture.
  std::size_t slices{};
};

[[nodiscard]] AccessUnitInfo analyze_access_unit(Codec codec, std::span<const std::uint8_t> au);

// Whether `au` can start decoding: a random access picture with every parameter set the codec
// needs (VPS for H.265, SPS, PPS).
[[nodiscard]] bool is_keyframe(Codec codec, std::span<const std::uint8_t> au);

// The parameter-set NAL units of `au` (VPS, SPS, PPS), each with its 4-byte start code, in the
// order they appear; empty when it has none.
[[nodiscard]] std::vector<std::uint8_t> parameter_sets(Codec codec,
                                                       std::span<const std::uint8_t> au);

}  // namespace holoscan::codec
