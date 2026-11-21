// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The reordered test streams of tests/data (see make_reordered_streams.sh): 30 frames of the test
// video at 320x240 with B-frames, as access units in decode order with each one's display index.

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/codec/bitstream.hpp>
#include <holoscan/codec/codec_types.hpp>

namespace holoscan::codec::test {

inline constexpr int kReorderedWidth = 320;
inline constexpr int kReorderedHeight = 240;
inline constexpr std::uint32_t kReorderedFrames = 30;

struct ReorderedUnit {
  std::vector<std::uint8_t> data;
  std::uint32_t display_index{};
};

// @throws std::runtime_error when the files are missing or do not match.
inline std::vector<ReorderedUnit> load_reordered_stream(Codec codec) {
  const std::filesystem::path base = std::filesystem::path(HOLOSCAN_CODEC_TEST_DATA_DIR) /
                                     ("reordered." + std::string(codec_name(codec)));
  std::ifstream stream_file(base, std::ios::binary);
  std::ifstream order_file(base.string() + ".order");
  if (!stream_file || !order_file) {
    throw std::runtime_error("cannot read " + base.string() + " and its .order file");
  }
  const std::vector<std::uint8_t> stream((std::istreambuf_iterator<char>(stream_file)),
                                         std::istreambuf_iterator<char>());
  // An access unit delimiter starts every access unit.
  const int delimiter = codec == Codec::kH264 ? 9 : 35;
  std::vector<std::size_t> starts;
  for (const NalUnit& nal : split_nal_units(codec, stream)) {
    if (nal.type == delimiter) {
      // Back to its start code, three or four bytes.
      std::size_t at = nal.offset - 3;
      if (at > 0 && stream[at - 1] == 0) --at;
      starts.push_back(at);
    }
  }
  std::vector<ReorderedUnit> units;
  for (std::size_t i = 0; i < starts.size(); ++i) {
    const std::size_t end = i + 1 < starts.size() ? starts[i + 1] : stream.size();
    units.push_back(
        {.data = std::vector<std::uint8_t>(stream.begin() + static_cast<std::ptrdiff_t>(starts[i]),
                                           stream.begin() + static_cast<std::ptrdiff_t>(end))});
  }
  for (ReorderedUnit& unit : units) {
    if (!(order_file >> unit.display_index)) {
      throw std::runtime_error(base.string() + ".order has fewer entries than access units");
    }
  }
  if (units.size() != kReorderedFrames) {
    throw std::runtime_error(base.string() + " has " + std::to_string(units.size()) +
                             " access units, not " + std::to_string(kReorderedFrames));
  }
  return units;
}

}  // namespace holoscan::codec::test
