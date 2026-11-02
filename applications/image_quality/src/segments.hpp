// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's synthetic camera: a moving scene that cycles through segments of
// degradations, and the flags each frame should get.

#pragma once

#include <cstdint>
#include <string_view>

#include <holoscan/image_quality/quality.hpp>
#include <holoscan/image_quality/synthetic.hpp>

namespace image_quality_app {

enum class Segment : std::uint8_t {
  kSharp,
  kBlurred,  // Gaussian blur, sigma 3.
  kDark,     // A quarter of the exposure.
  kNoisy,    // Gaussian noise, sigma 12 per channel.
  kFrozen,   // The segment's first frame, repeated.
  kSmudged,  // A hazy smudge that stays put while the scene moves.
  kTinted,   // Red up 30 %, blue down 25 %.
  kCount
};
inline constexpr int kSegmentCount = static_cast<int>(Segment::kCount);

[[nodiscard]] std::string_view segment_name(Segment segment) noexcept;

struct Schedule {
  int width{640};
  int height{360};
  // Frames per segment; must be more than the obstruction window to show it.
  std::uint32_t segment_frames{45};
  // The ImageQualityOp's obstruction window, for the expected flags.
  std::uint32_t obstruction_frames{30};

  [[nodiscard]] Segment segment(std::uint64_t frame) const noexcept;
  // Frame `frame` of the camera.
  [[nodiscard]] holoscan::image_quality::HostImage render(std::uint64_t frame) const;
  // The flags ImageQualityOp, with default thresholds and this obstruction
  // window, must give frame `frame`.
  [[nodiscard]] std::uint32_t expected_flags(std::uint64_t frame) const noexcept;
  // The smudge of the smudged segment, in pixels.
  [[nodiscard]] holoscan::image_quality::Smudge smudge() const noexcept;
};

}  // namespace image_quality_app
