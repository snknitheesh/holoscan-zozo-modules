// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "segments.hpp"

namespace image_quality_app {

namespace iq = holoscan::image_quality;

std::string_view segment_name(Segment segment) noexcept {
  switch (segment) {
    case Segment::kSharp:
      return "sharp";
    case Segment::kBlurred:
      return "blurred";
    case Segment::kDark:
      return "dark";
    case Segment::kNoisy:
      return "noisy";
    case Segment::kFrozen:
      return "frozen";
    case Segment::kSmudged:
      return "smudged";
    case Segment::kTinted:
      return "tinted";
    case Segment::kCount:
      break;
  }
  return "unknown";
}

Segment Schedule::segment(std::uint64_t frame) const noexcept {
  return static_cast<Segment>((frame / segment_frames) % kSegmentCount);
}

iq::Smudge Schedule::smudge() const noexcept {
  return iq::Smudge{.cx = 0.55 * width, .cy = 0.5 * height, .rx = 0.2 * width, .ry = 0.25 * height};
}

iq::HostImage Schedule::render(std::uint64_t frame) const {
  const Segment which = segment(frame);
  const std::uint64_t start = frame - frame % segment_frames;
  // The frozen segment repeats its first frame.
  const auto scene_frame = static_cast<std::uint32_t>(which == Segment::kFrozen ? start : frame);
  iq::HostImage image = iq::render_scene(width, height, scene_frame);
  switch (which) {
    case Segment::kBlurred:
      iq::gaussian_blur(image, 3.0);
      break;
    case Segment::kDark:
      iq::apply_gain(image, 0.25);
      break;
    case Segment::kNoisy:
      iq::add_noise(image, 12.0, frame);
      break;
    case Segment::kSmudged:
      iq::apply_smudge(image, smudge());
      break;
    case Segment::kTinted:
      iq::apply_channel_gains(image, 1.3, 1.0, 0.75);
      break;
    default:
      break;
  }
  return image;
}

std::uint32_t Schedule::expected_flags(std::uint64_t frame) const noexcept {
  const std::uint64_t index = frame % segment_frames;
  switch (segment(frame)) {
    case Segment::kBlurred:
      return iq::kBlurred;
    case Segment::kDark:
      return iq::kUnderexposed;
    case Segment::kNoisy:
      return iq::kNoisy;
    case Segment::kFrozen:
      return index == 0 ? 0U : iq::kFrozen;
    case Segment::kSmudged:
      // The smudge's tiles must be low-texture over the whole window first.
      return index + 1 >= obstruction_frames ? iq::kObstructed : 0U;
    case Segment::kTinted:
      return iq::kColorCast;
    default:
      return 0U;
  }
}

}  // namespace image_quality_app
