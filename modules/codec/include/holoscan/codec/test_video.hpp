// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic test video and a quality measure, for the tests, the demo and anyone checking a
// codec setting: frames generated on the host, each one different from the last, and PSNR. No
// SDK, CUDA or GPU.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace holoscan::codec {

// Frame `index` of the test video, `width` x `height` packed RGB8 (or RGBA8 with alpha 255 for
// `channels` 4): a diagonal color gradient sliding right 4 pixels a frame, vertical color bars
// sliding down 2 pixels a frame, and a band of blocky "text": rows of 5x7-pixel glyphs drawn at
// twice their size, scrolling left 6 pixels a frame and changing every 15 frames, under a still
// first row that starts with the frame's index in digits.
// The same arguments always give the same frame.
[[nodiscard]] std::vector<std::uint8_t> render_test_frame(int width, int height,
                                                          std::uint32_t index, int channels = 3);

// Peak signal-to-noise ratio of `test` against `reference`, in dB, over `width` x `height` pixels
// of `channels` bytes, rows `reference_pitch` and `test_pitch` bytes apart. Infinite when they are
// equal.
[[nodiscard]] double psnr(const std::uint8_t* reference, std::size_t reference_pitch,
                          const std::uint8_t* test, std::size_t test_pitch, int width, int height,
                          int channels);

// The same, for two packed images of one size.
[[nodiscard]] double psnr(std::span<const std::uint8_t> reference,
                          std::span<const std::uint8_t> test, int width, int height, int channels);

}  // namespace holoscan::codec
