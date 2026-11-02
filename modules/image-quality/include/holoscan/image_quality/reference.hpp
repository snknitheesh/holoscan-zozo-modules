// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CPU reference: gathers a frame's statistics on the host, pixel by pixel,
// exactly as the GPU analyzer's kernels do. The tests check the kernels against
// it bit for bit; it also checks images where no GPU is at hand. It is not fast.

#pragma once

#include <cstdint>
#include <vector>

#include <holoscan/image_quality/quality.hpp>

namespace holoscan::image_quality {

class ReferenceStats {
 public:
  // @throws std::invalid_argument if `options` is not valid.
  explicit ReferenceStats(StatsOptions options);

  // Statistics of `image`, in host memory, compared with the previous image when
  // it had the same size.
  // @throws std::invalid_argument for an empty image or a pitch below a row's bytes.
  [[nodiscard]] FrameStats gather(const ImageView& image);
  // Forgets the previous image.
  void reset() noexcept;

 private:
  StatsOptions options_;
  std::vector<std::uint8_t> previous_luma_;
  int previous_width_{};
  int previous_height_{};
};

// Checks `image`'s geometry for any backend.
// @throws std::invalid_argument for an empty image or a pitch below a row's bytes.
void check_image(const ImageView& image);

}  // namespace holoscan::image_quality
