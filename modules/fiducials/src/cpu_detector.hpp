// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The detector on the CPU, and what both backends share on the host. Free of CUDA and the SDK,
// so the SDK-free tests run it.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <holoscan/fiducials/dictionary.hpp>
#include <holoscan/fiducials/types.hpp>

#include "detection_core.hpp"

namespace holoscan::fiducials {

// The core's settings for `options` and a width x height image.
// @throws std::invalid_argument for options or a size no backend accepts.
[[nodiscard]] core::Params make_params(const DetectorOptions& options, int width, int height);

// Every marker's code turned 0, 1, 2 and 3 quarter turns clockwise, as core::identify reads them.
[[nodiscard]] std::vector<std::uint64_t> rotated_codes(const Dictionary& dictionary);

// The markers among examined candidates: of two quads one inside the other, only the outer one
// (a marker's own inner cells can form a quad), sorted by id and then by position, at most
// kMaxMarkersPerFrame of them. `dropped_candidates` were not examined.
[[nodiscard]] FrameMarkers collect_markers(std::span<const core::CandidateResult> results,
                                           std::uint32_t dropped_candidates);

class CpuDetector {
 public:
  // @throws std::invalid_argument for options or a size no backend accepts.
  CpuDetector(const DetectorOptions& options, int width, int height);

  // Markers in `image`, a host image of the configured size; corners without poses.
  // @throws std::invalid_argument for an image of another size or in device memory.
  [[nodiscard]] FrameMarkers detect(const GrayImageView& image);

  // Each seed moved to the chessboard corner in its window; nullopt where there is none.
  // @throws std::invalid_argument for an image of another size or in device memory.
  [[nodiscard]] std::vector<std::optional<Point2>> refine_saddles(
      const GrayImageView& image, std::span<const SaddleSeed> seeds) const;

 private:
  void check(const GrayImageView& image) const;

  core::Params params_;
  std::vector<std::uint64_t> codes_;
  std::vector<std::uint8_t> tile_lo_, tile_hi_, near_lo_, near_hi_;
  std::vector<std::int32_t> labels_;  // Component root per dark pixel, -1 elsewhere.
  std::vector<std::int32_t> slots_;   // Per root: pixel count, then candidate slot or -1.
  std::vector<core::Candidate> candidates_;
};

}  // namespace holoscan::fiducials
