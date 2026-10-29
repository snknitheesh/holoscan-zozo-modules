// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The interface both detector backends implement, and its factory. Backends find markers and
// refine chessboard corners; pose.hpp and charuco.hpp do the rest on the host, the same way for
// both.

#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <holoscan/fiducials/types.hpp>

namespace holoscan::fiducials {

class DetectorBackend {
 public:
  virtual ~DetectorBackend() = default;
  DetectorBackend(const DetectorBackend&) = delete;
  DetectorBackend& operator=(const DetectorBackend&) = delete;

  // "cpu" or "cuda".
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  // Where the backend reads its images; the caller converts and copies to match.
  [[nodiscard]] virtual MemoryLocation input_location() const noexcept = 0;

  // The markers in `image`, which must be of the size the backend was built for and in
  // input_location(); their corners, without poses. Device work is ordered on `stream`; the
  // results are on the host when this returns.
  // @throws std::invalid_argument for an image of another size or location.
  // @throws std::runtime_error on a backend failure.
  [[nodiscard]] virtual FrameMarkers detect(const GrayImageView& image, cudaStream_t stream) = 0;

  // Each seed moved to the chessboard corner within its window in `image` (as for detect);
  // nullopt where there is none.
  // @throws std::invalid_argument for an image of another size or location.
  // @throws std::runtime_error on a backend failure.
  [[nodiscard]] virtual std::vector<std::optional<Point2>> refine_saddles(
      const GrayImageView& image, std::span<const SaddleSeed> seeds, cudaStream_t stream) = 0;

 protected:
  DetectorBackend() = default;
};

// The facts about this process's platform that resolve_auto() needs.
[[nodiscard]] PlatformFacts probe_platform() noexcept;
// `backend`, or for kAuto, resolve_auto() of this platform's facts.
[[nodiscard]] Backend resolve_backend(Backend backend) noexcept;

// Builds `backend` for `width` x `height` gray images; kAuto builds resolve_backend()'s.
// @throws std::invalid_argument if the options or size are not usable.
// @throws std::runtime_error if the backend fails to initialize.
[[nodiscard]] std::unique_ptr<DetectorBackend> create_backend(Backend backend,
                                                              const DetectorOptions& options,
                                                              int width, int height);

}  // namespace holoscan::fiducials
