// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Converts an image of any supported packing, on the host or the device, to gray
// where a backend wants it. A gray image already in place is passed through.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include <holoscan/apriltag/detection.hpp>

namespace holoscan::apriltag {

enum class PixelFormat : std::uint8_t { kMono8, kRgb8, kBgr8, kRgba8, kBgra8 };

[[nodiscard]] int channels(PixelFormat format) noexcept;

struct ImageView {
  const std::uint8_t* data{};
  int width{};
  int height{};
  std::size_t pitch{};
  PixelFormat format{PixelFormat::kMono8};
  MemoryLocation location{MemoryLocation::kHost};
};

class GrayConverter {
 public:
  // `input` as gray in `target`, valid until the next call. Device work is
  // ordered on `stream`; a host result is complete when this returns.
  // @throws std::runtime_error on a CUDA failure.
  [[nodiscard]] GrayImageView convert(const ImageView& input, MemoryLocation target,
                                      cudaStream_t stream);

 private:
  struct HostFree {
    void operator()(std::uint8_t* data) const noexcept { cudaFreeHost(data); }
  };
  struct DeviceFree {
    void operator()(std::uint8_t* data) const noexcept { cudaFree(data); }
  };

  int width_{};
  int height_{};
  std::unique_ptr<std::uint8_t, HostFree> host_gray_;      // Pinned, width_ x height_.
  std::unique_ptr<std::uint8_t, DeviceFree> device_gray_;  // width_ x height_.
};

}  // namespace holoscan::apriltag
