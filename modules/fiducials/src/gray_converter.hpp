// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Converts an image of any supported packing, on the host or the device, to gray
// where a backend wants it. A gray image already in place is passed through. A host
// image for a host backend touches no CUDA at all, so the cpu backend runs without a
// GPU; the pinned and device buffers are made only for the paths that use them.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/fiducials/types.hpp>

namespace holoscan::fiducials {

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
  // ordered on `stream` (unused, and may be null, when the input and the target
  // are both on the host); a host result is complete when this returns.
  // @throws std::runtime_error on a CUDA failure.
  [[nodiscard]] GrayImageView convert(const ImageView& input, MemoryLocation target,
                                      cudaStream_t stream);

  // The CPU waits and host/device copies of the conversions since the last
  // call, and counts anew.
  [[nodiscard]] TransferCounts take_transfers() noexcept {
    const TransferCounts taken = transfers_;
    transfers_ = {};
    return taken;
  }

 private:
  // Makes the buffers `input` needs for `target`, again when its size changes.
  void reserve(const ImageView& input, MemoryLocation target);

  struct HostFree {
    void operator()(std::uint8_t* data) const noexcept { cudaFreeHost(data); }
  };
  struct DeviceFree {
    void operator()(std::uint8_t* data) const noexcept { cudaFree(data); }
  };

  int width_{};
  int height_{};
  // Each width_ x height_, made on first use: plain host memory for a host image
  // converted for a host backend, pinned host memory for one that crosses to or
  // from the device, and device memory.
  std::vector<std::uint8_t> plain_gray_;
  std::unique_ptr<std::uint8_t, HostFree> pinned_gray_;
  std::unique_ptr<std::uint8_t, DeviceFree> device_gray_;
  TransferCounts transfers_;
};

}  // namespace holoscan::fiducials
