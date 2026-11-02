// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Reading a holoscan::schema::ImageT as a codec image, and copying a host one to the device.
// Module-private.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>

#include <holoscan/codec/video_codec.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::codec::detail {

struct ImageView {
  // The pixels where they are: device or host memory.
  DeviceImage image;
  bool on_device{};
};

// `image` as a codec image, or why it cannot be one.
[[nodiscard]] holoscan::expected<ImageView, std::string> view_image(
    const holoscan::schema::ImageT& image);

// Bytes of the image's pixels, packed.
[[nodiscard]] std::size_t image_bytes(const DeviceImage& image);

// A device copy of host images, grown as needed.
class ImageStaging {
 public:
  ImageStaging() = default;
  ImageStaging(const ImageStaging&) = delete;
  ImageStaging& operator=(const ImageStaging&) = delete;
  ~ImageStaging() { release(); }

  // Copies `host` to the device on `stream` and returns the copy, valid until the next call. The
  // copy has been read from `host` when this returns.
  // @throws std::runtime_error on a CUDA failure.
  [[nodiscard]] DeviceImage to_device(const DeviceImage& host, cudaStream_t stream);
  void release() noexcept;

 private:
  std::uint8_t* data_{};
  std::size_t bytes_{};
};

}  // namespace holoscan::codec::detail
