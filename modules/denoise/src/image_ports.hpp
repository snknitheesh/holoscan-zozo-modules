// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What the module's operators share at their ports: reading an ImageT's
// pixels, staging host images on the GPU (logged once), and the device image
// outputs and their descriptors.

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/denoise/denoise_params.hpp>
#include <holoscan/denoise/fusion_params.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::denoise::detail {

holoscan::Error invalid(std::string_view op, const std::string& message);
holoscan::Error failure(std::string_view op, const std::string& message);

// An 8-bit image's pixels where its tensor holds them.
struct InputImage {
  ImageView view;
  bool on_device{};
  // Bytes from each plane's first to its last pixel.
  std::size_t plane_bytes[2]{};
};

// Reads a MONO8, RGB8, RGBA8 or (with `allow_nv12`) NV12 image: uint8 [H, W],
// [H, W, channels] with packed pixels and any row pitch; NV12 by its plane
// layouts. Host or CUDA memory; at most `max_width` x `max_height`.
[[nodiscard]] holoscan::expected<InputImage, holoscan::Error> describe_image(
    const holoscan::schema::ImageT& image, std::string_view op, bool allow_nv12, int max_width,
    int max_height);

// A high-bit-depth image (MONO16, RGB16, RGBA16, MONO32F, RGB32F, RGBA32F).
struct HdrInput {
  HdrImageView view;
  bool on_device{};
  std::size_t bytes{};
};
[[nodiscard]] holoscan::expected<HdrInput, holoscan::Error> describe_hdr(
    const holoscan::schema::ImageT& image, std::string_view op, int max_width, int max_height);

// Waits for the producer of host-memory input, which the runtime does not order
// on the operator's stream (device memory it does).
[[nodiscard]] holoscan::expected<void, holoscan::Error> wait_for_host_input(
    const holoscan::Tensor& tensor);

// A device copy of host-memory images, grown as needed. The first copy is logged,
// with its size and how to avoid it.
class DeviceStaging {
 public:
  DeviceStaging() = default;
  ~DeviceStaging();
  DeviceStaging(const DeviceStaging&) = delete;
  DeviceStaging& operator=(const DeviceStaging&) = delete;

  // `image` itself when on the device; otherwise a device copy queued on `stream`.
  // `op` and `port` name the input in the log.
  [[nodiscard]] holoscan::expected<ImageView, holoscan::Error> on_device(
      const InputImage& image, const holoscan::Tensor& tensor, cudaStream_t stream,
      std::string_view op, std::string_view port);
  [[nodiscard]] holoscan::expected<const void*, holoscan::Error> on_device(
      const HdrInput& image, const holoscan::Tensor& tensor, cudaStream_t stream,
      std::string_view op, std::string_view port);
  // Whether a copy has been logged.
  [[nodiscard]] bool logged() const noexcept { return logged_; }
  void release() noexcept;

 private:
  [[nodiscard]] holoscan::expected<std::uint8_t*, holoscan::Error> reserve(std::size_t bytes,
                                                                           cudaStream_t stream,
                                                                           std::string_view op);
  [[nodiscard]] holoscan::expected<void, holoscan::Error> finish_copy(
      const holoscan::Tensor& tensor, cudaStream_t stream, std::string_view op,
      std::string_view port, std::size_t bytes);

  std::uint8_t* data_{};
  std::size_t bytes_{};
  bool logged_{};
};

// The output port's tensor: uint8 [H, W, channels] (NV12: [H * 3 / 2, W, 1]) in
// CUDA device memory, up to `max_width` x `max_height` x 4 bytes.
[[nodiscard]] holoscan::TensorOutputSpec image_output_spec(int max_width, int max_height);
[[nodiscard]] holoscan::PayloadOptions image_payload();
// The shape of an output image of `format`.
[[nodiscard]] std::array<std::int64_t, 3> image_shape(PixelFormat format, int width, int height);
// A view of a packed output image of `format` at `data`.
[[nodiscard]] ImageView packed_view(PixelFormat format, int width, int height, std::uint8_t* data);
// The descriptor of an output image: `format`, the size, `source`'s header,
// exposure and gain, and NV12's plane layouts.
[[nodiscard]] holoscan::schema::ImageT image_descriptor(PixelFormat format, int width, int height,
                                                        const holoscan::schema::ImageT& source);
[[nodiscard]] holoscan::schema::ImageEncoding encoding_of(PixelFormat format) noexcept;

}  // namespace holoscan::denoise::detail
