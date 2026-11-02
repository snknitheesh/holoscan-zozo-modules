// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What ColorConvertOp and ResizeOp share: reading an image message into a
// view, staging host images on the GPU, choosing and logging the backend,
// and publishing the converted image with its descriptor.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/color_convert/backend.hpp>
#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/color_convert/converter.hpp>
#include <holoscan/color_convert/image_view.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::color_convert::detail {

// The tensor an image of `format` is published in.
[[nodiscard]] DLDataType dtype_of(PixelFormat format) noexcept;
// [H, W] for MONO, [H, W, C] for interleaved formats, [bytes] for planar ones.
[[nodiscard]] std::vector<std::int64_t> shape_of(PixelFormat format, int width, int height);
[[nodiscard]] holoscan::schema::ImageEncoding encoding_of(PixelFormat format) noexcept;
// The format of `encoding`; CUSTOM is `custom` when given.
[[nodiscard]] std::optional<PixelFormat> format_of(holoscan::schema::ImageEncoding encoding,
                                                   std::optional<PixelFormat> custom) noexcept;
// The matrix a YCbCr image's colour space names, if it names one.
[[nodiscard]] std::optional<ColorMatrix> matrix_of(holoscan::schema::ColorSpace space) noexcept;

// The output port's tensor promise: CUDA device memory, `format`'s element
// type and rank when it is known, at most `max_bytes`.
[[nodiscard]] holoscan::TensorOutputSpec output_spec(std::optional<PixelFormat> format,
                                                     std::uint64_t max_bytes);

// One operator's conversion of each received image.
struct Target {
  // Without one, the input's format.
  std::optional<PixelFormat> format;
  // Without one, the input's size.
  std::optional<int> width;
  std::optional<int> height;
  ConversionParams params;
  // A YCbCr input whose colour space names a matrix is converted with it.
  bool use_input_color_space{true};
  // What CUSTOM-encoded input holds.
  std::optional<PixelFormat> custom_format;
  // The output port's capacity, which a larger image exceeds.
  std::uint64_t max_bytes{};
  // Whether the image's ROI offsets carry over (a same-size conversion).
  bool keep_roi{};
  // Told the backend of the first frame and of every change.
  std::function<void(Backend)> on_backend;
};

class ConversionRunner {
 public:
  // @throws std::invalid_argument if `backend` is not compiled in.
  ConversionRunner(std::string op_name, Backend backend);
  ~ConversionRunner();
  ConversionRunner(const ConversionRunner&) = delete;
  ConversionRunner& operator=(const ConversionRunner&) = delete;

  // Receives one image from `input`, converts it as `target` says and emits
  // it on `output`.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> run(
      holoscan::ExecutionContext& context, holoscan::Input<holoscan::schema::ImageT>& input,
      holoscan::Output<holoscan::schema::ImageT>& output, const Target& target);

  // Frees the staging buffer and forgets the logged choices.
  void reset() noexcept;

 private:
  [[nodiscard]] holoscan::expected<void*, holoscan::Error> staging(std::size_t bytes,
                                                                   cudaStream_t stream);

  std::string name_;
  Converter converter_;
  std::optional<Backend> logged_backend_;
  bool copy_logged_{};
  void* staging_{};
  std::size_t staging_bytes_{};
};

}  // namespace holoscan::color_convert::detail
