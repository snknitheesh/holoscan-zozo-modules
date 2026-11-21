// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ColorConvertOp: an image in, the same image in another pixel format out, on
// the GPU or the VIC.

#pragma once

#include <functional>
#include <memory>
#include <optional>

#include <holoscan/color_convert/backend.hpp>
#include <holoscan/color_convert/pixel_format.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::color_convert {

namespace detail {
class ConversionRunner;
}  // namespace detail

// Largest image side the operators accept.
inline constexpr int kMaxImageSide = 16384;

struct ColorConvertConfig {
  PixelFormat output_format{PixelFormat::kRgb8};
  // The YCbCr side's matrix and range. A YCbCr input whose colour space names
  // a matrix (REC601, REC709, REC2020) is converted with that one instead,
  // unless `use_input_color_space` is false. The schema has no range, so the
  // range always comes from here.
  ColorMatrix matrix{ColorMatrix::kBt601};
  ColorRange range{ColorRange::kLimited};
  bool use_input_color_space{true};
  // kCuda, kVic (Jetson) or kAuto (the VIC when it can, else CUDA).
  Backend backend{Backend::kCuda};
  // What images with the CUSTOM encoding hold, such as kNv16, which the
  // schema has no encoding for; without it they are refused.
  std::optional<PixelFormat> custom_format;
  // The largest image converted, which sizes the output's memory pool.
  int max_width{3840};
  int max_height{2160};
  // Called on the operator's thread with the backend that converts the first
  // frame, and again whenever kAuto's choice changes with a new image size or
  // format. kAuto logs its choice too.
  std::function<void(Backend)> on_backend;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const ColorConvertConfig& config);

class ColorConvertOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit ColorConvertOp(ColorConvertConfig config);
  ~ColorConvertOp() override;
  ColorConvertOp(const ColorConvertOp&) = delete;
  ColorConvertOp& operator=(const ColorConvertOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // NV12, NV16 (CUSTOM), I420, YUYV, UYVY, RGB8, BGR8, RGBA8, BGRA8, MONO8,
  // MONO16 or RGB16, in host or CUDA device memory.
  holoscan::Input<holoscan::schema::ImageT> image;
  // The image in `output_format`, in CUDA device memory: [H, W] for MONO,
  // [H, W, C] for interleaved formats (C = 2 for YUYV and UYVY), and a packed
  // byte buffer with plane layouts for NV12, NV16 and I420. The descriptor
  // keeps the input's header, ROI, exposure and gain, and names the matrix's
  // colour space for YCbCr output.
  holoscan::Output<holoscan::schema::ImageT> converted;

 private:
  ColorConvertConfig config_;
  std::unique_ptr<detail::ConversionRunner> runner_;
};

}  // namespace holoscan::color_convert
