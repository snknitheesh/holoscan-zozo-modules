// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ResizeOp: an image in, a resized (and optionally cropped, letterboxed and
// converted) image out, in one pass on the GPU or the VIC.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include <holoscan/color_convert/backend.hpp>
#include <holoscan/color_convert/color_convert_op.hpp>
#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/color_convert/pixel_format.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::color_convert {

struct ResizeConfig {
  // The output size, 1..kMaxImageSide.
  int width{640};
  int height{480};
  ResizeMode mode{ResizeMode::kBilinear};
  // The part of the input to resize; the whole input without one. It must
  // lie inside every input image.
  std::optional<Rect> crop;
  // Keep the aspect ratio: fit the input (or crop) inside the output,
  // centered, and fill the rest with `fill_rgb`.
  bool letterbox{false};
  std::array<std::uint8_t, 3> fill_rgb{0, 0, 0};
  // Convert to this format in the same pass; without one, the output keeps
  // each input's format.
  std::optional<PixelFormat> output_format;
  // As in ColorConvertConfig.
  ColorMatrix matrix{ColorMatrix::kBt601};
  ColorRange range{ColorRange::kLimited};
  bool use_input_color_space{true};
  Backend backend{Backend::kCuda};
  std::optional<PixelFormat> custom_format;
  std::function<void(Backend)> on_backend;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const ResizeConfig& config);

class ResizeOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit ResizeOp(ResizeConfig config);
  ~ResizeOp() override;
  ResizeOp(const ResizeOp&) = delete;
  ResizeOp& operator=(const ResizeOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // As ColorConvertOp::image.
  holoscan::Input<holoscan::schema::ImageT> image;
  // The `width` x `height` image, in `output_format` or the input's, laid out
  // as ColorConvertOp::converted. Its ROI offsets are reset to 0.
  holoscan::Output<holoscan::schema::ImageT> resized;

 private:
  ResizeConfig config_;
  std::unique_ptr<detail::ConversionRunner> runner_;
};

}  // namespace holoscan::color_convert
