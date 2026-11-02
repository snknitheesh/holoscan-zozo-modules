// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/color_convert/color_convert_op.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/color_convert/converter.hpp>
#include <holoscan/core/payload_options.hpp>

#include "op_support.hpp"

namespace holoscan::color_convert {

void validate_config(const ColorConvertConfig& config) {
  check_available(config.backend);
  if (config.max_width < 1 || config.max_height < 1 || config.max_width > kMaxImageSide ||
      config.max_height > kMaxImageSide) {
    throw std::invalid_argument("max_width and max_height must be 1.." +
                                std::to_string(kMaxImageSide));
  }
  if (config.custom_format && *config.custom_format != PixelFormat::kNv16) {
    throw std::invalid_argument(
        "custom_format names what CUSTOM images hold, which only nv16 needs: every other "
        "format has its own encoding");
  }
}

ColorConvertOp::ColorConvertOp(ColorConvertConfig config) : config_(std::move(config)) {
  validate_config(config_);
  runner_ = std::make_unique<detail::ConversionRunner>("ColorConvertOp", config_.backend);
}

ColorConvertOp::~ColorConvertOp() = default;

void ColorConvertOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots: one image in compute() while the next one arrives.
  spec.input(image, "image").queue_depth(2U);
  spec.output(converted, "converted")
      .produces_tensor(detail::output_spec(
          config_.output_format,
          packed_size(config_.output_format, config_.max_width, config_.max_height)))
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract ColorConvertOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void ColorConvertOp::stop() { runner_->reset(); }

holoscan::expected<void, holoscan::Error> ColorConvertOp::compute(
    holoscan::ExecutionContext& context) {
  detail::Target target{
      .format = config_.output_format,
      .params =
          ConversionParams{
              .matrix = config_.matrix, .range = config_.range, .mode = ResizeMode::kNearest},
      .use_input_color_space = config_.use_input_color_space,
      .custom_format = config_.custom_format,
      .max_bytes = packed_size(config_.output_format, config_.max_width, config_.max_height),
      .keep_roi = true,
      .on_backend = config_.on_backend,
  };
  return runner_->run(context, image, converted, target);
}

}  // namespace holoscan::color_convert
