// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/color_convert/resize_op.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/color_convert/converter.hpp>
#include <holoscan/core/payload_options.hpp>

#include "op_support.hpp"

namespace holoscan::color_convert {
namespace {

// The output's capacity: its size in its format, or in the widest format
// when it keeps each input's.
std::uint64_t max_bytes(const ResizeConfig& config) {
  if (config.output_format) {
    return packed_size(*config.output_format, config.width, config.height);
  }
  std::uint64_t most = 0;
  for (const PixelFormat format : kAllFormats) {
    most = std::max<std::uint64_t>(most, packed_size(format, config.width, config.height));
  }
  return most;
}

}  // namespace

void validate_config(const ResizeConfig& config) {
  check_available(config.backend);
  if (config.width < 1 || config.height < 1 || config.width > kMaxImageSide ||
      config.height > kMaxImageSide) {
    throw std::invalid_argument("width and height must be 1.." + std::to_string(kMaxImageSide));
  }
  if (config.crop && (config.crop->width < 1 || config.crop->height < 1 || config.crop->x < 0 ||
                      config.crop->y < 0)) {
    throw std::invalid_argument("crop needs a nonnegative offset and a positive size");
  }
  if (config.output_format && needs_even_width(*config.output_format) && config.width % 2 != 0) {
    throw std::invalid_argument(std::string(format_name(*config.output_format)) +
                                " output needs an even width");
  }
  if (config.custom_format && *config.custom_format != PixelFormat::kNv16) {
    throw std::invalid_argument(
        "custom_format names what CUSTOM images hold, which only nv16 needs: every other "
        "format has its own encoding");
  }
}

ResizeOp::ResizeOp(ResizeConfig config) : config_(std::move(config)) {
  validate_config(config_);
  runner_ = std::make_unique<detail::ConversionRunner>("ResizeOp", config_.backend);
}

ResizeOp::~ResizeOp() = default;

void ResizeOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(image, "image").queue_depth(2U);
  spec.output(resized, "resized")
      .produces_tensor(detail::output_spec(config_.output_format, max_bytes(config_)))
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract ResizeOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void ResizeOp::stop() { runner_->reset(); }

holoscan::expected<void, holoscan::Error> ResizeOp::compute(holoscan::ExecutionContext& context) {
  detail::Target target{
      .format = config_.output_format,
      .width = config_.width,
      .height = config_.height,
      .params = ConversionParams{.matrix = config_.matrix,
                                 .range = config_.range,
                                 .mode = config_.mode,
                                 .crop = config_.crop,
                                 .letterbox = config_.letterbox,
                                 .fill_rgb = config_.fill_rgb},
      .use_input_color_space = config_.use_input_color_space,
      .custom_format = config_.custom_format,
      .max_bytes = max_bytes(config_),
      .keep_roi = false,
      .on_backend = config_.on_backend,
  };
  return runner_->run(context, image, resized, target);
}

}  // namespace holoscan::color_convert
