// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "render_check_op.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "image_files.hpp"

namespace sam3_app {

RenderCheckOp::RenderCheckOp(RenderCheckConfig config) : config_(std::move(config)) {
  if (!config_.check || config_.width < 1 || config_.height < 1) {
    throw std::invalid_argument("RenderCheckOp needs a check and a positive size");
  }
  if (!config_.expected.empty() &&
      config_.expected.size() != static_cast<std::size_t>(config_.width) * config_.height * 4U) {
    throw std::invalid_argument("RenderCheckOp: the expected frame does not have the size");
  }
}

void RenderCheckOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(frame, "frame").queue_depth(2U);
}

holoscan::Contract RenderCheckOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{frame});
  return result;
}

holoscan::expected<void, holoscan::Error> RenderCheckOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = frame.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  const holoscan::schema::ImageT& image = sample->data;
  const std::size_t bytes = static_cast<std::size_t>(config_.width) * config_.height * 4U;
  if (!image.data || image.encoding != holoscan::schema::ImageEncoding_RGBA8 ||
      static_cast<std::size_t>(image.data->nbytes()) != bytes) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                     "RenderCheckOp: expected an RGBA8 frame of " +
                                                         std::to_string(config_.width) + "x" +
                                                         std::to_string(config_.height)});
  }
  pixels_.resize(bytes);
  if (auto copied = image.data->copy_to_host(pixels_.data(), bytes, context.cuda_stream());
      !copied) {
    return copied;
  }
  RenderCheck& check = *config_.check;
  if (!config_.expected.empty()) {
    std::size_t mismatched = 0;
    for (std::size_t p = 0; p < bytes; p += 4) {
      for (std::size_t c = 0; c < 3; ++c) {
        if (std::abs(static_cast<int>(pixels_[p + c]) - static_cast<int>(config_.expected[p + c])) >
            kChannelTolerance) {
          ++mismatched;
          break;
        }
      }
    }
    const double fraction = static_cast<double>(mismatched) / static_cast<double>(bytes / 4U);
    ++check.checked;
    check.worst_fraction = std::max(check.worst_fraction, fraction);
    if (fraction > config_.max_mismatched_fraction) {
      ++check.failed;
    }
  }
  if (!config_.write.empty()) {
    try {
      write_ppm(config_.write, config_.width, config_.height,
                to_rgb(pixels_.data(), static_cast<std::size_t>(config_.width) * 4U, config_.width,
                       config_.height, PixelFormat::kRgba8));
      check.written = true;
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, error.what()});
    }
  }
  check.frames.fetch_add(1U);
  return {};
}

}  // namespace sam3_app
