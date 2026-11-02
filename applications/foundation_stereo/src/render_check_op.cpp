// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "render_check_op.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>

namespace foundation_stereo_app {
namespace {

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "RenderCheckOp: " + message};
}

}  // namespace

RenderCheckOp::RenderCheckOp(RenderCheckConfig config) : config_(std::move(config)) {
  if (!config_.check || config_.width < 2 || config_.height < 1) {
    throw std::invalid_argument("RenderCheckOp needs a check and a frame size");
  }
}

void RenderCheckOp::setup(holoscan::OperatorSpec& spec) { spec.input(frame, "frame"); }

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
  if (!image.data || image.encoding != holoscan::schema::ImageEncoding_RGBA8) {
    return holoscan::make_unexpected(invalid("the frame must be an RGBA8 image"));
  }
  const holoscan::Tensor& tensor = *image.data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (shape.size() != 3 || shape[0] != config_.height || shape[1] != config_.width ||
      shape[2] != 4) {
    return holoscan::make_unexpected(invalid("the frame is not the configured size"));
  }
  pixels_.resize(static_cast<std::size_t>(config_.width) * config_.height * 4U);
  if (auto copied = tensor.copy_to_host(pixels_.data(), pixels_.size(), context.cuda_stream());
      !copied) {
    return copied;
  }
  RenderCheck& check = *config_.check;
  ++check.frames;
  const int half = config_.width / 2;
  std::uint64_t colored = 0;
  std::uint64_t right_pixels = 0;
  double brightness = 0;
  std::uint64_t left_pixels = 0;
  for (int y = 0; y < config_.height; ++y) {
    for (int x = 0; x < config_.width; ++x) {
      const std::uint8_t* p =
          pixels_.data() + (static_cast<std::size_t>(y) * config_.width + x) * 4U;
      if (config_.left_image && x < half) {
        brightness += (p[0] + p[1] + p[2]) / 3.0;
        ++left_pixels;
      } else if (!config_.left_image || x >= half) {
        const int spread = std::max({p[0], p[1], p[2]}) - std::min({p[0], p[1], p[2]});
        colored += spread > kColoredSpread ? 1U : 0U;
        ++right_pixels;
      }
    }
  }
  check.colored = right_pixels == 0 ? 0.0 : static_cast<double>(colored) / right_pixels;
  check.left_brightness = left_pixels == 0 ? 0.0 : brightness / static_cast<double>(left_pixels);
  if (check.colored < kMinColoredShare ||
      (config_.left_image && check.left_brightness < kMinLeftBrightness)) {
    ++check.failed;
  }
  if (!config_.write.empty()) {
    std::ofstream out(config_.write, std::ios::binary | std::ios::trunc);
    out << "P6\n" << config_.width << ' ' << config_.height << "\n255\n";
    for (std::size_t i = 0; i < pixels_.size(); i += 4) {
      out.write(reinterpret_cast<const char*>(pixels_.data() + i), 3);
    }
    check.written = static_cast<bool>(out);
  }
  return {};
}

}  // namespace foundation_stereo_app
