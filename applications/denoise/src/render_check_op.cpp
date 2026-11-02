// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "render_check_op.hpp"

#include <cstddef>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <utility>

namespace denoise_demo {

RenderCheckOp::RenderCheckOp(std::shared_ptr<RenderTally> tally) : tally_(std::move(tally)) {
  if (!tally_) {
    throw std::invalid_argument("RenderCheckOp needs a tally");
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
  if (!sample->data.data) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                     "RenderCheckOp: a frame without pixels"});
  }
  const holoscan::Tensor& tensor = *sample->data.data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (shape.size() != 3 || shape[2] != 4) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument, "RenderCheckOp: expected an RGBA8 [H, W, 4] frame"});
  }
  const auto height = static_cast<std::size_t>(shape[0]);
  const auto width = static_cast<std::size_t>(shape[1]);
  pixels_.resize(height * width * 4U);
  if (auto copied = tensor.copy_to_host(pixels_.data(), pixels_.size(), context.cuda_stream());
      !copied) {
    return copied;
  }
  // The top two thirds hold the full-size frames; the roughness of each half is
  // the mean step between horizontal neighbours' green.
  const std::size_t top = height * 2 / 3;
  double rough[2] = {};
  std::size_t lit_below = 0;
  for (std::size_t v = 0; v < height; ++v) {
    for (std::size_t u = 1; u < width; ++u) {
      const std::uint8_t* p = pixels_.data() + 4U * (v * width + u);
      if (v < top) {
        if (u == width / 2) continue;
        rough[u < width / 2 ? 0 : 1] += std::abs(p[1] - p[1 - 4]);
      } else {
        lit_below += p[0] + p[1] + p[2] > 30 ? 1U : 0U;
      }
    }
  }
  const std::size_t below = (height - top) * width;
  ++tally_->frames;
  tally_->showing +=
      rough[1] > 0.0 && rough[0] > 1.5 * rough[1] && 2U * lit_below > below ? 1U : 0U;
  return {};
}

}  // namespace denoise_demo
