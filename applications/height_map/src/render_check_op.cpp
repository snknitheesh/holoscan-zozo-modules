// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "render_check_op.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

namespace height_map_app {

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
  std::size_t lit[2] = {0, 0};
  for (std::size_t v = 0; v < height; ++v) {
    for (std::size_t u = 0; u < width; ++u) {
      const std::uint8_t* p = pixels_.data() + 4U * (v * width + u);
      lit[u < width / 2 ? 0 : 1] += p[0] != 0U || p[1] != 0U || p[2] != 0U ? 1U : 0U;
    }
  }
  const std::size_t half = height * (width / 2);
  ++tally_->frames;
  tally_->showing += 2U * lit[0] > half && 2U * lit[1] > half ? 1U : 0U;
  return {};
}

}  // namespace height_map_app
