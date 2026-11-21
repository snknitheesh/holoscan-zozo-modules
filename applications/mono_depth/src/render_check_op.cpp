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

#include <holoscan/logger/logger.hpp>

#include "image_files.hpp"

namespace mono_depth_app {

RenderCheckOp::RenderCheckOp(RenderCheckConfig config) : config_(std::move(config)) {
  if (!config_.check || config_.width < 1 || config_.height < 1) {
    throw std::invalid_argument("RenderCheckOp needs a check and a positive size");
  }
}

void RenderCheckOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots: one frame in compute() while the next one arrives. HoloViz's edge to it
  // keeps only the newest frame (edges.hpp), which sets that edge's own depth.
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
  if (!copy_logged_) {
    copy_logged_ = true;
    HOLOSCAN_LOG_INFO(
        "RenderCheckOp: each rendered frame is copied to the host, which waits for "
        "the copy");
  }
  if (auto copied = image.data->copy_to_host(pixels_.data(), bytes, context.cuda_stream());
      !copied) {
    return copied;
  }
  RenderCheck& check = *config_.check;
  if (config_.expected) {
    // The fraction of pixels off in a channel, against the frame expected for
    // `id`.
    const auto mismatch = [&](std::uint64_t id) -> holoscan::expected<double, holoscan::Error> {
      auto found = expected_.find(id);
      if (found == expected_.end()) {
        expected_.clear();
        found = expected_.emplace(id, config_.expected(id)).first;
      }
      const std::vector<std::uint8_t>& expected = found->second;
      if (expected.size() != bytes) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument, "RenderCheckOp: the expected frame's size"});
      }
      std::size_t mismatched = 0;
      for (std::size_t p = 0; p < bytes; p += 4) {
        for (std::size_t c = 0; c < 3; ++c) {
          if (std::abs(static_cast<int>(pixels_[p + c]) - static_cast<int>(expected[p + c])) >
              kChannelTolerance) {
            ++mismatched;
            break;
          }
        }
      }
      return static_cast<double>(mismatched) / static_cast<double>(bytes / 4U);
    };
    // The first id that matches, or else the best: the frame's own when it
    // carries one.
    const std::uint64_t own = sample->metadata.frame_id;
    const std::uint64_t first = own != 0 ? own : next_id_;
    const std::uint64_t last = own != 0 ? own : std::max(next_id_, config_.last_id);
    double best = 1.0;
    std::uint64_t best_id = first;
    for (std::uint64_t id = first; id <= last; ++id) {
      auto fraction = mismatch(id);
      if (!fraction) {
        return holoscan::make_unexpected(std::move(fraction).error());
      }
      if (*fraction < best) {
        best = *fraction;
        best_id = id;
      }
      if (best <= kMaxMismatchedFraction) {
        break;
      }
    }
    ++check.checked;
    check.worst_fraction = std::max(check.worst_fraction, best);
    if (best > kMaxMismatchedFraction) {
      ++check.failed;
    } else {
      next_id_ = best_id;
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

}  // namespace mono_depth_app
