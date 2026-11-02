// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "render_check_op.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/yolo26/synthetic_scene.hpp>

#include "image_files.hpp"

namespace yolo26_app {
namespace {

using Rgb = std::array<int, 3>;

// The overlays' colors of classes 0 and 1 (Yolo26OverlayOp's palette).
constexpr std::array<Rgb, 2> kClassColors{{{230, 25, 75}, {60, 180, 75}}};
// How far HoloViz's blending and rounding may move a color, per channel.
constexpr int kTolerance = 4;
// How far a drawn box edge may lie from its rectangle's, in pixels.
constexpr int kEdgeSlack = 3;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "RenderCheckOp: " + message};
}

bool near(const std::uint8_t* rgba, const Rgb& want) {
  for (int c = 0; c < 3; ++c) {
    if (std::abs(static_cast<int>(rgba[c]) - want[static_cast<std::size_t>(c)]) > kTolerance) {
      return false;
    }
  }
  return true;
}

}  // namespace

RenderCheckOp::RenderCheckOp(RenderCheckConfig config) : config_(std::move(config)) {
  if (!config_.report || config_.width < 1 || config_.height < 1) {
    throw std::invalid_argument("RenderCheckOp needs a report and a positive size");
  }
}

void RenderCheckOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(rendered, "rendered").queue_depth(2U);
}

holoscan::Contract RenderCheckOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{rendered});
  return result;
}

holoscan::expected<void, holoscan::Error> RenderCheckOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = rendered.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  const int width = config_.width;
  const int height = config_.height;
  if (!sample->data.data) {
    return holoscan::make_unexpected(invalid("the render has no tensor"));
  }
  const holoscan::Tensor& tensor = *sample->data.data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (shape.size() != 3 || shape[0] != height || shape[1] != width || shape[2] != 4 ||
      !tensor.is_contiguous()) {
    return holoscan::make_unexpected(invalid("the render must be RGBA8 of the frames' size"));
  }
  pixels_.resize(static_cast<std::size_t>(width) * height * 4U);
  if (auto copied = tensor.copy_to_host(pixels_.data(), pixels_.size(), context.cuda_stream());
      !copied) {
    return copied;
  }
  RenderReport& report = *config_.report;
  const std::uint32_t index = report.renders.load();
  if (!config_.image_dir.empty()) {
    std::vector<std::uint8_t> rgb;
    rgb.reserve(static_cast<std::size_t>(width) * height * 3U);
    for (std::size_t i = 0; i < pixels_.size(); i += 4) {
      rgb.insert(rgb.end(), pixels_.begin() + static_cast<std::ptrdiff_t>(i),
                 pixels_.begin() + static_cast<std::ptrdiff_t>(i + 3));
    }
    std::ostringstream name;
    name << "render_" << std::setw(4) << std::setfill('0') << index << ".ppm";
    try {
      write_ppm(config_.image_dir / name.str(), width, height, rgb);
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, error.what()});
    }
  }
  const auto at = [&](int x, int y) {
    return pixels_.data() + (static_cast<std::size_t>(y) * width + x) * 4U;
  };
  const double alpha = config_.masks ? config_.mask_alpha / 255.0 : 0.0;
  // Some frame's two rectangles: a box edge near each one's left edge, and
  // the center as the overlay leaves it.
  const auto shows = [&](std::uint32_t frame) {
    const holoscan::yolo26::SyntheticScene scene =
        holoscan::yolo26::make_scene(width, height, frame - 1U);
    for (std::size_t c = 0; c < kClassColors.size(); ++c) {
      const auto& rect = scene.rects[c];
      const int cy = (rect.y0 + rect.y1) / 2;
      bool edge = false;
      for (int x = std::max(rect.x0 - kEdgeSlack, 0);
           x <= std::min(rect.x0 + kEdgeSlack, width - 1) && !edge; ++x) {
        edge = near(at(x, cy), kClassColors[c]);
      }
      Rgb center{};
      for (std::size_t k = 0; k < 3; ++k) {
        center[k] =
            static_cast<int>(std::lround(rect.rgb[k] * (1.0 - alpha) + kClassColors[c][k] * alpha));
      }
      if (!edge || !near(at((rect.x0 + rect.x1) / 2, cy), center)) {
        return false;
      }
    }
    return true;
  };
  bool matched = false;
  for (std::uint32_t frame = 1; frame <= config_.frames && !matched; ++frame) {
    matched = shows(frame);
  }
  if (matched) {
    ++report.matched;
  }
  report.renders.fetch_add(1U);
  return {};
}

}  // namespace yolo26_app
