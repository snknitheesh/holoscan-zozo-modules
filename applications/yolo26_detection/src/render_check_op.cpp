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
// How far from a rectangle's edge the video is sampled, in pixels: past the
// 2 px box line wherever kEdgeSlack puts it, and short of the 7 or more pixels
// a rectangle moves from one frame to the next.
constexpr int kVideoProbe = 5;
// The synthetic frames' background.
constexpr Rgb kBackground{40, 40, 40};

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

// `color` blended with `over` at opacity `alpha`, as HoloViz blends a layer.
Rgb blend(const Rgb& color, const Rgb& over, double alpha) {
  Rgb out{};
  for (std::size_t k = 0; k < 3; ++k) {
    out[k] = static_cast<int>(std::lround(color[k] * (1.0 - alpha) + over[k] * alpha));
  }
  return out;
}

}  // namespace

bool render_shows(std::span<const std::uint8_t> rgba, int width, int height, std::uint32_t frame,
                  bool masks, std::uint8_t mask_alpha) {
  if (frame == 0 || rgba.size() != static_cast<std::size_t>(width) * height * 4U) {
    return false;
  }
  const auto at = [&](int x, int y) {
    return rgba.data() + (static_cast<std::size_t>(y) * width + x) * 4U;
  };
  const double alpha = masks ? mask_alpha / 255.0 : 0.0;
  const holoscan::yolo26::SyntheticScene scene =
      holoscan::yolo26::make_scene(width, height, frame - 1U);
  for (std::size_t c = 0; c < kClassColors.size(); ++c) {
    const auto& rect = scene.rects[c];
    const int cy = (rect.y0 + rect.y1) / 2;
    // A box edge near the rectangle's left edge.
    bool edge = false;
    for (int x = std::max(rect.x0 - kEdgeSlack, 0);
         x <= std::min(rect.x0 + kEdgeSlack, width - 1) && !edge; ++x) {
      edge = near(at(x, cy), kClassColors[c]);
    }
    // At the center, the frame's color under the mask.
    const Rgb color{rect.rgb[0], rect.rgb[1], rect.rgb[2]};
    if (!edge || !near(at((rect.x0 + rect.x1) / 2, cy), blend(color, kClassColors[c], alpha))) {
      return false;
    }
    // Next to the edges, the same frame's video: its color inside, the
    // background outside. A mask's edge can be a few pixels off its
    // rectangle's, so either side may or may not be under the mask.
    const auto video = [&](int x, const Rgb& want) {
      return x < 0 || x >= width || near(at(x, cy), want) ||
             near(at(x, cy), blend(want, kClassColors[c], alpha));
    };
    const int last = rect.x1 - 1;
    if (!video(rect.x0 + kVideoProbe, color) || !video(last - kVideoProbe, color) ||
        !video(rect.x0 - kVideoProbe, kBackground) || !video(last + kVideoProbe, kBackground)) {
      return false;
    }
  }
  return true;
}

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
  bool matched = false;
  for (std::uint32_t frame = 1; frame <= config_.frames && !matched; ++frame) {
    matched = render_shows(pixels_, width, height, frame, config_.masks, config_.mask_alpha);
  }
  if (matched) {
    ++report.matched;
  }
  report.renders.fetch_add(1U);
  return {};
}

}  // namespace yolo26_app
