// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "render_check_op.hpp"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

namespace image_quality_app {
namespace {

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "RenderCheckOp: " + std::move(message)};
}

// The overlay's magenta blended over the smudge's haze, about (232, 92, 232): brighter
// red and blue than any scene color (at most 215) has, with little green.
bool magenta(const std::uint8_t* p) { return p[0] >= 220 && p[2] >= 220 && p[1] <= 130; }

}  // namespace

RenderCheckOp::RenderCheckOp(RenderCheckConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("RenderCheckOp needs a report");
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

holoscan::expected<void, holoscan::Error> RenderCheckOp::compute(holoscan::ExecutionContext&) {
  auto sample = frame.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  if (!sample->data.data) {
    return holoscan::make_unexpected(failure("frame has no tensor"));
  }
  const holoscan::Tensor& tensor = *sample->data.data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (shape.size() != 3 || shape[2] != 4 || tensor.dtype().bits != 8) {
    return holoscan::make_unexpected(failure("frames must be RGBA8"));
  }
  const int height = static_cast<int>(shape[0]);
  const int width = static_cast<int>(shape[1]);
  pixels_.resize(static_cast<std::size_t>(width) * height * 4U);
  if (auto ready = tensor.synchronize_for_host(); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  if (const cudaError_t status =
          cudaMemcpy(pixels_.data(), tensor.data(), pixels_.size(), cudaMemcpyDefault);
      status != cudaSuccess) {
    return holoscan::make_unexpected(failure(cudaGetErrorString(status)));
  }

  RenderReport& report = *config_.report;
  ++report.frames;
  const holoscan::image_quality::Smudge& s = config_.smudge;
  std::uint64_t inside = 0;
  std::uint64_t inside_magenta = 0;
  std::uint64_t outside = 0;
  std::uint64_t outside_magenta = 0;
  std::uint64_t white = 0;
  std::uint64_t distinct = 0;
  std::uint8_t previous = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::uint8_t* p = pixels_.data() + (static_cast<std::size_t>(y) * width + x) * 4U;
      // The core of the smudge, where every overlay tile lies inside it.
      const double dx = (x - s.cx) / (0.6 * s.rx);
      const double dy = (y - s.cy) / (0.6 * s.ry);
      const bool core = dx * dx + dy * dy <= 1.0;
      const double ox = (x - s.cx) / (1.2 * s.rx);
      const double oy = (y - s.cy) / (1.2 * s.ry);
      const bool far = ox * ox + oy * oy > 1.0;
      if (core) {
        ++inside;
        inside_magenta += magenta(p) ? 1U : 0U;
      } else if (far) {
        ++outside;
        outside_magenta += magenta(p) ? 1U : 0U;
      }
      white += p[0] >= 250 && p[1] >= 250 && p[2] >= 250 ? 1U : 0U;
      distinct += p[1] != previous ? 1U : 0U;
      previous = p[1];
    }
  }
  const double pixels = static_cast<double>(width) * height;
  report.frames_with_scene += static_cast<double>(distinct) > 0.01 * pixels ? 1U : 0U;
  report.frames_with_text += white > 50 ? 1U : 0U;
  const double fraction = inside == 0 ? 0.0 : static_cast<double>(inside_magenta) / inside;
  const double stray = outside == 0 ? 0.0 : static_cast<double>(outside_magenta) / outside;
  if (fraction > 0.5 && stray < 0.02) {
    ++report.frames_with_overlay;
  }
  if (fraction > report.best_fraction || report.best.empty()) {
    report.best_fraction = fraction;
    report.best = pixels_;
    report.width = width;
    report.height = height;
  }
  return {};
}

void write_ppm(const char* path, const std::vector<std::uint8_t>& pixels, int width, int height) {
  std::unique_ptr<std::FILE, int (*)(std::FILE*)> file(std::fopen(path, "wb"), &std::fclose);
  if (!file) {
    throw std::runtime_error(std::string("cannot write ") + path);
  }
  std::fprintf(file.get(), "P6\n%d %d\n255\n", width, height);
  for (std::size_t i = 0; i + 4 <= pixels.size(); i += 4) {
    if (std::fwrite(pixels.data() + i, 1, 3, file.get()) != 3) {
      throw std::runtime_error(std::string("cannot write ") + path);
    }
  }
}

}  // namespace image_quality_app
