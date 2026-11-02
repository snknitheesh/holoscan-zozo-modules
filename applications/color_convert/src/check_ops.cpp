// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "check_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/color_convert/image_view.hpp>
#include <holoscan/color_convert/reference.hpp>

#include "camera_op.hpp"

namespace color_convert_app {
namespace {

namespace cc = holoscan::color_convert;
using std::chrono_literals::operator""ns;

cc::ImageView host_view(std::vector<std::uint8_t>& bytes, cc::PixelFormat format, int width,
                        int height) {
  return cc::packed_view(bytes.data(), format, width, height, cc::MemoryLocation::kHost);
}

// An image's bytes copied to the host, after checking its format and size.
holoscan::expected<std::vector<std::uint8_t>, holoscan::Error> fetch(
    const holoscan::schema::ImageT& image, std::size_t bytes, int width, int height,
    holoscan::ExecutionContext& context, const char* what) {
  if (!image.data || image.width != width || image.height != height ||
      static_cast<std::size_t>(image.data->nbytes()) != bytes) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument,
        std::string("check: the ") + what + " image is not " + std::to_string(width) + "x" +
            std::to_string(height) + " in " + std::to_string(bytes) + " packed bytes"});
  }
  std::vector<std::uint8_t> host(bytes);
  if (auto copied = image.data->copy_to_host(host.data(), host.size(), context.cuda_stream());
      !copied) {
    return holoscan::make_unexpected(std::move(copied).error());
  }
  return host;
}

bool within(const Difference& d, const Tolerance& t) {
  return t.exact ? d.wrong == 0 : (d.mean <= t.mean && d.p99 <= t.p99);
}

void keep_worst(Difference& worst, const Difference& d) {
  worst.mean = std::max(worst.mean, d.mean);
  worst.p99 = std::max(worst.p99, d.p99);
  worst.max = std::max(worst.max, d.max);
  worst.wrong = std::max(worst.wrong, d.wrong);
}

std::string describe(const Difference& d) {
  std::ostringstream out;
  out << "mean " << std::fixed << std::setprecision(3) << d.mean << " p99 " << d.p99 << " max "
      << d.max << " differing " << d.wrong;
  return out.str();
}

}  // namespace

std::vector<std::uint8_t> camera_frame(const Pipeline& p, std::uint32_t frame) {
  std::vector<std::uint8_t> rgb(cc::packed_size(cc::PixelFormat::kRgb8, p.width, p.height));
  cc::draw_test_pattern(host_view(rgb, cc::PixelFormat::kRgb8, p.width, p.height), frame);
  std::vector<std::uint8_t> camera(cc::packed_size(p.camera, p.width, p.height));
  cc::reference_convert(
      host_view(rgb, cc::PixelFormat::kRgb8, p.width, p.height),
      host_view(camera, p.camera, p.width, p.height),
      cc::ConversionParams{
          .matrix = kCameraMatrix, .range = kCameraRange, .mode = cc::ResizeMode::kNearest});
  return camera;
}

std::vector<std::uint8_t> expected_converted(const Pipeline& p, std::uint32_t frame) {
  std::vector<std::uint8_t> camera = camera_frame(p, frame);
  std::vector<std::uint8_t> out(cc::packed_size(p.converted, p.width, p.height));
  cc::reference_convert(
      host_view(camera, p.camera, p.width, p.height),
      host_view(out, p.converted, p.width, p.height),
      cc::ConversionParams{
          .matrix = kCameraMatrix, .range = kCameraRange, .mode = cc::ResizeMode::kNearest});
  return out;
}

std::vector<std::uint8_t> expected_resized(const Pipeline& p, std::uint32_t frame) {
  std::vector<std::uint8_t> camera = camera_frame(p, frame);
  std::vector<std::uint8_t> out(cc::packed_size(p.resized, p.resized_width, p.resized_height));
  cc::reference_convert(host_view(camera, p.camera, p.width, p.height),
                        host_view(out, p.resized, p.resized_width, p.resized_height),
                        p.resize_params);
  return out;
}

Difference compare(const std::vector<std::uint8_t>& got, const std::vector<std::uint8_t>& want) {
  std::array<std::uint64_t, 256> histogram{};
  double sum = 0.0;
  const std::size_t n = std::min(got.size(), want.size());
  for (std::size_t i = 0; i < n; ++i) {
    const int d = std::abs(static_cast<int>(got[i]) - static_cast<int>(want[i]));
    ++histogram[d];
    sum += d;
  }
  Difference result{.mean = n == 0 ? 0.0 : sum / static_cast<double>(n)};
  result.wrong = n - histogram[0] +
                 (got.size() > want.size() ? got.size() - want.size() : want.size() - got.size());
  std::uint64_t seen = 0;
  bool found = false;
  for (int d = 0; d < 256; ++d) {
    seen += histogram[d];
    if (!found && seen * 100 >= n * 99) {
      result.p99 = d;
      found = true;
    }
    if (histogram[d] != 0) result.max = d;
  }
  return result;
}

FrameCheckOp::FrameCheckOp(FrameCheckConfig config) : config_(std::move(config)) {}

void FrameCheckOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(converted, "converted").queue_depth(2U);
  spec.input(resized, "resized").queue_depth(2U);
}

holoscan::Contract FrameCheckOp::contract() const {
  holoscan::Contract result;
  // A frame's two outputs carry its capture time.
  result.trigger(
      holoscan::OnAligned{.ports = holoscan::PortList{{converted, resized}}, .window = 0ns});
  result.input(converted, holoscan::Exact{});
  result.input(resized, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> FrameCheckOp::compute(
    holoscan::ExecutionContext& context) {
  auto a = converted.receive();
  if (!a) return holoscan::make_unexpected(std::move(a).error());
  auto b = resized.receive();
  if (!b) return holoscan::make_unexpected(std::move(b).error());
  const Pipeline& p = config_.pipeline;
  const auto frame = static_cast<std::uint32_t>(a->metadata.frame_id - 1U);
  auto got_a = fetch(a->data, cc::packed_size(p.converted, p.width, p.height), p.width, p.height,
                     context, "converted");
  if (!got_a) return holoscan::make_unexpected(std::move(got_a).error());
  auto got_b = fetch(b->data, cc::packed_size(p.resized, p.resized_width, p.resized_height),
                     p.resized_width, p.resized_height, context, "resized");
  if (!got_b) return holoscan::make_unexpected(std::move(got_b).error());
  const Difference da = compare(*got_a, expected_converted(p, frame));
  const Difference db = compare(*got_b, expected_resized(p, frame));
  Report& report = *config_.report;
  {
    const std::lock_guard lock(report.mutex);
    keep_worst(report.converted, da);
    keep_worst(report.resized, db);
    if (!within(da, config_.converted) || !within(db, config_.resized)) {
      if (report.failed_frames++ == 0) {
        report.first_failure = "frame " + std::to_string(frame) + ": converted " + describe(da) +
                               "; resized " + describe(db);
      }
    }
  }
  report.frames.fetch_add(1);
  return {};
}

RenderCheckOp::RenderCheckOp(RenderCheckConfig config) : config_(std::move(config)) {}

void RenderCheckOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(rendered, "rendered").queue_depth(2U);
}

holoscan::Contract RenderCheckOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{rendered});
  return result;
}

namespace {

// The camera frame a render shows, from where its colour bars are: they move
// right by four pixels a frame, so the black bar's edge into the white one,
// on a row above the inset, is at four times the frame number.
std::optional<std::uint32_t> frame_shown(const std::vector<std::uint8_t>& rgba, int width,
                                         int row) {
  const auto bright = [&](int x) {
    const std::size_t i = (static_cast<std::size_t>(row) * width + x) * 4U;
    return rgba[i] > 128 && rgba[i + 1] > 128 && rgba[i + 2] > 128;
  };
  const auto dark = [&](int x) {
    const std::size_t i = (static_cast<std::size_t>(row) * width + x) * 4U;
    return rgba[i] < 128 && rgba[i + 1] < 128 && rgba[i + 2] < 128;
  };
  for (int x = 0; x < width; ++x) {
    if (bright(x) && dark((x + width - 1) % width)) {
      return static_cast<std::uint32_t>(x / 4);
    }
  }
  return std::nullopt;
}

}  // namespace

holoscan::expected<void, holoscan::Error> RenderCheckOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = rendered.receive();
  if (!sample) return holoscan::make_unexpected(std::move(sample).error());
  const Pipeline& p = config_.pipeline;
  Report& report = *config_.report;
  const std::uint32_t index = report.rendered.load();
  auto got = fetch(sample->data, cc::packed_size(cc::PixelFormat::kRgba8, p.width, p.height),
                   p.width, p.height, context, "rendered");
  if (!got) return holoscan::make_unexpected(std::move(got).error());
  if (!config_.image_dir.empty()) {
    std::ostringstream name;
    name << "render_" << std::setw(4) << std::setfill('0') << index << ".ppm";
    std::ofstream file(config_.image_dir / name.str(), std::ios::binary);
    file << "P6\n" << p.width << ' ' << p.height << "\n255\n";
    for (std::size_t i = 0; i < got->size(); i += 4) {
      file.write(reinterpret_cast<const char*>(got->data() + i), 3);
    }
  }
  const std::optional<std::uint32_t> frame = frame_shown(*got, p.width, config_.inset_y / 2);
  std::string problem;
  Difference d;
  bool inset_seen = false;
  if (!frame) {
    problem = "no colour bars on row " + std::to_string(config_.inset_y / 2);
  } else {
    // Outside the inset, the render is the converted frame (RGBA8, opaque).
    const std::vector<std::uint8_t> want = expected_converted(p, *frame);
    std::vector<std::uint8_t> outside_got;
    std::vector<std::uint8_t> outside_want;
    for (int y = 0; y < p.height; ++y) {
      for (int x = 0; x < p.width; ++x) {
        // Two pixels of margin, where HoloViz filters the inset's edge.
        const bool near_inset =
            x >= config_.inset_x - 2 && x < config_.inset_x + config_.inset_width + 2 &&
            y >= config_.inset_y - 2 && y < config_.inset_y + config_.inset_height + 2;
        if (near_inset) continue;
        const std::size_t i = (static_cast<std::size_t>(y) * p.width + x) * 4U;
        outside_got.insert(outside_got.end(), got->begin() + i, got->begin() + i + 3);
        outside_want.insert(outside_want.end(), want.begin() + i, want.begin() + i + 3);
      }
    }
    d = compare(outside_got, outside_want);
    if (d.mean > 0.5 || d.max > 2) {
      problem = "outside the inset " + describe(d);
    }
    // Inside the inset, its letterbox band (the resized image's top rows) has
    // the fill colour; a render before the first inset arrived shows the frame.
    const cc::Mapping mapping = cc::compute_mapping(p.width, p.height, p.resized_width,
                                                    p.resized_height, p.resized, p.resize_params);
    if (mapping.content.y > 4) {
      const int band = mapping.content.y * config_.inset_height / p.resized_height;
      const int y = config_.inset_y + band / 2;
      const int x = config_.inset_x + config_.inset_width / 2;
      const std::size_t i = (static_cast<std::size_t>(y) * p.width + x) * 4U;
      inset_seen = true;
      for (int c = 0; c < 3; ++c) {
        inset_seen = inset_seen && std::abs(static_cast<int>((*got)[i + c]) -
                                            static_cast<int>(p.resize_params.fill_rgb[c])) <= 3;
      }
    } else {
      inset_seen = true;  // No band to look for.
    }
  }
  {
    const std::lock_guard lock(report.mutex);
    keep_worst(report.render, d);
    report.insets_seen += inset_seen ? 1U : 0U;
    if (!problem.empty() && report.failed_renders++ == 0 && report.first_failure.empty()) {
      report.first_failure = "render " + std::to_string(index) + ": " + problem;
    }
  }
  report.rendered.fetch_add(1);
  return {};
}

}  // namespace color_convert_app
