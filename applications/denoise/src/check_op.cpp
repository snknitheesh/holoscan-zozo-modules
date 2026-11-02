// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "check_op.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace denoise_demo {
namespace {

namespace dn = holoscan::denoise;
using std::chrono_literals::operator""ns;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "CheckOp: " + message};
}

// Copies an RGB8 image of `width` x `height` to the host.
holoscan::expected<void, holoscan::Error> download(const holoscan::schema::ImageT& image, int width,
                                                   int height, std::vector<std::uint8_t>& out,
                                                   holoscan::ExecutionContext& context) {
  if (!image.data || image.encoding != holoscan::schema::ImageEncoding_RGB8 ||
      image.width != width || image.height != height) {
    return holoscan::make_unexpected(invalid("expected RGB8 images of the scene's size"));
  }
  out.resize(static_cast<std::size_t>(width) * height * 3U);
  return image.data->copy_to_host(out.data(), out.size(), context.cuda_stream());
}

// Pixels this close to a region's edge or the image's are left out of the
// fusion checks: there the blend's halo, where the window meets the mid-tones
// 13 stops darker, may clip.
constexpr int kMargin = 16;

bool inside(const dn::HdrScene& hdr, int x, int y, dn::HdrRegion region) {
  return x >= kMargin && y >= kMargin && x + kMargin < hdr.width && y + kMargin < hdr.height &&
         dn::hdr_region(hdr, x) == region && dn::hdr_region(hdr, x - kMargin) == region &&
         dn::hdr_region(hdr, x + kMargin) == region;
}

// The standard deviation of the gray of `image` over `region`.
double region_spread(const dn::HdrScene& hdr, const std::vector<std::uint8_t>& image,
                     dn::HdrRegion region) {
  double sum = 0.0;
  double squares = 0.0;
  std::size_t count = 0;
  for (int y = 0; y < hdr.height; ++y) {
    for (int x = 0; x < hdr.width; ++x) {
      if (!inside(hdr, x, y, region)) continue;
      const std::uint8_t* p = image.data() + (static_cast<std::size_t>(y) * hdr.width + x) * 3U;
      const double v = (p[0] + p[1] + p[2]) / 3.0;
      sum += v;
      squares += v * v;
      ++count;
    }
  }
  if (count == 0) return 0.0;
  const double mean = sum / static_cast<double>(count);
  return std::sqrt(std::max(0.0, squares / static_cast<double>(count) - mean * mean));
}

}  // namespace

CheckOp::CheckOp(CheckConfig config) : config_(std::move(config)) {
  if (!config_.report || !config_.finished) {
    throw std::invalid_argument("CheckOp needs a report and a frame counter");
  }
}

void CheckOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(noisy, "noisy").queue_depth(2U);
  spec.input(denoised, "denoised").queue_depth(2U);
  spec.input(fused, "fused").queue_depth(2U);
}

holoscan::Contract CheckOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{noisy, denoised, fused}},
      .window = 0ns});
  result.input(noisy, holoscan::Exact{});
  result.input(denoised, holoscan::Exact{});
  result.input(fused, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> CheckOp::compute(holoscan::ExecutionContext& context) {
  auto noisy_sample = noisy.receive();
  if (!noisy_sample) return holoscan::make_unexpected(std::move(noisy_sample).error());
  auto denoised_sample = denoised.receive();
  if (!denoised_sample) return holoscan::make_unexpected(std::move(denoised_sample).error());
  auto fused_sample = fused.receive();
  if (!fused_sample) return holoscan::make_unexpected(std::move(fused_sample).error());
  const dn::LowLightScene& scene = config_.scene;
  for (auto [image, out] :
       {std::pair{&noisy_sample->data, &noisy_}, std::pair{&denoised_sample->data, &denoised_},
        std::pair{&fused_sample->data, &fused_}}) {
    if (auto ok = download(*image, scene.width, scene.height, *out, context); !ok) {
      return ok;
    }
  }
  Report& r = *config_.report;
  const auto frame_id = noisy_sample->metadata.frame_id;
  if (frame_id != r.frames + 1U) ++r.frame_gaps;
  const int frame = static_cast<int>(frame_id) - 1;

  // Denoising, once the sequence has settled.
  const int since_start =
      scene.cut_frame >= 0 && frame >= scene.cut_frame ? frame - scene.cut_frame : frame;
  const auto clean = dn::render_clean(scene, frame);
  if (since_start >= static_cast<int>(config_.settle_frames)) {
    const double before = dn::psnr(noisy_.data(), clean.data(), clean.size());
    const double after = dn::psnr(denoised_.data(), clean.data(), clean.size());
    ++r.checked;
    r.noisy_psnr_sum += before;
    r.denoised_psnr_sum += after;
    r.worst_gain = std::min(r.worst_gain, after - before);
  }
  // The trail behind the disc.
  if (since_start >= 4) {
    for (int y = 0; y < scene.height; ++y) {
      for (int x = 0; x < scene.width; ++x) {
        if (dn::object_covers(scene, frame, x, y)) continue;
        bool trail = false;
        for (int back = 1; back <= 4 && !trail; ++back) {
          trail = dn::object_covers(scene, frame - back, x, y);
        }
        if (!trail) continue;
        for (int c = 0; c < 3; ++c) {
          const std::size_t i = (static_cast<std::size_t>(y) * scene.width + x) * 3U + c;
          const double out = static_cast<double>(denoised_[i]) - clean[i];
          const double in = static_cast<double>(noisy_[i]) - clean[i];
          r.trail_squared += out * out;
          r.trail_input_squared += in * in;
          ++r.trail_samples;
        }
      }
    }
  }
  // Fusion.
  const dn::HdrScene& hdr = config_.hdr;
  for (int y = 0; y < hdr.height; ++y) {
    for (int x = 0; x < hdr.width; ++x) {
      const bool window = inside(hdr, x, y, dn::HdrRegion::kWindow);
      const bool shadow = inside(hdr, x, y, dn::HdrRegion::kShadow);
      for (int c = 0; c < 3; ++c) {
        const std::uint8_t v = fused_[(static_cast<std::size_t>(y) * hdr.width + x) * 3U + c];
        r.fused_clipped += window && v >= 254 ? 1U : 0U;
        r.fused_crushed += shadow && v <= 1 ? 1U : 0U;
      }
    }
  }
  r.worst_shadow_texture =
      std::min(r.worst_shadow_texture, region_spread(hdr, fused_, dn::HdrRegion::kShadow));
  r.worst_window_texture =
      std::min(r.worst_window_texture, region_spread(hdr, fused_, dn::HdrRegion::kWindow));

  ++r.frames;
  config_.finished->fetch_add(1U, std::memory_order_release);
  if (r.frames == config_.expected_frames) {
    context.request_completion();
  }
  return {};
}

}  // namespace denoise_demo
