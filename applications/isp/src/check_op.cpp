// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "check_op.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace isp_demo {
namespace {

using std::chrono_literals::operator""ns;
namespace isp = holoscan::isp;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "CheckOp: " + message};
}

}  // namespace

CheckOp::CheckOp(CheckConfig config)
    : config_(std::move(config)), lut_(isp::tone_lut(config_.params)) {
  if (!config_.report || !config_.finished) {
    throw std::invalid_argument("CheckOp needs a report and a frame counter");
  }
}

void CheckOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(image, "image").queue_depth(2U);
  spec.input(stats, "stats").queue_depth(2U);
  spec.input(histogram, "histogram").queue_depth(2U);
}

holoscan::Contract CheckOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{image, stats, histogram}},
      .window = 0ns});
  result.input(image, holoscan::Exact{});
  result.input(stats, holoscan::Exact{});
  result.input(histogram, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> CheckOp::compute(holoscan::ExecutionContext& context) {
  auto picture = image.receive();
  if (!picture) {
    return holoscan::make_unexpected(std::move(picture).error());
  }
  auto values = stats.receive();
  if (!values) {
    return holoscan::make_unexpected(std::move(values).error());
  }
  auto bins = histogram.receive();
  if (!bins) {
    return holoscan::make_unexpected(std::move(bins).error());
  }
  const int width = config_.camera.width;
  const int height = config_.camera.height;
  if (!picture->data.data || picture->data.encoding != holoscan::schema::ImageEncoding_RGB8 ||
      picture->data.width != width || picture->data.height != height) {
    return holoscan::make_unexpected(invalid("expected the camera's size in RGB8"));
  }
  pixels_.resize(static_cast<std::size_t>(width) * height * 3U);
  if (auto copied =
          picture->data.data->copy_to_host(pixels_.data(), pixels_.size(), context.cuda_stream());
      !copied) {
    return copied;
  }
  std::array<float, isp::stats::kStatsSize> numbers{};
  if (auto copied =
          values->data.copy_to_host(numbers.data(), sizeof(numbers), context.cuda_stream());
      !copied) {
    return copied;
  }

  counts_.resize(static_cast<std::size_t>(isp::kHistogramChannels) * isp::kHistogramBins);
  if (auto copied =
          bins->data.copy_to_host(counts_.data(), counts_.size() * 4U, context.cuda_stream());
      !copied) {
    return copied;
  }
  Report& report = *config_.report;
  for (int row = 0; row < isp::kHistogramChannels; ++row) {
    std::uint64_t sum = 0;
    for (int b = 0; b < isp::kHistogramBins; ++b) {
      sum += counts_[static_cast<std::size_t>(row) * isp::kHistogramBins + b];
    }
    report.bad_histograms += sum == static_cast<std::uint64_t>(width) * height ? 0U : 1U;
  }

  // The frame the source rendered: frame ids count from 1.
  isp::SyntheticRawConfig camera = config_.camera;
  const std::uint64_t id = picture->metadata.frame_id;
  camera.frame = static_cast<std::uint32_t>(id == 0 ? 0 : id - 1);
  report.frame_gaps += id != report.frames + 1U ? 1U : 0U;
  const isp::SyntheticRawScene scene = isp::make_synthetic_scene(camera);
  const auto coded = [this](float v) {
    const float f = v * static_cast<float>(lut_.size() - 1);
    const auto i = static_cast<std::size_t>(f);
    return i >= lut_.size() - 1 ? lut_.back()
                                : lut_[i] + (f - static_cast<float>(i)) * (lut_[i + 1] - lut_[i]);
  };
  double squared = 0.0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float truth[3];
      isp::synthetic_truth(scene, x, y, truth);
      for (int k = 0; k < 3; ++k) {
        const double d =
            static_cast<double>(pixels_[(static_cast<std::size_t>(y) * width + x) * 3U + k]) -
            std::round(255.0 * coded(truth[k]));
        squared += d * d;
      }
    }
  }
  const double psnr =
      squared == 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 * 3.0 * width * height / squared);
  report.worst_psnr = std::min(report.worst_psnr, psnr);
  report.psnr_sum += psnr;
  report.wb_gains = {numbers[isp::stats::kWbGainR], numbers[isp::stats::kWbGainG],
                     numbers[isp::stats::kWbGainB]};
  report.ae_error_ev = numbers[isp::stats::kAeErrorEv];
  report.mean_luma = numbers[isp::stats::kMeanLuma];
  ++report.frames;
  config_.finished->fetch_add(1U, std::memory_order_release);
  if (report.frames == config_.expected_frames) {
    context.request_completion();
  }
  return {};
}

}  // namespace isp_demo
