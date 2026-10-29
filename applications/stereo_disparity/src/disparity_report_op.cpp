// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "disparity_report_op.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/stereo_disparity/synthetic_scene.hpp>

namespace stereo_disparity_app {
namespace {

using std::chrono_literals::operator""ns;
namespace sd = holoscan::stereo_disparity;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "DisparityReportOp: " + message};
}

// Copies image `name`'s packed [H, W] pixels of type T to `out`.
template <typename T>
holoscan::expected<void, holoscan::Error> fetch(const holoscan::schema::ImageT& image,
                                                const char* name, ::cuda::stream_ref stream,
                                                std::vector<T>& out, int& width, int& height) {
  if (!image.data) {
    return holoscan::make_unexpected(invalid(std::string(name) + " has no tensor"));
  }
  const holoscan::Tensor& tensor = *image.data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (shape.size() != 2 ||
      tensor.nbytes() != static_cast<std::int64_t>(shape[0] * shape[1] * sizeof(T))) {
    return holoscan::make_unexpected(invalid(std::string(name) + " must be a packed [H, W] image"));
  }
  height = static_cast<int>(shape[0]);
  width = static_cast<int>(shape[1]);
  out.resize(static_cast<std::size_t>(shape[0] * shape[1]));
  return tensor.copy_to_host(out.data(), out.size() * sizeof(T), stream);
}

}  // namespace

double DisparityReport::error_quantile(double fraction) const {
  if (both_pixels == 0) return 0.0;
  const auto target = static_cast<std::uint64_t>(fraction * static_cast<double>(both_pixels));
  std::uint64_t seen = 0;
  for (std::size_t bin = 0; bin < kErrorBins; ++bin) {
    seen += error_bins[bin];
    if (seen > target) return (static_cast<double>(bin) + 0.5) * kErrorBin;
  }
  return static_cast<double>(kErrorBins) * kErrorBin;
}

double DisparityReport::share_within(double relative) const {
  if (both_pixels == 0) return 0.0;
  const auto bins = static_cast<std::size_t>(std::lround(relative / kErrorBin));
  std::uint64_t within = 0;
  for (std::size_t bin = 0; bin < std::min(bins, kErrorBins); ++bin) within += error_bins[bin];
  return static_cast<double>(within) / static_cast<double>(both_pixels);
}

double DisparityReport::truth_median() const {
  std::uint64_t seen = 0;
  for (std::size_t bin = 0; bin < kTruthBins; ++bin) {
    seen += truth_bins[bin];
    if (2 * seen > matched_pixels) return (static_cast<double>(bin) + 0.5) * kTruthBin;
  }
  return static_cast<double>(kTruthBins) * kTruthBin;
}

DisparityReportOp::DisparityReportOp(DisparityReportConfig config) : config_(std::move(config)) {
  if (!config_.report || !config_.remaining) {
    throw std::invalid_argument("DisparityReportOp needs a report and a completion counter");
  }
}

void DisparityReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(disparity, "disparity").queue_depth(2U);
  if (config_.camera_depth_scale) {
    spec.input(depth, "depth").queue_depth(2U);
    spec.input(camera_depth, "camera_depth").queue_depth(2U);
  }
  if (config_.confidence) {
    spec.input(confidence, "confidence").queue_depth(2U);
  }
}

holoscan::Contract DisparityReportOp::contract() const {
  holoscan::Contract result;
  if (!config_.camera_depth_scale && !config_.confidence) {
    result.trigger(holoscan::OnEach{disparity});
    return result;
  }
  // Every image of one frame carries the same capture time.
  std::vector<holoscan::PortRef> ports{disparity};
  if (config_.camera_depth_scale) {
    ports.emplace_back(depth);
    ports.emplace_back(camera_depth);
  }
  if (config_.confidence) ports.emplace_back(confidence);
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
  result.input(disparity, holoscan::Exact{});
  if (config_.camera_depth_scale) {
    result.input(depth, holoscan::Exact{});
    result.input(camera_depth, holoscan::Exact{});
  }
  if (config_.confidence) result.input(confidence, holoscan::Exact{});
  return result;
}

void DisparityReportOp::start() {
  finished_ = false;
  last_frame_id_ = 0;
}

holoscan::expected<void, holoscan::Error> DisparityReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = disparity.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  DisparityReport& report = *config_.report;
  // Latency from capture to now, when the frame has a capture time on this
  // clock.
  const holoscan::SampleMetadata& metadata = sample->metadata;
  const auto now = context.clock().now();
  if (now && metadata.source_clock_id.valid() && metadata.source_clock_id == now->clock_id) {
    report.latency_ms.push_back(
        static_cast<float>(now->timestamp_ns - metadata.capture_timestamp_ns) / 1e6F);
  }
  int width = 0;
  int height = 0;
  if (auto fetched =
          fetch(sample->data, "disparity", context.cuda_stream(), disparity_, width, height);
      !fetched) {
    return fetched;
  }
  const std::size_t pixels = disparity_.size();
  // Sources number their frames from 1.
  const std::uint64_t frame_id = metadata.frame_id;
  if (last_frame_id_ != 0 && frame_id > last_frame_id_ + 1U) {
    report.frame_gaps += static_cast<std::uint32_t>(frame_id - last_frame_id_ - 1U);
  }
  last_frame_id_ = frame_id;
  report.pixels += pixels;
  for (const float d : disparity_) {
    if (d > 0.0F) {
      ++report.valid_pixels;
      report.whole_pixels += d == std::floor(d) ? 1U : 0U;
    }
  }

  if (config_.camera_depth_scale) {
    auto depth_sample = depth.receive();
    if (!depth_sample) {
      return holoscan::make_unexpected(std::move(depth_sample).error());
    }
    auto camera_sample = camera_depth.receive();
    if (!camera_sample) {
      return holoscan::make_unexpected(std::move(camera_sample).error());
    }
    int depth_width = 0;
    int depth_height = 0;
    if (auto fetched = fetch(depth_sample->data, "depth", context.cuda_stream(), depth_,
                             depth_width, depth_height);
        !fetched) {
      return fetched;
    }
    int camera_width = 0;
    int camera_height = 0;
    if (auto fetched = fetch(camera_sample->data, "camera_depth", context.cuda_stream(),
                             camera_depth_, camera_width, camera_height);
        !fetched) {
      return fetched;
    }
    if (depth_width != width || depth_height != height || camera_width != width ||
        camera_height != height) {
      return holoscan::make_unexpected(
          invalid("the depth images differ from the disparity in size"));
    }
    report.compared = true;
    const double scale = *config_.camera_depth_scale;
    for (std::size_t i = 0; i < pixels; ++i) {
      if (camera_depth_[i] == 0U) continue;
      ++report.camera_pixels;
      if (!(depth_[i] > 0.0F)) continue;
      ++report.both_pixels;
      const double camera = camera_depth_[i] * scale;
      const double relative = std::abs(depth_[i] - camera) / camera;
      ++report.error_bins[std::min(static_cast<std::size_t>(relative / kErrorBin), kErrorBins - 1)];
    }
  }
  if (config_.confidence) {
    auto confidence_sample = confidence.receive();
    if (!confidence_sample) {
      return holoscan::make_unexpected(std::move(confidence_sample).error());
    }
    int confidence_width = 0;
    int confidence_height = 0;
    if (auto fetched = fetch(confidence_sample->data, "confidence", context.cuda_stream(),
                             confidence_, confidence_width, confidence_height);
        !fetched) {
      return fetched;
    }
    if (confidence_width != width || confidence_height != height) {
      return holoscan::make_unexpected(
          invalid("the confidence differs from the disparity in size"));
    }
    for (std::size_t i = 0; i < pixels; ++i) {
      if (disparity_[i] > 0.0F) {
        ++report.confidence_pixels;
        report.confidence_sum += confidence_[i];
      }
    }
  }
  if (config_.truth) {
    if (frame_id == 0 || width != config_.width || height != config_.height) {
      return holoscan::make_unexpected(
          invalid("ground truth needs frames numbered from 1, of the scene's size"));
    }
    report.verified = true;
    const sd::SyntheticScene scene =
        sd::make_scene(static_cast<std::uint32_t>(frame_id - 1U), config_.width, config_.height);
    for (int v = 0; v < height; ++v) {
      for (int u = 0; u < width; ++u) {
        const float truth = sd::true_disparity(scene, static_cast<float>(u), static_cast<float>(v));
        if (truth < 0.0F) continue;
        ++report.truth_pixels;
        const float d = disparity_[static_cast<std::size_t>(v) * width + u];
        if (d > 0.0F) {
          const double error = std::abs(d - truth);
          ++report.matched_pixels;
          report.abs_error_sum += error;
          report.within_1px += error <= 1.0 ? 1U : 0U;
          ++report
                .truth_bins[std::min(static_cast<std::size_t>(error / kTruthBin), kTruthBins - 1)];
        }
      }
    }
  }
  ++report.frames;
  if (!finished_ && config_.expected_frames != 0 && report.frames >= config_.expected_frames) {
    finished_ = true;
    if (config_.remaining->fetch_sub(1) == 1) {
      context.request_completion();
    }
  }
  return {};
}

}  // namespace stereo_disparity_app
