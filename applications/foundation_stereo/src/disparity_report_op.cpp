// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "disparity_report_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/foundation_stereo/model_input.hpp>
#include <holoscan/foundation_stereo/reference.hpp>
#include <holoscan/foundation_stereo/synthetic_scene.hpp>

namespace foundation_stereo_app {
namespace {

using std::chrono_literals::operator""ns;
namespace sd = holoscan::foundation_stereo;

constexpr std::uint32_t kQueueDepth = 8U;

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
  // Room for a few frames: the checks run on the CPU, and on a lossless edge
  // a full queue would fail the frame upstream.
  spec.input(disparity, "disparity").queue_depth(kQueueDepth);
  if (config_.camera_depth_scale) {
    spec.input(depth, "depth").queue_depth(kQueueDepth);
    spec.input(camera_depth, "camera_depth").queue_depth(kQueueDepth);
  }
}

holoscan::Contract DisparityReportOp::contract() const {
  holoscan::Contract result;
  if (!config_.camera_depth_scale) {
    result.trigger(holoscan::OnEach{disparity});
    return result;
  }
  // Every image of one frame carries the same capture time.
  std::vector<holoscan::PortRef> ports{disparity};
  if (config_.camera_depth_scale) {
    ports.emplace_back(depth);
    ports.emplace_back(camera_depth);
  }
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
  result.input(disparity, holoscan::Exact{});
  if (config_.camera_depth_scale) {
    result.input(depth, holoscan::Exact{});
    result.input(camera_depth, holoscan::Exact{});
  }
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
  int width = 0;
  int height = 0;
  if (auto fetched =
          fetch(sample->data, "disparity", context.cuda_stream(), disparity_, width, height);
      !fetched) {
    return fetched;
  }
  // Latency from capture to the disparity on the host (the copy above waits
  // for the GPU), when the frame has a capture time on this clock.
  const holoscan::SampleMetadata& metadata = sample->metadata;
  const auto now = context.clock().now();
  if (now && metadata.source_clock_id.valid() && metadata.source_clock_id == now->clock_id) {
    report.latency_ms.push_back(
        static_cast<float>(now->timestamp_ns - metadata.capture_timestamp_ns) / 1e6F);
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
    report.valid_pixels += d > 0.0F ? 1U : 0U;
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
  if (config_.stub_model_size) {
    if (frame_id == 0 || width != config_.width || height != config_.height) {
      return holoscan::make_unexpected(
          invalid("the stub check needs frames numbered from 1, of the scene's size"));
    }
    check_stub(sd::make_scene(static_cast<std::uint32_t>(frame_id - 1U), width, height), report);
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

void DisparityReportOp::check_stub(const sd::SyntheticScene& scene, DisparityReport& report) {
  const int width = scene.width;
  const int height = scene.height;
  const int channels = config_.color ? 3 : 1;
  const auto [mw, mh] = *config_.stub_model_size;
  // The scene's views as the source rendered them, on the CPU.
  std::array<std::vector<std::uint8_t>, 2> views;
  for (int side = 0; side < 2; ++side) {
    views[side].resize(static_cast<std::size_t>(width) * height * channels);
    for (int v = 0; v < height; ++v) {
      for (int u = 0; u < width; ++u) {
        std::uint8_t rgb[3];
        sd::scene_color(scene, side == 0 ? 0.0F : scene.baseline, static_cast<float>(u),
                        static_cast<float>(v), rgb);
        std::uint8_t* out =
            views[side].data() + (static_cast<std::size_t>(v) * width + u) * channels;
        for (int c = 0; c < channels; ++c) out[c] = rgb[channels == 1 ? 1 : c];
      }
    }
  }
  const sd::PixelFormat format = config_.color ? sd::PixelFormat::kRgb8 : sd::PixelFormat::kMono8;
  const std::size_t pitch = static_cast<std::size_t>(width) * channels;
  const auto left =
      sd::reference::preprocess(views[0].data(), pitch, width, height, format, mw, mh);
  const auto right =
      sd::reference::preprocess(views[1].data(), pitch, width, height, format, mw, mh);
  const auto raw =
      sd::reference::rescale(sd::reference::stub(left, right, mw, mh), mw, mh, width, height);
  report.stub_checked = true;
  for (std::size_t i = 0; i < raw.size() && i < disparity_.size(); ++i) {
    const int x = static_cast<int>(i % static_cast<std::size_t>(width));
    const double limit = config_.stub_tolerance * std::max(1.0, std::abs(raw[i]));
    if (sd::reference::near_limit(raw[i], x, limit)) continue;
    const double error = std::abs(disparity_[i] - sd::reference::valid(raw[i], x, true));
    ++report.stub_pixels;
    report.stub_wrong += error > limit ? 1U : 0U;
    report.stub_worst = std::max(report.stub_worst, error);
  }
}

}  // namespace foundation_stereo_app
