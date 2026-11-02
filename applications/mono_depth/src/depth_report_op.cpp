// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "depth_report_op.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <new>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/mono_depth/image_fit.hpp>
#include <holoscan/mono_depth/reference.hpp>
#include <holoscan/mono_depth/synthetic_scene.hpp>

#include "image_files.hpp"

namespace mono_depth_app {
namespace {

using std::chrono_literals::operator""ns;
namespace md = holoscan::mono_depth;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "DepthReportOp: " + message};
}

std::optional<PixelFormat> pixel_format(holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_RGB8:
      return PixelFormat::kRgb8;
    case holoscan::schema::ImageEncoding_BGR8:
      return PixelFormat::kBgr8;
    case holoscan::schema::ImageEncoding_RGBA8:
      return PixelFormat::kRgba8;
    case holoscan::schema::ImageEncoding_BGRA8:
      return PixelFormat::kBgra8;
    default:
      return std::nullopt;
  }
}

// Copies an image's packed tensor of `channels` bytes per pixel, or floats
// when T is float, to `out`; returns its width and height.
template <typename T>
holoscan::expected<std::pair<int, int>, holoscan::Error> fetch(
    const holoscan::schema::ImageT& image, int channels, ::cuda::stream_ref stream,
    std::vector<T>& out) {
  if (!image.data) {
    return holoscan::make_unexpected(invalid("an image has no tensor"));
  }
  const holoscan::Tensor& tensor = *image.data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const bool rank_ok =
      channels == 1 ? shape.size() == 2 : shape.size() == 3 && shape[2] == channels;
  if (!rank_ok || !tensor.is_contiguous() || tensor.dtype().bits != sizeof(T) * 8U) {
    return holoscan::make_unexpected(invalid("an image is not packed as expected"));
  }
  const int height = static_cast<int>(shape[0]);
  const int width = static_cast<int>(shape[1]);
  out.resize(static_cast<std::size_t>(width) * height * channels);
  if (auto copied = tensor.copy_to_host(out.data(), out.size() * sizeof(T), stream); !copied) {
    return holoscan::make_unexpected(std::move(copied).error());
  }
  return std::pair{width, height};
}

std::string numbered(std::uint64_t frame_id) {
  std::string number = std::to_string(frame_id);
  number.insert(0, number.size() < 6 ? 6 - number.size() : 0, '0');
  return number;
}

}  // namespace

DepthReportOp::DepthReportOp(DepthReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("DepthReportOp needs a report");
  }
  if (config_.delay < std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("DepthReportOp: delay must not be negative");
  }
  if (config_.verify == Verify::kOrdering) {
    (void)md::make_depth_scene(config_.scene_width, config_.scene_height, 0);
  }
}

void DepthReportOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: one frame in compute() while the next one arrives.
  // A real-time edge sets its own depth (see edges.hpp).
  spec.input(depth, "depth").queue_depth(2U);
  if (md::gives_metric_depth(config_.estimator)) {
    spec.input(metric_depth, "metric_depth").queue_depth(2U);
  }
  if (config_.verify == Verify::kReference) {
    spec.input(image, "image").queue_depth(2U);
  }
  if (!config_.image_dir.empty()) {
    spec.input(colorized, "colorized").queue_depth(2U);
  }
}

holoscan::Contract DepthReportOp::contract() const {
  holoscan::Contract result;
  std::vector<holoscan::PortRef> ports{depth};
  const bool metric = md::gives_metric_depth(config_.estimator);
  if (metric) {
    ports.emplace_back(metric_depth);
  }
  if (config_.verify == Verify::kReference) {
    ports.emplace_back(image);
  }
  if (!config_.image_dir.empty()) {
    ports.emplace_back(colorized);
  }
  if (ports.size() == 1) {
    result.trigger(holoscan::OnEach{depth});
    return result;
  }
  // One frame's depth with its image and colorization, by capture time.
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
  result.input(depth, holoscan::Exact{});
  if (metric) {
    result.input(metric_depth, holoscan::Exact{});
  }
  if (config_.verify == Verify::kReference) {
    result.input(image, holoscan::Exact{});
  }
  if (!config_.image_dir.empty()) {
    result.input(colorized, holoscan::Exact{});
  }
  return result;
}

// Each run counts its own frames from the start.
void DepthReportOp::start() {
  seen_ = 0;
  last_frame_id_ = 0;
}

holoscan::expected<void, holoscan::Error> DepthReportOp::compute(
    holoscan::ExecutionContext& context) {
  // The host buffers grow at the first frame, or at a larger image.
  try {
    return process(context);
  } catch (const std::bad_alloc&) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kResourceExhausted,
                                                     "DepthReportOp: out of host memory"});
  }
}

holoscan::expected<void, holoscan::Error> DepthReportOp::process(
    holoscan::ExecutionContext& context) {
  auto depth_sample = depth.receive();
  if (!depth_sample) {
    return holoscan::make_unexpected(std::move(depth_sample).error());
  }
  auto size = fetch(depth_sample->data, 1, context.cuda_stream(), depth_);
  if (!size) {
    return holoscan::make_unexpected(std::move(size).error());
  }
  const auto [width, height] = *size;

  DepthReport& report = *config_.report;
  const holoscan::SampleMetadata& metadata = depth_sample->metadata;
  // The depth is on the host now: the time since capture is the latency a
  // host consumer sees.
  if (auto now = context.clock().now();
      now && metadata.source_clock_id.valid() && now->clock_id == metadata.source_clock_id) {
    report.latencies_ms.push_back(
        static_cast<double>(now->timestamp_ns - metadata.capture_timestamp_ns) * 1e-6);
  }
  // Sources number their frames from 1; the depth keeps its image's.
  const std::uint64_t frame_id = metadata.frame_id;
  if (config_.realtime && frame_id > last_frame_id_) {
    report.skipped += frame_id - last_frame_id_ - 1U;
  } else if (frame_id != last_frame_id_ + 1U) {
    ++report.frame_gaps;
  }
  last_frame_id_ = frame_id;
  report.last_frame_id = frame_id;
  const std::array<float, 2> range = md::reference_range(depth_);
  report.min_depth = std::min(report.min_depth, range[0]);
  report.max_depth = std::max(report.max_depth, range[1]);
  const bool metric = md::gives_metric_depth(config_.estimator);
  if (metric) {
    auto metric_sample = metric_depth.receive();
    if (!metric_sample) {
      return holoscan::make_unexpected(std::move(metric_sample).error());
    }
    auto metric_size = fetch(metric_sample->data, 1, context.cuda_stream(), meters_);
    if (!metric_size) {
      return holoscan::make_unexpected(std::move(metric_size).error());
    }
    if (*metric_size != *size) {
      return holoscan::make_unexpected(invalid("the metric and inverse depth differ in size"));
    }
    const std::array<float, 2> meters = md::reference_range(meters_);
    report.min_meters = std::min(report.min_meters, meters[0]);
    report.max_meters = std::max(report.max_meters, meters[1]);
  }

  if (config_.verify == Verify::kReference) {
    auto frame = image.receive();
    if (!frame) {
      return holoscan::make_unexpected(std::move(frame).error());
    }
    const auto format = pixel_format(frame->data.encoding);
    if (!format) {
      return holoscan::make_unexpected(
          invalid("can check only RGB8, BGR8, RGBA8 and BGRA8 images"));
    }
    auto image_size = fetch(frame->data, channels(*format), context.cuda_stream(), pixels_);
    if (!image_size) {
      return holoscan::make_unexpected(std::move(image_size).error());
    }
    if (*image_size != *size) {
      return holoscan::make_unexpected(invalid("the depth and its image differ in size"));
    }
    // The stub's output has the model input's size.
    const md::MonoDepthConfig& e = config_.estimator;
    const md::ImageFit fit = md::make_fit(width, height, e.input_width, e.input_height,
                                          e.input_width, e.input_height, e.resize_mode);
    const std::vector<float> input = md::reference_preprocess(
        pixels_.data(), static_cast<std::size_t>(width) * channels(*format), *format, fit,
        e.normalization);
    const md::InverseDepthAffine* affine =
        e.metric_from_relative ? &*e.metric_from_relative : nullptr;
    const md::ReferenceDepth expected = md::reference_postprocess(
        md::reference_stub(input, e.input_width, e.input_height), fit, e.model_output, affine);
    float worst = 0.0F;
    const auto compare = [&worst](const std::vector<float>& actual,
                                  const std::vector<float>& reference) {
      for (std::size_t i = 0; i < actual.size() && i < reference.size(); ++i) {
        const float error =
            std::fabs(actual[i] - reference[i]) / std::max(1.0F, std::fabs(reference[i]));
        worst = std::max(worst, std::isfinite(error) ? error : 1e30F);
      }
    };
    compare(depth_, expected.inverse);
    if (metric) {
      compare(meters_, expected.metric);
    }
    ++report.reference_checked;
    report.max_reference_error = std::max(report.max_reference_error, worst);
    if (!(worst <= config_.tolerance)) {
      ++report.reference_failed;
    }
  }

  if (config_.verify == Verify::kOrdering && frame_id != 0) {
    const md::DepthScene scene =
        md::make_depth_scene(config_.scene_width, config_.scene_height,
                             config_.still ? 0U : static_cast<std::uint32_t>(frame_id - 1U));
    if (width != scene.width || height != scene.height) {
      return holoscan::make_unexpected(invalid("the depth is not the synthetic scene's size"));
    }
    const md::OrderingResult result = md::check_ordering(scene, depth_);
    ++report.ordering_checked;
    report.min_spearman = std::min(report.min_spearman, result.spearman);
    report.sum_spearman += result.spearman;
    if (!(result.spearman >= config_.min_spearman) || !result.ordered) {
      ++report.ordering_failed;
    }
  }

  if (config_.print && !config_.files.empty() && frame_id != 0 &&
      frame_id <= config_.files.size()) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4) << "mono_depth: "
        << config_.files[(frame_id - 1U) % config_.files.size()].filename().string() << ": "
        << width << "x" << height << " inverse depth " << range[0] << ".." << range[1] << '\n';
    std::cout << out.str() << std::flush;
  }

  if (!config_.image_dir.empty()) {
    auto color = colorized.receive();
    if (!color) {
      return holoscan::make_unexpected(std::move(color).error());
    }
    std::vector<std::uint8_t> rgba;
    auto color_size = fetch(color->data, 4, context.cuda_stream(), rgba);
    if (!color_size) {
      return holoscan::make_unexpected(std::move(color_size).error());
    }
    const auto [color_width, color_height] = *color_size;
    try {
      write_ppm(config_.image_dir / ("depth_" + numbered(frame_id) + ".ppm"), color_width,
                color_height,
                to_rgb(rgba.data(), static_cast<std::size_t>(color_width) * 4U, color_width,
                       color_height, PixelFormat::kRgba8));
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, error.what()});
    }
    ++report.files_written;
  }

  if (config_.delay > std::chrono::nanoseconds::zero()) {
    std::this_thread::sleep_for(config_.delay);
  }
  ++seen_;
  report.frames.fetch_add(1U);
  const bool last =
      config_.realtime ? frame_id >= config_.expected_frames : seen_ == config_.expected_frames;
  if (config_.expected_frames != 0 && last) {
    context.request_completion();
  }
  return {};
}

}  // namespace mono_depth_app
