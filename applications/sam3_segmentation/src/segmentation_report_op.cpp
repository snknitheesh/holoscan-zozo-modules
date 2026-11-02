// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "segmentation_report_op.hpp"

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
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/sam3/reference.hpp>

#include "image_files.hpp"

namespace sam3_app {
namespace {

using std::chrono_literals::operator""ns;
namespace s3 = holoscan::sam3;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "SegmentationReportOp: " + message};
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

// Copies a packed tensor of element type T to `out`; returns its shape.
template <typename T>
holoscan::expected<std::vector<std::int64_t>, holoscan::Error> fetch(const holoscan::Tensor& tensor,
                                                                     ::cuda::stream_ref stream,
                                                                     std::vector<T>& out) {
  if (!tensor.is_contiguous() || tensor.dtype().bits != sizeof(T) * 8U) {
    return holoscan::make_unexpected(invalid("a tensor is not packed as expected"));
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  std::size_t elements = 1;
  for (const std::int64_t extent : shape) {
    elements *= static_cast<std::size_t>(extent);
  }
  out.resize(elements);
  if (auto copied = tensor.copy_to_host(out.data(), out.size() * sizeof(T), stream); !copied) {
    return holoscan::make_unexpected(std::move(copied).error());
  }
  return std::vector<std::int64_t>(shape.begin(), shape.end());
}

std::string numbered(std::uint64_t frame_id) {
  std::string number = std::to_string(frame_id);
  number.insert(0, number.size() < 6 ? 6 - number.size() : 0, '0');
  return number;
}

// The prompts in effect: the control's, once set, or the configured ones.
std::vector<s3::Sam3Prompt> prompts_of(const s3::Sam3SegmentationConfig& config) {
  if (config.prompt_control) {
    auto [generation, prompts] = config.prompt_control->latest();
    if (generation != 0) {
      return prompts;
    }
  }
  return config.prompts;
}

}  // namespace

SegmentationReportOp::SegmentationReportOp(SegmentationReportConfig config)
    : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("SegmentationReportOp needs a report");
  }
  if (config_.delay < std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("SegmentationReportOp: delay must not be negative");
  }
  if (config_.verify == Verify::kTruth) {
    (void)s3::make_shape_scene(config_.scene_width, config_.scene_height, 0);
    if (config_.expected_shapes.size() != config_.segmenter.prompts.size()) {
      throw std::invalid_argument("SegmentationReportOp: one expected shape per prompt");
    }
  }
}

bool SegmentationReportOp::needs_image() const noexcept {
  return config_.verify == Verify::kReference || !config_.image_dir.empty();
}

void SegmentationReportOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: one frame in compute() while the next one arrives.
  // A real-time edge sets its own depth (see edges.hpp).
  spec.input(masks, "masks").queue_depth(2U);
  spec.input(labels, "labels").queue_depth(2U);
  spec.input(scores, "scores").queue_depth(2U);
  spec.input(boxes, "boxes").queue_depth(2U);
  spec.input(prompt_ids, "prompt_ids").queue_depth(2U);
  spec.input(count, "count").queue_depth(2U);
  if (needs_image()) {
    spec.input(image, "image").queue_depth(2U);
  }
}

holoscan::Contract SegmentationReportOp::contract() const {
  // One image's instances, and the image, by capture time.
  holoscan::Contract result;
  std::vector<holoscan::PortRef> ports{masks, labels, scores, boxes, prompt_ids, count};
  if (needs_image()) {
    ports.emplace_back(image);
  }
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
  result.input(masks, holoscan::Exact{});
  result.input(labels, holoscan::Exact{});
  result.input(scores, holoscan::Exact{});
  result.input(boxes, holoscan::Exact{});
  result.input(prompt_ids, holoscan::Exact{});
  result.input(count, holoscan::Exact{});
  if (needs_image()) {
    result.input(image, holoscan::Exact{});
  }
  return result;
}

// Each run counts its own frames from the start.
void SegmentationReportOp::start() {
  seen_ = 0;
  last_frame_id_ = 0;
}

holoscan::expected<void, holoscan::Error> SegmentationReportOp::compute(
    holoscan::ExecutionContext& context) {
  // The host buffers grow at the first frame, or at a larger image.
  try {
    return process(context);
  } catch (const std::bad_alloc&) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kResourceExhausted,
                                                     "SegmentationReportOp: out of host memory"});
  }
}

holoscan::expected<void, holoscan::Error> SegmentationReportOp::process(
    holoscan::ExecutionContext& context) {
  auto count_sample = count.receive();
  auto masks_sample = masks.receive();
  auto labels_sample = labels.receive();
  auto scores_sample = scores.receive();
  auto boxes_sample = boxes.receive();
  auto prompt_ids_sample = prompt_ids.receive();
  if (!count_sample) return holoscan::make_unexpected(std::move(count_sample).error());
  if (!masks_sample) return holoscan::make_unexpected(std::move(masks_sample).error());
  if (!labels_sample) return holoscan::make_unexpected(std::move(labels_sample).error());
  if (!scores_sample) return holoscan::make_unexpected(std::move(scores_sample).error());
  if (!boxes_sample) return holoscan::make_unexpected(std::move(boxes_sample).error());
  if (!prompt_ids_sample) return holoscan::make_unexpected(std::move(prompt_ids_sample).error());
  const ::cuda::stream_ref stream = context.cuda_stream();
  std::vector<std::int32_t> counted;
  auto label_shape = fetch(labels_sample->data, stream, labels_);
  auto mask_shape = fetch(masks_sample->data, stream, masks_);
  for (auto fetched : {fetch(count_sample->data, stream, counted).has_value(),
                       fetch(scores_sample->data, stream, scores_).has_value(),
                       fetch(boxes_sample->data, stream, boxes_).has_value(),
                       fetch(prompt_ids_sample->data, stream, prompt_ids_).has_value()}) {
    if (!fetched) {
      return holoscan::make_unexpected(invalid("cannot read the instances"));
    }
  }
  if (!label_shape || !mask_shape || label_shape->size() != 2 || mask_shape->size() != 3 ||
      counted.size() != 1) {
    return holoscan::make_unexpected(invalid("unexpected instance tensors"));
  }
  const int height = static_cast<int>((*label_shape)[0]);
  const int width = static_cast<int>((*label_shape)[1]);
  const int capacity = static_cast<int>((*mask_shape)[0]);
  const int n = counted[0];
  if (n < 0 || n > capacity) {
    return holoscan::make_unexpected(invalid("instance count out of range"));
  }
  const std::size_t plane = static_cast<std::size_t>(width) * height;

  SegmentationReport& report = *config_.report;
  const holoscan::SampleMetadata& metadata = count_sample->metadata;
  // The instances are on the host now: the time since capture is the latency
  // a host consumer sees.
  if (auto now = context.clock().now();
      now && metadata.source_clock_id.valid() && now->clock_id == metadata.source_clock_id) {
    report.latencies_ms.push_back(
        static_cast<double>(now->timestamp_ns - metadata.capture_timestamp_ns) * 1e-6);
  }
  // Sources number their frames from 1; the instances keep their image's.
  const std::uint64_t frame_id = metadata.frame_id;
  if (config_.realtime && frame_id > last_frame_id_) {
    report.skipped += frame_id - last_frame_id_ - 1U;
  } else if (frame_id != last_frame_id_ + 1U) {
    ++report.frame_gaps;
  }
  last_frame_id_ = frame_id;
  report.last_frame_id = frame_id;
  report.instances += static_cast<std::uint64_t>(n);
  const std::vector<s3::Sam3Prompt> prompts = prompts_of(config_.segmenter);

  std::optional<PixelFormat> format;
  if (needs_image()) {
    auto frame = image.receive();
    if (!frame) {
      return holoscan::make_unexpected(std::move(frame).error());
    }
    format = pixel_format(frame->data.encoding);
    if (!format || !frame->data.data) {
      return holoscan::make_unexpected(
          invalid("can check only RGB8, BGR8, RGBA8 and BGRA8 images"));
    }
    auto image_shape = fetch(*frame->data.data, stream, pixels_);
    if (!image_shape) {
      return holoscan::make_unexpected(std::move(image_shape).error());
    }
    if (image_shape->size() != 3 || (*image_shape)[0] != height || (*image_shape)[1] != width) {
      return holoscan::make_unexpected(invalid("the instances and their image differ in size"));
    }
  }

  if (config_.verify == Verify::kReference) {
    // The stub's outputs, from the image, through the references.
    const s3::Sam3SegmentationConfig& seg = config_.segmenter;
    const std::vector<float> input = s3::reference_preprocess(
        pixels_.data(), static_cast<std::size_t>(width) * channels(*format), *format, width, height,
        config_.stub.input_size, config_.stub.input_size, s3::Normalization{});
    const std::vector<float> fpn0 = s3::reference_stub_encoder(input, config_.stub);
    const s3::PromptTable table = s3::read_prompt_table(seg.prompt_table);
    std::vector<s3::DecoderOutput> outputs;
    for (const s3::Sam3Prompt& prompt : prompts) {
      outputs.push_back(s3::reference_stub_decoder(
          fpn0, s3::make_decoder_prompt(prompt, table, config_.stub.boxes), config_.stub));
    }
    const float threshold = s3::logit_of(seg.mask_threshold);
    const s3::ReferenceInstances expected = s3::reference_instances(
        outputs, seg.max_instances, seg.score_threshold, threshold, width, height);
    bool same = expected.count == n && expected.prompt_ids == prompt_ids_;
    for (std::size_t k = 0; same && k < scores_.size(); ++k) {
      same = std::fabs(scores_[k] - expected.scores[k]) <= 1e-3F;
    }
    std::uint64_t differing = 0;
    for (std::size_t k = 0; same && k < static_cast<std::size_t>(n); ++k) {
      for (std::size_t i = 0; i < plane; ++i) {
        if (masks_[k * plane + i] != expected.masks[k * plane + i] &&
            std::fabs(expected.logits[k * plane + i] - threshold) > config_.logit_slack) {
          ++differing;
        }
      }
    }
    ++report.reference_checked;
    report.worst_reference_pixels = std::max(report.worst_reference_pixels, differing);
    if (!same || differing != 0) {
      ++report.reference_failed;
    }
  }

  if (config_.verify == Verify::kTruth && frame_id != 0) {
    const s3::ShapeScene scene =
        s3::make_shape_scene(config_.scene_width, config_.scene_height,
                             config_.still ? 0U : static_cast<std::uint32_t>(frame_id - 1U));
    if (width != scene.width || height != scene.height) {
      return holoscan::make_unexpected(invalid("the instances are not the synthetic scene's size"));
    }
    std::vector<s3::SceneShape> shapes(plane);
    std::vector<std::uint8_t> ignored(plane * 3U);
    s3::render_shape_scene(scene, PixelFormat::kRgb8, ignored.data(),
                           static_cast<std::size_t>(width) * 3U, shapes.data());
    bool failed = false;
    for (std::size_t p = 0; p < config_.expected_shapes.size(); ++p) {
      if (!config_.expected_shapes[p]) {
        continue;
      }
      // Instances are best first: the prompt's first is its best.
      const auto end = prompt_ids_.begin() + n;
      const auto first = std::find(prompt_ids_.begin(), end, static_cast<std::int32_t>(p));
      double iou = 0.0;
      if (first != end) {
        const auto k = static_cast<std::size_t>(first - prompt_ids_.begin());
        iou = s3::mask_iou(std::span(masks_.data() + k * plane, plane), shapes,
                           *config_.expected_shapes[p]);
      }
      report.min_iou = std::min(report.min_iou, iou);
      report.sum_iou += iou;
      ++report.ious;
      failed |= !(iou >= config_.min_iou);
    }
    ++report.truth_checked;
    if (failed) {
      ++report.truth_failed;
    }
  }

  if (config_.print) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << "sam3_segmentation: ";
    if (!config_.files.empty() && frame_id != 0) {
      out << config_.files[(frame_id - 1U) % config_.files.size()].filename().string() << ": ";
    } else {
      out << "frame " << frame_id << ": ";
    }
    out << width << "x" << height << ", " << n << " instances";
    for (std::size_t k = 0; k < static_cast<std::size_t>(n); ++k) {
      const std::int32_t id = prompt_ids_[k];
      const std::string name = id >= 0 && static_cast<std::size_t>(id) < prompts.size()
                                   ? s3::display_name(prompts[static_cast<std::size_t>(id)])
                                   : "prompt " + std::to_string(id);
      out << (k == 0 ? ": " : ", ") << name << ' ' << scores_[k] << " [" << std::setprecision(0)
          << boxes_[4 * k] << ' ' << boxes_[4 * k + 1] << ' ' << boxes_[4 * k + 2] << ' '
          << boxes_[4 * k + 3] << ']' << std::setprecision(2);
    }
    std::cout << out.str() << '\n' << std::flush;
  }

  if (!config_.image_dir.empty()) {
    // The image under its instances' overlay, blended as HoloViz blends it.
    const std::vector<std::uint8_t> rgb =
        to_rgb(pixels_.data(), static_cast<std::size_t>(width) * channels(*format), width, height,
               *format);
    const std::vector<std::uint8_t> overlay = s3::reference_overlay(labels_, width, height, 110);
    std::vector<std::uint8_t> blended(rgb.size());
    for (std::size_t i = 0; i < plane; ++i) {
      const float alpha = static_cast<float>(overlay[4 * i + 3]) / 255.0F;
      for (std::size_t c = 0; c < 3; ++c) {
        blended[3 * i + c] = static_cast<std::uint8_t>(
            std::lround(alpha * static_cast<float>(overlay[4 * i + c]) +
                        (1.0F - alpha) * static_cast<float>(rgb[3 * i + c])));
      }
    }
    try {
      write_ppm(config_.image_dir / ("segment_" + numbered(frame_id) + ".ppm"), width, height,
                blended);
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

}  // namespace sam3_app
