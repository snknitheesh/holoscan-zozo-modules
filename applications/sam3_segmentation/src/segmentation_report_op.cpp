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

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return holoscan::Error{
      holoscan::ErrorCode::kFailure,
      std::string("SegmentationReportOp: ") + what + ": " + cudaGetErrorString(status)};
}

// The number of elements of a packed tensor of element type T, or an error.
template <typename T>
holoscan::expected<std::size_t, holoscan::Error> packed_elements(const holoscan::Tensor& tensor) {
  if (!tensor.is_contiguous() || tensor.dtype().bits != sizeof(T) * 8U) {
    return holoscan::make_unexpected(invalid("a tensor is not packed as expected"));
  }
  std::size_t elements = 1;
  for (const std::int64_t extent : tensor.shape_span()) {
    elements *= static_cast<std::size_t>(extent);
  }
  return elements;
}

// Queues a copy of a packed tensor of element type T to `out` on `stream`,
// after its producer's work, growing `out` to fit.
template <typename T>
holoscan::expected<void, holoscan::Error> fetch_async(const holoscan::Tensor& tensor,
                                                      cudaStream_t stream, PinnedBuffer& out) {
  auto elements = packed_elements<T>(tensor);
  if (!elements) {
    return holoscan::make_unexpected(std::move(elements).error());
  }
  if (auto reserved = out.reserve(*elements * sizeof(T)); !reserved) {
    return reserved;
  }
  return tensor.copy_to_host_async(out.as<void>(), *elements * sizeof(T), stream);
}

std::string numbered(std::uint64_t frame_id) {
  std::string number = std::to_string(frame_id);
  number.insert(0, number.size() < 6 ? 6 - number.size() : 0, '0');
  return number;
}

// The prompts the image captured at `capture_ns` was segmented with: those
// the segmenter recorded with its control, or the configured ones.
std::vector<s3::Sam3Prompt> prompts_of(const s3::Sam3SegmentationConfig& config,
                                       std::int64_t capture_ns) {
  if (config.prompt_control) {
    if (auto prompts = config.prompt_control->prompts_at(capture_ns)) {
      return *std::move(prompts);
    }
  }
  return config.prompts;
}

}  // namespace

PinnedBuffer::~PinnedBuffer() {
  if (data_ != nullptr) {
    cudaFreeHost(data_);
  }
}

holoscan::expected<void, holoscan::Error> PinnedBuffer::reserve(std::size_t bytes) {
  if (bytes <= bytes_) {
    return {};
  }
  if (data_ != nullptr) {
    cudaFreeHost(data_);
    data_ = nullptr;
    bytes_ = 0;
  }
  if (const cudaError_t status = cudaMallocHost(&data_, bytes); status != cudaSuccess) {
    data_ = nullptr;
    return holoscan::make_unexpected(cuda_failure("cudaMallocHost", status));
  }
  bytes_ = bytes;
  return {};
}

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

SegmentationReportOp::~SegmentationReportOp() { release(); }

void SegmentationReportOp::release() noexcept {
  if (instances_copied_ != nullptr) {
    cudaEventDestroy(instances_copied_);
    instances_copied_ = nullptr;
  }
}

// Each run counts its own frames from the start.
void SegmentationReportOp::start() {
  release();
  seen_ = 0;
  last_frame_id_ = 0;
  if (config_.verify == Verify::kReference) {
    table_ = s3::read_prompt_table(config_.segmenter.prompt_table);
  }
  if (const cudaError_t status =
          cudaEventCreateWithFlags(&instances_copied_, cudaEventDisableTiming);
      status != cudaSuccess) {
    instances_copied_ = nullptr;
    throw std::runtime_error(std::string("SegmentationReportOp: cudaEventCreateWithFlags: ") +
                             cudaGetErrorString(status));
  }
}

void SegmentationReportOp::stop() { release(); }

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
  // In real time a source sends its last frame again until it is reported
  // (image_sources.hpp): a frame already reported is passed over.
  if (config_.realtime && seen_ != 0 && count_sample->metadata.frame_id <= last_frame_id_) {
    if (needs_image()) {
      if (auto frame = image.receive(); !frame) {
        return holoscan::make_unexpected(std::move(frame).error());
      }
    }
    return {};
  }
  if (instances_copied_ == nullptr) {
    return holoscan::make_unexpected(invalid("not started"));
  }
  // The runtime's stream, which outlives the run's output pools: they wait on
  // it before they reuse what was copied. Without one, every copy completes
  // before it returns.
  cudaStream_t stream = context.cuda_stream().get();
  const holoscan::Tensor& label_image = labels_sample->data;
  const holoscan::Tensor& mask_planes = masks_sample->data;
  const std::span<const std::int64_t> label_shape = label_image.shape_span();
  const std::span<const std::int64_t> mask_shape = mask_planes.shape_span();
  const std::span<const std::int64_t> count_shape = count_sample->data.shape_span();
  if (label_shape.size() != 2 || mask_shape.size() != 3 || count_shape.size() != 1 ||
      count_shape[0] != 1) {
    return holoscan::make_unexpected(invalid("unexpected instance tensors"));
  }
  const int height = static_cast<int>(label_shape[0]);
  const int width = static_cast<int>(label_shape[1]);
  const int capacity = static_cast<int>(mask_shape[0]);
  const std::size_t plane = static_cast<std::size_t>(width) * height;
  const bool verifies = config_.verify != Verify::kNone;
  const bool reads_instances = verifies || config_.print;
  const bool writes_images = !config_.image_dir.empty();

  // The instances, then the masks, labels and image the settings read, to
  // pinned memory on the stream.
  if (auto copied = fetch_async<std::int32_t>(count_sample->data, stream, count_); !copied) {
    return copied;
  }
  if (reads_instances) {
    for (auto copied : {fetch_async<float>(scores_sample->data, stream, scores_),
                        fetch_async<float>(boxes_sample->data, stream, boxes_),
                        fetch_async<std::int32_t>(prompt_ids_sample->data, stream, prompt_ids_)}) {
      if (!copied) {
        return copied;
      }
    }
  }
  if (stream != nullptr) {
    if (const cudaError_t status = cudaEventRecord(instances_copied_, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaEventRecord", status));
    }
  }
  bool more = false;
  if (verifies) {
    if (auto copied = fetch_async<std::uint8_t>(mask_planes, stream, masks_); !copied) {
      return copied;
    }
    more = true;
  }
  if (writes_images) {
    if (auto copied = fetch_async<std::uint16_t>(label_image, stream, labels_); !copied) {
      return copied;
    }
    more = true;
  }
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
    const std::span<const std::int64_t> image_shape = frame->data.data->shape_span();
    if (image_shape.size() != 3 || image_shape[0] != height || image_shape[1] != width) {
      return holoscan::make_unexpected(invalid("the instances and their image differ in size"));
    }
    if (auto copied = fetch_async<std::uint8_t>(*frame->data.data, stream, pixels_); !copied) {
      return copied;
    }
    more = true;
  }

  // The instances are on the host now: the time since capture is the latency
  // a host consumer sees.
  if (stream != nullptr) {
    if (const cudaError_t status = cudaEventSynchronize(instances_copied_); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaEventSynchronize", status));
    }
  }
  SegmentationReport& report = *config_.report;
  const holoscan::SampleMetadata& metadata = count_sample->metadata;
  if (auto now = context.clock().now();
      now && metadata.source_clock_id.valid() && now->clock_id == metadata.source_clock_id) {
    report.latencies_ms.push_back(
        static_cast<double>(now->timestamp_ns - metadata.capture_timestamp_ns) * 1e-6);
  }
  if (more && stream != nullptr) {
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
    }
  }
  const int n = *count_.as<std::int32_t>();
  if (n < 0 || n > capacity) {
    return holoscan::make_unexpected(invalid("instance count out of range"));
  }
  const auto slots = static_cast<std::size_t>(capacity);
  const std::span<const float> scores_host(scores_.as<float>(), reads_instances ? slots : 0U);
  const std::span<const float> boxes_host(boxes_.as<float>(), reads_instances ? 4U * slots : 0U);
  const std::span<const std::int32_t> ids(prompt_ids_.as<std::int32_t>(),
                                          reads_instances ? slots : 0U);
  const std::uint8_t* masks_host = masks_.as<std::uint8_t>();
  const std::uint8_t* pixels = pixels_.as<std::uint8_t>();

  // Sources number their frames from 1; the instances keep their image's.
  const std::uint64_t frame_id = metadata.frame_id;
  if (config_.realtime && frame_id > last_frame_id_) {
    report.skipped += frame_id - last_frame_id_ - 1U;
  } else if (frame_id != last_frame_id_ + 1U) {
    ++report.frame_gaps;
  }
  last_frame_id_ = frame_id;
  report.last_frame_id.store(frame_id, std::memory_order_release);
  report.instances += static_cast<std::uint64_t>(n);
  const std::vector<s3::Sam3Prompt> prompts =
      prompts_of(config_.segmenter, metadata.capture_timestamp_ns);

  if (config_.verify == Verify::kReference) {
    // The stub's outputs, from the image, through the references.
    const s3::Sam3SegmentationConfig& seg = config_.segmenter;
    const std::vector<float> input = s3::reference_preprocess(
        pixels, static_cast<std::size_t>(width) * channels(*format), *format, width, height,
        config_.stub.input_size, config_.stub.input_size, s3::Normalization{});
    const std::vector<float> fpn0 = s3::reference_stub_encoder(input, config_.stub);
    std::vector<s3::DecoderOutput> outputs;
    for (const s3::Sam3Prompt& prompt : prompts) {
      outputs.push_back(s3::reference_stub_decoder(
          fpn0, s3::make_decoder_prompt(prompt, *table_, config_.stub.boxes), config_.stub));
    }
    const float threshold = s3::logit_of(seg.mask_threshold);
    const s3::ReferenceInstances expected = s3::reference_instances(
        outputs, seg.max_instances, seg.score_threshold, threshold, width, height);
    bool same =
        expected.count == n &&
        std::equal(expected.prompt_ids.begin(), expected.prompt_ids.end(), ids.begin(), ids.end());
    for (std::size_t k = 0; same && k < scores_host.size(); ++k) {
      same = std::fabs(scores_host[k] - expected.scores[k]) <= 1e-3F;
    }
    std::uint64_t differing = 0;
    for (std::size_t k = 0; same && k < static_cast<std::size_t>(n); ++k) {
      for (std::size_t i = 0; i < plane; ++i) {
        if (masks_host[k * plane + i] != expected.masks[k * plane + i] &&
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
      const auto end = ids.begin() + n;
      const auto first = std::find(ids.begin(), end, static_cast<std::int32_t>(p));
      double iou = 0.0;
      if (first != end) {
        const auto k = static_cast<std::size_t>(first - ids.begin());
        iou = s3::mask_iou(std::span(masks_host + k * plane, plane), shapes,
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
      const std::int32_t id = ids[k];
      const std::string name = id >= 0 && static_cast<std::size_t>(id) < prompts.size()
                                   ? s3::display_name(prompts[static_cast<std::size_t>(id)])
                                   : "prompt " + std::to_string(id);
      out << (k == 0 ? ": " : ", ") << name << ' ' << scores_host[k] << " [" << std::setprecision(0)
          << boxes_host[4 * k] << ' ' << boxes_host[4 * k + 1] << ' ' << boxes_host[4 * k + 2]
          << ' ' << boxes_host[4 * k + 3] << ']' << std::setprecision(2);
    }
    std::cout << out.str() << '\n' << std::flush;
  }

  if (!config_.image_dir.empty()) {
    // The image under its instances' overlay, blended as HoloViz blends it.
    const std::vector<std::uint8_t> rgb =
        to_rgb(pixels, static_cast<std::size_t>(width) * channels(*format), width, height, *format);
    const std::vector<std::uint8_t> overlay = s3::reference_overlay(
        std::span<const std::uint16_t>(labels_.as<std::uint16_t>(), plane), width, height, 110);
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
