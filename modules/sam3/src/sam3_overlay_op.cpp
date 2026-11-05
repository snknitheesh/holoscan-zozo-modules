// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sam3/kernels.hpp>
#include <holoscan/sam3/sam3_overlay_op.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

namespace holoscan::sam3 {
namespace {

using std::chrono_literals::operator""ns;
namespace viz = holoscan::visualization;

using Result = holoscan::expected<void, holoscan::Error>;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr std::uint32_t kLabelRowBytes = viz::kTextRunHeaderBytes + kOverlayLabelBytes;
constexpr float kLabelSizePx = 18.0F;
constexpr float kShadowPx = 1.5F;
constexpr std::uint32_t kShadowColor = 0xFF000000U;  // Opaque black, 0xAABBGGRR.

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "Sam3OverlayOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         std::string("Sam3OverlayOp: ") + what + ": " + cudaGetErrorString(status)};
}

// Copies a packed tensor of `shape` and element type T to `out`, waiting for
// its producer.
template <typename T>
Result fetch(const holoscan::Tensor& tensor, std::span<const std::int64_t> shape,
             ::cuda::stream_ref stream, std::vector<T>& out) {
  const DLDataType dtype = tensor.dtype();
  const DLDataTypeCode code = std::is_floating_point_v<T> ? kDLFloat : kDLInt;
  const std::span<const std::int64_t> actual = tensor.shape_span();
  if (dtype.code != code || dtype.bits != sizeof(T) * 8U || dtype.lanes != 1U ||
      !std::equal(actual.begin(), actual.end(), shape.begin(), shape.end())) {
    return holoscan::make_unexpected(invalid("unexpected instance tensor type or shape"));
  }
  std::size_t elements = 1;
  for (const std::int64_t extent : shape) {
    elements *= static_cast<std::size_t>(extent);
  }
  out.resize(elements);
  return tensor.copy_to_host(out.data(), out.size() * sizeof(T), stream);
}

// Writes one text layer row: `text`, cut to fit, at (x, y) in framebuffer
// fractions.
void write_label(std::uint8_t* row, const std::string& text, float x, float y,
                 std::uint32_t color) {
  const viz::TextRunHeader header{.x = x, .y = y, .size_px = kLabelSizePx, .color = color};
  std::memcpy(row, &header, sizeof(header));
  std::memcpy(row + viz::kTextRunHeaderBytes, text.data(),
              std::min<std::size_t>(text.size(), kOverlayLabelBytes));
}

}  // namespace

Sam3OverlayOp::Sam3OverlayOp(Sam3OverlayConfig config) : config_(std::move(config)) {
  if (config_.max_width < 1 || config_.max_height < 1 || config_.max_width > kMaxImageSide ||
      config_.max_height > kMaxImageSide || config_.max_instances < 1 ||
      config_.max_instances > 65535) {
    throw std::invalid_argument(
        "Sam3OverlayOp needs a positive max_width, max_height and max_instances");
  }
}

void Sam3OverlayOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: one image's instances in compute() while the next
  // arrive.
  spec.input(labels, "labels").queue_depth(2U);
  spec.input(scores, "scores").queue_depth(2U);
  spec.input(boxes, "boxes").queue_depth(2U);
  spec.input(prompt_ids, "prompt_ids").queue_depth(2U);
  spec.input(count, "count").queue_depth(2U);
  spec.output(text, "text")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kHost,
                  .dtype = kUInt8,
                  .rank = 2U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(2U * static_cast<std::uint64_t>(config_.max_instances) *
                                            kLabelRowBytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
  spec.output(overlay, "overlay")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kUInt8,
                  .rank = 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(static_cast<std::uint64_t>(config_.max_width) *
                                            static_cast<std::uint64_t>(config_.max_height) * 4U),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract Sam3OverlayOp::contract() const {
  // One image's outputs, paired by their capture time.
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                                         labels, scores, boxes, prompt_ids, count}},
                                     .window = 0ns});
  result.input(labels, holoscan::Exact{});
  result.input(scores, holoscan::Exact{});
  result.input(boxes, holoscan::Exact{});
  result.input(prompt_ids, holoscan::Exact{});
  result.input(count, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> Sam3OverlayOp::compute(
    holoscan::ExecutionContext& context) {
  auto count_sample = count.receive();
  auto labels_sample = labels.receive();
  auto scores_sample = scores.receive();
  auto boxes_sample = boxes.receive();
  auto prompt_ids_sample = prompt_ids.receive();
  if (!count_sample) return holoscan::make_unexpected(std::move(count_sample).error());
  if (!labels_sample) return holoscan::make_unexpected(std::move(labels_sample).error());
  if (!scores_sample) return holoscan::make_unexpected(std::move(scores_sample).error());
  if (!boxes_sample) return holoscan::make_unexpected(std::move(boxes_sample).error());
  if (!prompt_ids_sample) return holoscan::make_unexpected(std::move(prompt_ids_sample).error());

  // The label image, colored on the GPU where it is.
  const holoscan::Tensor& label_image = labels_sample->data;
  const std::span<const std::int64_t> label_shape = label_image.shape_span();
  const DLDataType label_type = label_image.dtype();
  if (label_type.code != kDLUInt || label_type.bits != 16U || label_type.lanes != 1U ||
      label_shape.size() != 2 || label_shape[0] < 1 || label_shape[1] < 1 ||
      label_shape[0] > config_.max_height || label_shape[1] > config_.max_width) {
    return holoscan::make_unexpected(
        invalid("the label image must be uint16 [height, width], at most max_width x max_height"));
  }
  const DLDevice where = label_image.device();
  if (where.device_type != kDLCUDA && where.device_type != kDLCUDAManaged) {
    return holoscan::make_unexpected(invalid("the label image must be in GPU memory"));
  }
  const std::vector<std::int64_t> label_strides = label_image.strides();
  const int height = static_cast<int>(label_shape[0]);
  const int width = static_cast<int>(label_shape[1]);
  // Byte strides of packed uint16 rows.
  if (label_strides.size() != 2 || label_strides[1] != 2 || label_strides[0] != 2 * width) {
    return holoscan::make_unexpected(invalid("the label image must be packed"));
  }

  // One small copy each to the host, at most max_instances rows.
  const std::int64_t capacity = config_.max_instances;
  const std::array<std::int64_t, 1> one{1};
  const std::array<std::int64_t, 2> box_shape{capacity, 4};
  const std::array<std::int64_t, 1> row_shape{capacity};
  std::vector<std::int32_t> counted;
  for (auto fetched :
       {fetch(count_sample->data, one, context.cuda_stream(), counted),
        fetch(scores_sample->data, row_shape, context.cuda_stream(), scores_),
        fetch(boxes_sample->data, box_shape, context.cuda_stream(), boxes_),
        fetch(prompt_ids_sample->data, row_shape, context.cuda_stream(), prompt_ids_)}) {
    if (!fetched) {
      return fetched;
    }
  }
  const std::int32_t n = counted.front();
  if (n < 0 || n > capacity) {
    return holoscan::make_unexpected(invalid("instance count out of range"));
  }

  // The image's capture time and frame id, so HoloViz draws these layers over
  // their own image; unstamped instances are stamped now, since HoloViz
  // selects layers by capture time.
  holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*count_sample);
  if (!count_sample->metadata.source_clock_id.valid()) {
    emit.capture_time = context.activation_time();
  }

  // The names of the prompts these instances are likely from: a control's
  // newest, which the segmenter takes from its next image on.
  std::vector<std::string> names = config_.prompt_names;
  if (config_.prompt_control) {
    auto [generation, prompts] = config_.prompt_control->latest();
    if (generation != 0) {
      names.clear();
      for (const Sam3Prompt& prompt : prompts) {
        names.push_back(display_name(prompt));
      }
    }
  }
  const std::array<std::int64_t, 2> text_shape{2 * capacity, kLabelRowBytes};
  auto text_loan =
      text.allocate_tensor(holoscan::TensorLoanRequest{.shape = text_shape, .dtype = kUInt8});
  if (!text_loan) {
    return holoscan::make_unexpected(std::move(text_loan).error());
  }
  auto written = text_loan->write_host_and_commit(
      [&](const holoscan::TensorOutputWriteGuard& writer) -> Result {
        std::uint8_t* out = writer.data_as<std::uint8_t>();
        std::memset(out, 0, static_cast<std::size_t>(2 * capacity) * kLabelRowBytes);
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
          const std::int32_t id = prompt_ids_[i];
          std::string label = id >= 0 && static_cast<std::size_t>(id) < names.size()
                                  ? names[static_cast<std::size_t>(id)]
                                  : "prompt " + std::to_string(id);
          char score[16];
          std::snprintf(score, sizeof(score), " %.2f", scores_[i]);
          label += score;
          // Above the box, or just inside it when there is no room above.
          const float x1 = boxes_[4 * i];
          const float y1 = boxes_[4 * i + 1];
          const float top = y1 >= kLabelSizePx + 2.0F ? y1 - kLabelSizePx - 2.0F : y1 + 2.0F;
          std::uint8_t rgb[3];
          overlay_color(static_cast<int>(i), rgb);
          const std::uint32_t color =
              0xFF000000U | (std::uint32_t{rgb[2]} << 16U) | (std::uint32_t{rgb[1]} << 8U) | rgb[0];
          // A dark shadow first, so the label reads on any background.
          write_label(out + 2 * i * kLabelRowBytes, label,
                      (x1 + 2.0F + kShadowPx) / static_cast<float>(width),
                      (top + kShadowPx) / static_cast<float>(height), kShadowColor);
          write_label(out + (2 * i + 1) * kLabelRowBytes, label,
                      (x1 + 2.0F) / static_cast<float>(width), top / static_cast<float>(height),
                      color);
        }
        return {};
      });
  if (!written) {
    return written;
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "Sam3OverlayOp: no CUDA stream"});
  }
  const std::array<std::int64_t, 3> overlay_shape{height, width, 4};
  auto overlay_loan =
      overlay.allocate_tensor(holoscan::TensorLoanRequest{.shape = overlay_shape, .dtype = kUInt8});
  if (!overlay_loan) {
    return holoscan::make_unexpected(std::move(overlay_loan).error());
  }
  auto writer = overlay_loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (const cudaError_t status =
          launch_overlay(static_cast<const std::uint16_t*>(label_image.data()), width, height,
                         config_.fill_alpha, writer->data_as<std::uint8_t>(), stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("overlay kernel", status));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }

  // The labels first: the overlay triggers the frame, which takes the newest
  // labels.
  if (auto sent = text.emit(std::move(*text_loan), emit); !sent) {
    return sent;
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGBA8;
  return overlay.emit_tensor(std::move(*overlay_loan), descriptor, emit);
}

}  // namespace holoscan::sam3
