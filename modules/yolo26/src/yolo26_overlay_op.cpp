// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoscan/yolo26/yolo26_overlay_op.hpp>
#include <holoscan/yolo26/yolo26_segmentation_overlay_op.hpp>

namespace holoscan::yolo26 {
namespace {

using std::chrono_literals::operator""ns;
namespace viz = holoscan::visualization;

using Result = holoscan::expected<void, holoscan::Error>;
using Rgba = std::array<std::uint8_t, 4>;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr std::uint32_t kLabelRowBytes = viz::kTextRunHeaderBytes + kOverlayLabelBytes;
constexpr float kLabelSizePx = 18.0F;
constexpr float kShadowPx = 1.5F;
constexpr std::uint32_t kShadowColor = 0xFF000000U;  // Opaque black, 0xAABBGGRR.

// Ten distinct, bright colors; class c gets the color c % 10.
const Rgba& class_color(std::int32_t class_id) {
  static constexpr std::array<Rgba, 10> kPalette{{{230, 25, 75, 255},
                                                  {60, 180, 75, 255},
                                                  {255, 225, 25, 255},
                                                  {0, 130, 200, 255},
                                                  {245, 130, 48, 255},
                                                  {145, 30, 180, 255},
                                                  {70, 240, 240, 255},
                                                  {240, 50, 230, 255},
                                                  {210, 245, 60, 255},
                                                  {250, 190, 212, 255}}};
  return kPalette[static_cast<std::uint32_t>(class_id) % kPalette.size()];
}

holoscan::Error invalid(std::string_view op, const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, std::string(op) + ": " + message};
}

// Copies a packed tensor of `shape` and element type T to `out`, waiting for
// its producer.
template <typename T>
Result fetch(std::string_view op, const holoscan::Tensor& tensor,
             std::span<const std::int64_t> shape, ::cuda::stream_ref stream, std::vector<T>& out) {
  const DLDataType dtype = tensor.dtype();
  const DLDataTypeCode code = std::is_floating_point_v<T> ? kDLFloat : kDLInt;
  const std::span<const std::int64_t> actual = tensor.shape_span();
  if (dtype.code != code || dtype.bits != sizeof(T) * 8U || dtype.lanes != 1U ||
      !std::equal(actual.begin(), actual.end(), shape.begin(), shape.end())) {
    return holoscan::make_unexpected(invalid(op, "unexpected detection tensor type or shape"));
  }
  std::size_t elements = 1;
  for (const std::int64_t extent : shape) {
    elements *= static_cast<std::size_t>(extent);
  }
  out.resize(elements);
  return tensor.copy_to_host(out.data(), out.size() * sizeof(T), stream);
}

// Draws the 2 px wide outline of the box from (x1, y1) to (x2, y2), in image
// pixels, inside its edges and clipped to the image.
void draw_box(std::uint8_t* rgba, int width, int height, float x1, float y1, float x2, float y2,
              const Rgba& color) {
  if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)) {
    return;
  }
  const auto column = [width](float x) {
    return static_cast<int>(std::clamp(std::floor(x), 0.0F, static_cast<float>(width - 1)));
  };
  const auto row = [height](float y) {
    return static_cast<int>(std::clamp(std::floor(y), 0.0F, static_cast<float>(height - 1)));
  };
  const int left = column(x1);
  const int top = row(y1);
  const int right = column(x2 - 1.0F);
  const int bottom = row(y2 - 1.0F);
  if (right < left || bottom < top) {
    return;
  }
  const auto fill = [&](int x0, int y0, int x_end, int y_end) {
    for (int y = y0; y <= y_end; ++y) {
      for (int x = x0; x <= x_end; ++x) {
        std::memcpy(rgba + (static_cast<std::size_t>(y) * width + x) * 4, color.data(), 4);
      }
    }
  };
  fill(left, top, right, std::min(top + 1, bottom));
  fill(left, std::max(bottom - 1, top), right, bottom);
  fill(left, top, std::min(left + 1, right), bottom);
  fill(std::max(right - 1, left), top, right, bottom);
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

// The labels and overlay outputs, as host tensors from the runtime's pool.
void declare_layers(holoscan::OperatorSpec& spec, holoscan::Output<holoscan::Tensor>& labels,
                    holoscan::Output<holoscan::schema::ImageT>& overlay, int capacity, int width,
                    int height) {
  const auto host_bytes = [](std::uint32_t rank, std::uint64_t bytes) {
    return holoscan::TensorOutputSpec{
        .representation =
            holoscan::TensorRepresentation{
                .memory_kind = holoscan::MemoryKind::kHost,
                .dtype = kUInt8,
                .rank = rank,
                .strides = holoscan::TensorStridePolicy::kContiguousOnly,
            },
        .bounds = holoscan::tensor_bounds(bytes),
        .storage = holoscan::TensorOutputStorage::kRuntimePool,
    };
  };
  spec.output(labels, "labels")
      .produces_tensor(host_bytes(2U, 2U * static_cast<std::uint64_t>(capacity) * kLabelRowBytes));
  spec.output(overlay, "overlay")
      .produces_tensor(host_bytes(
          3U, static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) * 4U))
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

// What one overlay draws.
struct Drawing {
  std::string_view op;
  int width;
  int height;
  int capacity;
  const std::vector<std::string>& names;
  // For the segmentation overlay: the label image, and the masks' opacity.
  const holoscan::Tensor* label_image;
  std::uint8_t mask_alpha;
};

// One image's detections, as received, copied to the host, drawn and emitted
// as the two layers.
template <typename Count, typename Boxes, typename Scores, typename ClassIds>
Result draw_and_emit(holoscan::ExecutionContext& context, const Drawing& drawing,
                     Count& count_sample, Boxes& boxes_sample, Scores& scores_sample,
                     ClassIds& class_ids_sample, std::vector<float>& boxes,
                     std::vector<float>& scores, std::vector<std::int32_t>& class_ids,
                     std::vector<std::int32_t>& label_image,
                     holoscan::Output<holoscan::Tensor>& labels,
                     holoscan::Output<holoscan::schema::ImageT>& overlay) {
  // One small copy each to the host, at most `capacity` rows, and the label
  // image.
  const std::int64_t capacity = drawing.capacity;
  const int width = drawing.width;
  const int height = drawing.height;
  const std::array<std::int64_t, 1> one{1};
  const std::array<std::int64_t, 2> box_shape{capacity, 4};
  const std::array<std::int64_t, 1> row_shape{capacity};
  std::vector<std::int32_t> counted;
  for (auto fetched :
       {fetch(drawing.op, count_sample.data, one, context.cuda_stream(), counted),
        fetch(drawing.op, boxes_sample.data, box_shape, context.cuda_stream(), boxes),
        fetch(drawing.op, scores_sample.data, row_shape, context.cuda_stream(), scores),
        fetch(drawing.op, class_ids_sample.data, row_shape, context.cuda_stream(), class_ids)}) {
    if (!fetched) {
      return fetched;
    }
  }
  if (drawing.label_image != nullptr) {
    const std::array<std::int64_t, 2> image_shape{height, width};
    const std::span<const std::int64_t> actual = drawing.label_image->shape_span();
    const DLDataType dtype = drawing.label_image->dtype();
    if (dtype.code != kDLInt || dtype.bits != 32U || dtype.lanes != 1U ||
        !std::equal(actual.begin(), actual.end(), image_shape.begin(), image_shape.end())) {
      return holoscan::make_unexpected(
          invalid(drawing.op, "the label image must be int32 [" + std::to_string(height) + ", " +
                                  std::to_string(width) + "], the configured size"));
    }
    if (auto fetched = fetch(drawing.op, *drawing.label_image, image_shape, context.cuda_stream(),
                             label_image);
        !fetched) {
      return fetched;
    }
  }
  const std::int32_t n = counted.front();
  if (n < 0 || n > capacity) {
    return holoscan::make_unexpected(invalid(drawing.op, "detection count out of range"));
  }
  const auto detections = static_cast<std::size_t>(n);

  // The image's capture time and frame id, so Holoviz draws these layers over
  // their own image; unstamped detections are stamped now, since Holoviz
  // selects layers by capture time.
  holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(count_sample);
  if (!count_sample.metadata.source_clock_id.valid()) {
    emit.capture_time = context.activation_time();
  }

  // Allocates a host tensor on `output` and has `fill` write it.
  const auto write =
      [](auto& output, holoscan::TensorShapeView shape,
         const auto& fill) -> holoscan::expected<holoscan::TensorOutputLoan, holoscan::Error> {
    auto loan =
        output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto written =
        loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer) -> Result {
          fill(writer.data_as<std::uint8_t>());
          return {};
        });
    if (!written) {
      return holoscan::make_unexpected(std::move(written).error());
    }
    return std::move(*loan);
  };

  const std::array<std::int64_t, 2> label_shape{2 * capacity, kLabelRowBytes};
  auto label_loan = write(labels, label_shape, [&](std::uint8_t* out) {
    std::memset(out, 0, static_cast<std::size_t>(2 * capacity) * kLabelRowBytes);
    for (std::size_t i = 0; i < detections; ++i) {
      const std::int32_t id = class_ids[i];
      std::string text = id >= 0 && static_cast<std::size_t>(id) < drawing.names.size()
                             ? drawing.names[static_cast<std::size_t>(id)]
                             : "class " + std::to_string(id);
      char score[16];
      std::snprintf(score, sizeof(score), " %.2f", scores[i]);
      text += score;
      // Above the box, or just inside it when there is no room above.
      const float x1 = boxes[4 * i];
      const float y1 = boxes[4 * i + 1];
      const float top = y1 >= kLabelSizePx + 2.0F ? y1 - kLabelSizePx - 2.0F : y1 + 2.0F;
      const Rgba& rgba = class_color(id);
      const std::uint32_t color =
          0xFF000000U | (std::uint32_t{rgba[2]} << 16U) | (std::uint32_t{rgba[1]} << 8U) | rgba[0];
      // A dark shadow first, so the label reads on any background.
      write_label(out + 2 * i * kLabelRowBytes, text, (x1 + 2.0F + kShadowPx) / width,
                  (top + kShadowPx) / height, kShadowColor);
      write_label(out + (2 * i + 1) * kLabelRowBytes, text, (x1 + 2.0F) / width, top / height,
                  color);
    }
  });
  if (!label_loan) {
    return holoscan::make_unexpected(std::move(label_loan).error());
  }

  const std::array<std::int64_t, 3> overlay_shape{height, width, 4};
  auto overlay_loan = write(overlay, overlay_shape, [&](std::uint8_t* out) {
    // Transparent without detections, so the previous frame's boxes vanish.
    std::memset(out, 0, static_cast<std::size_t>(width) * height * 4);
    if (drawing.label_image != nullptr) {
      // Each instance's pixels in its class's color, the masks' opacity.
      for (std::size_t p = 0; p < label_image.size(); ++p) {
        const std::int32_t label = label_image[p];
        if (label < 1 || static_cast<std::size_t>(label) > detections) {
          continue;
        }
        const Rgba& color = class_color(class_ids[static_cast<std::size_t>(label) - 1U]);
        std::uint8_t* pixel = out + p * 4;
        pixel[0] = color[0];
        pixel[1] = color[1];
        pixel[2] = color[2];
        pixel[3] = drawing.mask_alpha;
      }
    }
    for (std::size_t i = 0; i < detections; ++i) {
      draw_box(out, width, height, boxes[4 * i], boxes[4 * i + 1], boxes[4 * i + 2],
               boxes[4 * i + 3], class_color(class_ids[i]));
    }
  });
  if (!overlay_loan) {
    return holoscan::make_unexpected(std::move(overlay_loan).error());
  }

  // The labels first: the boxes trigger the frame, which takes the newest labels.
  if (auto sent = labels.emit(std::move(*label_loan), emit); !sent) {
    return sent;
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGBA8;
  return overlay.emit_tensor(std::move(*overlay_loan), descriptor, emit);
}

}  // namespace

Yolo26OverlayOp::Yolo26OverlayOp(Yolo26OverlayConfig config) : config_(std::move(config)) {
  if (config_.width < 1 || config_.height < 1 || config_.max_detections < 1) {
    throw std::invalid_argument("Yolo26OverlayOp needs a positive size and max_detections");
  }
}

void Yolo26OverlayOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: one frame in compute() while the next one arrives.
  spec.input(boxes, "boxes").queue_depth(2U);
  spec.input(scores, "scores").queue_depth(2U);
  spec.input(class_ids, "class_ids").queue_depth(2U);
  spec.input(count, "count").queue_depth(2U);
  declare_layers(spec, labels, overlay, config_.max_detections, config_.width, config_.height);
}

holoscan::Contract Yolo26OverlayOp::contract() const {
  // One image's four outputs, paired by their capture time.
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{boxes, scores, class_ids, count}},
      .window = 0ns});
  result.input(boxes, holoscan::Exact{});
  result.input(scores, holoscan::Exact{});
  result.input(class_ids, holoscan::Exact{});
  result.input(count, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> Yolo26OverlayOp::compute(
    holoscan::ExecutionContext& context) {
  auto count_sample = count.receive();
  auto boxes_sample = boxes.receive();
  auto scores_sample = scores.receive();
  auto class_ids_sample = class_ids.receive();
  if (!count_sample) return holoscan::make_unexpected(std::move(count_sample).error());
  if (!boxes_sample) return holoscan::make_unexpected(std::move(boxes_sample).error());
  if (!scores_sample) return holoscan::make_unexpected(std::move(scores_sample).error());
  if (!class_ids_sample) return holoscan::make_unexpected(std::move(class_ids_sample).error());
  std::vector<std::int32_t> no_label_image;
  return draw_and_emit(context,
                       Drawing{.op = "Yolo26OverlayOp",
                               .width = config_.width,
                               .height = config_.height,
                               .capacity = config_.max_detections,
                               .names = config_.labels,
                               .label_image = nullptr,
                               .mask_alpha = 0},
                       *count_sample, *boxes_sample, *scores_sample, *class_ids_sample, boxes_,
                       scores_, class_ids_, no_label_image, labels, overlay);
}

Yolo26SegmentationOverlayOp::Yolo26SegmentationOverlayOp(Yolo26SegmentationOverlayConfig config)
    : config_(std::move(config)) {
  if (config_.width < 1 || config_.height < 1 || config_.max_detections < 1) {
    throw std::invalid_argument(
        "Yolo26SegmentationOverlayOp needs a positive size and max_detections");
  }
}

void Yolo26SegmentationOverlayOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: one frame in compute() while the next one arrives.
  spec.input(boxes, "boxes").queue_depth(2U);
  spec.input(scores, "scores").queue_depth(2U);
  spec.input(class_ids, "class_ids").queue_depth(2U);
  spec.input(count, "count").queue_depth(2U);
  spec.input(label_image, "label_image").queue_depth(2U);
  declare_layers(spec, labels, overlay, config_.max_detections, config_.width, config_.height);
}

holoscan::Contract Yolo26SegmentationOverlayOp::contract() const {
  // One image's five outputs, paired by their capture time.
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                                         boxes, scores, class_ids, count, label_image}},
                                     .window = 0ns});
  result.input(boxes, holoscan::Exact{});
  result.input(scores, holoscan::Exact{});
  result.input(class_ids, holoscan::Exact{});
  result.input(count, holoscan::Exact{});
  result.input(label_image, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> Yolo26SegmentationOverlayOp::compute(
    holoscan::ExecutionContext& context) {
  auto count_sample = count.receive();
  auto boxes_sample = boxes.receive();
  auto scores_sample = scores.receive();
  auto class_ids_sample = class_ids.receive();
  auto label_sample = label_image.receive();
  if (!count_sample) return holoscan::make_unexpected(std::move(count_sample).error());
  if (!boxes_sample) return holoscan::make_unexpected(std::move(boxes_sample).error());
  if (!scores_sample) return holoscan::make_unexpected(std::move(scores_sample).error());
  if (!class_ids_sample) return holoscan::make_unexpected(std::move(class_ids_sample).error());
  if (!label_sample) return holoscan::make_unexpected(std::move(label_sample).error());
  return draw_and_emit(context,
                       Drawing{.op = "Yolo26SegmentationOverlayOp",
                               .width = config_.width,
                               .height = config_.height,
                               .capacity = config_.max_detections,
                               .names = config_.labels,
                               .label_image = &label_sample->data,
                               .mask_alpha = config_.mask_alpha},
                       *count_sample, *boxes_sample, *scores_sample, *class_ids_sample, boxes_,
                       scores_, class_ids_, label_image_, labels, overlay);
}

}  // namespace holoscan::yolo26
