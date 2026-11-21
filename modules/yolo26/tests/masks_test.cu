// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The segmentation kernels against the host:
//
// - launch_segmentation_decode() keeps the same detections as launch_decode()
//   of the rows' first six fields, and gathers each one's coefficients;
// - launch_masks() against a host decode of the masks as their definition
//   reads: prototypes times coefficients, upsampled bilinearly to the image
//   through the letterbox, cropped to the box and thresholded; and the label
//   image against the masks; for random prototypes, coefficients and boxes,
//   wide, tall, odd and upscaled images, a capacity above the count, and
//   other thresholds;
// - the letterbox inversion on its own: prototypes that hold a rectangle in
//   input pixels give masks of that rectangle mapped back to the image;
// - overlapping masks, an empty count, a count above the capacity, optional
//   outputs, and rejected arguments.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/yolo26/kernels.hpp>

#include "test_support.hpp"

namespace holoscan::yolo26 {
namespace {

// A GPU copy of `values`, freed with the object.
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(std::max<std::size_t>(count, 1)) {
    if (cudaMalloc(&data_, count_ * sizeof(T)) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
    // Garbage, so the kernels must write every value they own.
    cudaMemset(data_, 0x5A, count_ * sizeof(T));
  }
  explicit DeviceBuffer(const std::vector<T>& values) : DeviceBuffer(values.size()) {
    cudaMemcpy(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { cudaFree(data_); }
  [[nodiscard]] T* get() const { return data_; }
  [[nodiscard]] std::vector<T> read(std::size_t count) const {
    std::vector<T> out(count);
    cudaMemcpy(out.data(), data_, count * sizeof(T), cudaMemcpyDeviceToHost);
    return out;
  }

 private:
  std::size_t count_;
  T* data_{};
};

struct Case {
  int width;
  int height;
  int input_size;
  int channels;
  int capacity;
  int count;
  float logit_threshold;
  std::uint32_t seed;
};

struct Masks {
  std::vector<std::uint8_t> masks;
  std::vector<std::int32_t> labels;
};

// The masks as kernels.hpp defines them, on the host.
Masks reference(const std::vector<float>& protos, const std::vector<float>& coefficients,
                const std::vector<float>& boxes, int count, const MaskParams& p) {
  const Letterbox& lb = p.letterbox;
  const int width = lb.source_width;
  const int height = lb.source_height;
  const int plane = p.proto_width * p.proto_height;
  const int n = std::clamp(count, 0, p.capacity);
  // Prototypes times coefficients, summed in channel order.
  std::vector<float> logits(static_cast<std::size_t>(n) * plane);
  for (int k = 0; k < n; ++k) {
    for (int i = 0; i < plane; ++i) {
      float sum = 0.0F;
      for (int m = 0; m < p.mask_channels; ++m) {
        sum += coefficients[static_cast<std::size_t>(k) * p.mask_channels + m] *
               protos[static_cast<std::size_t>(m) * plane + i];
      }
      logits[static_cast<std::size_t>(k) * plane + i] = sum;
    }
  }
  const float rx = static_cast<float>(lb.resized_width) / static_cast<float>(width);
  const float ry = static_cast<float>(lb.resized_height) / static_cast<float>(height);
  const float px = static_cast<float>(p.proto_width) / static_cast<float>(lb.size);
  const float py = static_cast<float>(p.proto_height) / static_cast<float>(lb.size);
  Masks out;
  out.masks.assign(static_cast<std::size_t>(p.capacity) * width * height, 0);
  out.labels.assign(static_cast<std::size_t>(width) * height, 0);
  for (int y = 0; y < height; ++y) {
    // The pixel center in input pixels, then in prototype pixels, where
    // PyTorch's bilinear interpolate without align_corners samples it.
    const float sy = std::fmax(
        (static_cast<float>(lb.pad_top) + (static_cast<float>(y) + 0.5F) * ry) * py - 0.5F, 0.0F);
    const int y0 = std::min(static_cast<int>(sy), p.proto_height - 1);
    const int y1 = std::min(y0 + 1, p.proto_height - 1);
    const float wy = sy - static_cast<float>(y0);
    for (int x = 0; x < width; ++x) {
      const float sx = std::fmax(
          (static_cast<float>(lb.pad_left) + (static_cast<float>(x) + 0.5F) * rx) * px - 0.5F,
          0.0F);
      const int x0 = std::min(static_cast<int>(sx), p.proto_width - 1);
      const int x1 = std::min(x0 + 1, p.proto_width - 1);
      const float wx = sx - static_cast<float>(x0);
      const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
      for (int k = 0; k < n; ++k) {
        const float* box = boxes.data() + 4 * k;
        if (!(x >= box[0] && x < box[2] && y >= box[1] && y < box[3])) {
          continue;  // Cropped to the box.
        }
        const float* map = logits.data() + static_cast<std::size_t>(k) * plane;
        const auto at = [&](int u, int v) { return map[v * p.proto_width + u]; };
        const float top = at(x0, y0) * (1.0F - wx) + at(x1, y0) * wx;
        const float bottom = at(x0, y1) * (1.0F - wx) + at(x1, y1) * wx;
        const float value = top * (1.0F - wy) + bottom * wy;
        if (value > p.logit_threshold) {
          out.masks[static_cast<std::size_t>(k) * width * height + pixel] = 1;
          if (out.labels[pixel] == 0) {
            out.labels[pixel] = k + 1;
          }
        }
      }
    }
  }
  return out;
}

// Runs launch_masks() and returns both outputs.
Masks run_masks(const std::vector<float>& protos, const std::vector<float>& coefficients,
                const std::vector<float>& boxes, int count, const MaskParams& p,
                bool with_masks = true, bool with_labels = true) {
  const std::size_t pixels =
      static_cast<std::size_t>(p.letterbox.source_width) * p.letterbox.source_height;
  DeviceBuffer<float> d_protos(protos);
  DeviceBuffer<float> d_coefficients(coefficients);
  DeviceBuffer<float> d_boxes(boxes);
  DeviceBuffer<std::int32_t> d_count(std::vector<std::int32_t>{count});
  DeviceBuffer<std::uint8_t> d_masks(static_cast<std::size_t>(p.capacity) * pixels);
  DeviceBuffer<std::int32_t> d_labels(pixels);
  DeviceBuffer<float> d_logits(static_cast<std::size_t>(p.capacity) * p.proto_width *
                               p.proto_height);
  const cudaError_t status = launch_masks(
      MaskInputs{d_protos.get(), d_coefficients.get(), d_boxes.get(), d_count.get()}, p,
      MaskOutputs{with_masks ? d_masks.get() : nullptr, with_labels ? d_labels.get() : nullptr,
                  d_logits.get()},
      nullptr);
  if (status != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
    throw std::runtime_error("mask kernels failed");
  }
  Masks out;
  if (with_masks) out.masks = d_masks.read(static_cast<std::size_t>(p.capacity) * pixels);
  if (with_labels) out.labels = d_labels.read(pixels);
  return out;
}

MaskParams params_of(const Case& c) {
  const int side = c.input_size / 4;
  return MaskParams{.capacity = c.capacity,
                    .mask_channels = c.channels,
                    .proto_width = side,
                    .proto_height = side,
                    .logit_threshold = c.logit_threshold,
                    .letterbox = make_letterbox(c.width, c.height, c.input_size)};
}

// Random prototypes of smooth blobs, as a model's are, random coefficients,
// and random boxes inside and around the image.
void check_against_reference(const Case& c) {
  const MaskParams p = params_of(c);
  std::mt19937 random(c.seed);
  std::uniform_real_distribution<float> unit(0.0F, 1.0F);
  const int plane = p.proto_width * p.proto_height;
  std::vector<float> protos(static_cast<std::size_t>(c.channels) * plane);
  for (int m = 0; m < c.channels; ++m) {
    const float fx = 0.05F + 0.3F * unit(random);
    const float fy = 0.05F + 0.3F * unit(random);
    const float phase = 6.28F * unit(random);
    for (int v = 0; v < p.proto_height; ++v) {
      for (int u = 0; u < p.proto_width; ++u) {
        protos[static_cast<std::size_t>(m) * plane + v * p.proto_width + u] =
            std::sin(fx * u + phase) * std::cos(fy * v) + 0.2F * (unit(random) - 0.5F);
      }
    }
  }
  std::vector<float> coefficients(static_cast<std::size_t>(c.capacity) * c.channels);
  for (float& value : coefficients) value = 2.0F * unit(random) - 1.0F;
  std::vector<float> boxes(static_cast<std::size_t>(c.capacity) * 4);
  for (int k = 0; k < c.capacity; ++k) {
    const float x1 = (1.2F * unit(random) - 0.1F) * c.width;
    const float y1 = (1.2F * unit(random) - 0.1F) * c.height;
    boxes[4 * k] = x1;
    boxes[4 * k + 1] = y1;
    boxes[4 * k + 2] = x1 + unit(random) * c.width * 0.7F;
    boxes[4 * k + 3] = y1 + unit(random) * c.height * 0.7F;
  }
  const Masks cpu = reference(protos, coefficients, boxes, c.count, p);
  const Masks gpu = run_masks(protos, coefficients, boxes, c.count, p);
  const std::string what = std::to_string(c.width) + "x" + std::to_string(c.height) + " in " +
                           std::to_string(c.input_size) + ", " + std::to_string(c.count) + " of " +
                           std::to_string(c.capacity);
  std::size_t wrong_masks = 0;
  std::size_t set = 0;
  for (std::size_t i = 0; i < cpu.masks.size(); ++i) {
    wrong_masks += gpu.masks[i] != cpu.masks[i];
    set += cpu.masks[i];
  }
  std::size_t wrong_labels = 0;
  for (std::size_t i = 0; i < cpu.labels.size(); ++i)
    wrong_labels += gpu.labels[i] != cpu.labels[i];
  VERIFY_MSG(wrong_masks == 0, what + ": " + std::to_string(wrong_masks) + " mask values differ");
  VERIFY_MSG(wrong_labels == 0, what + ": " + std::to_string(wrong_labels) + " labels differ");
  VERIFY_MSG(c.count == 0 || set > 0, what + ": the case sets some mask pixels");
}

// Prototype 0 holds +1 on a rectangle of input pixels and -1 elsewhere. With
// a box around the whole image, the mask must be that rectangle mapped back
// through the letterbox, (u - pad) / scale, to within the prototypes'
// resolution: 4 input pixels, 4 / scale image pixels.
void check_letterbox_inversion(int width, int height) {
  constexpr int kSize = 640;
  MaskParams p = params_of(Case{width, height, kSize, 1, 1, 1, 0.0F, 0});
  const Letterbox& lb = p.letterbox;
  // A rectangle inside the letterboxed image, on prototype pixel edges.
  const int u0 = (lb.pad_left + lb.resized_width / 4) / 4 * 4;
  const int u1 = (lb.pad_left + lb.resized_width * 3 / 4) / 4 * 4;
  const int v0 = (lb.pad_top + lb.resized_height / 3) / 4 * 4;
  const int v1 = (lb.pad_top + lb.resized_height * 2 / 3) / 4 * 4;
  std::vector<float> protos(static_cast<std::size_t>(p.proto_width) * p.proto_height, -1.0F);
  for (int v = v0 / 4; v < v1 / 4; ++v) {
    for (int u = u0 / 4; u < u1 / 4; ++u) protos[v * p.proto_width + u] = 1.0F;
  }
  const std::vector<float> box{0.0F, 0.0F, static_cast<float>(width), static_cast<float>(height)};
  const Masks gpu = run_masks(protos, {1.0F}, box, 1, p);
  // The rectangle's edges in the image.
  const float x0 = (u0 - lb.pad_left) / lb.scale;
  const float x1 = (u1 - lb.pad_left) / lb.scale;
  const float y0 = (v0 - lb.pad_top) / lb.scale;
  const float y1 = (v1 - lb.pad_top) / lb.scale;
  const float tolerance = 2.0F / lb.scale + 1.0F;
  std::size_t wrong = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float cx = x + 0.5F;
      const float cy = y + 0.5F;
      const bool inside =
          cx > x0 + tolerance && cx < x1 - tolerance && cy > y0 + tolerance && cy < y1 - tolerance;
      const bool outside =
          cx < x0 - tolerance || cx > x1 + tolerance || cy < y0 - tolerance || cy > y1 + tolerance;
      const std::uint8_t value = gpu.masks[static_cast<std::size_t>(y) * width + x];
      wrong += (inside && value != 1) || (outside && value != 0);
    }
  }
  VERIFY_MSG(wrong == 0, std::to_string(width) + "x" + std::to_string(height) + ": " +
                             std::to_string(wrong) + " pixels on the wrong side");
}

// Two overlapping instances: each mask is whole, and the label image gives
// the overlap to the first.
void check_overlap() {
  MaskParams p = params_of(Case{64, 64, 64, 1, 3, 2, 0.0F, 0});
  const std::vector<float> protos(static_cast<std::size_t>(16 * 16), 1.0F);
  const std::vector<float> boxes{0, 0, 40, 40, 20, 20, 64, 64, 0, 0, 64, 64};
  const Masks gpu = run_masks(protos, {1.0F, 1.0F, 1.0F}, boxes, 2, p);
  const auto mask = [&](int k, int x, int y) { return gpu.masks[k * 64 * 64 + y * 64 + x]; };
  const auto label = [&](int x, int y) { return gpu.labels[y * 64 + x]; };
  VERIFY(mask(0, 30, 30) == 1 && mask(1, 30, 30) == 1 && label(30, 30) == 1);
  VERIFY(mask(0, 10, 10) == 1 && mask(1, 10, 10) == 0 && label(10, 10) == 1);
  VERIFY(mask(0, 50, 50) == 0 && mask(1, 50, 50) == 1 && label(50, 50) == 2);
  VERIFY(mask(0, 39, 39) == 1 && mask(0, 40, 39) == 0 && label(5, 50) == 0);
  // The third slot, past the count, is empty although its box covers all.
  VERIFY(std::all_of(gpu.masks.begin() + 2 * 64 * 64, gpu.masks.end(),
                     [](std::uint8_t v) { return v == 0; }));
}

// No detection: empty masks and labels. A count beyond the capacity: the
// capacity. Either output alone gives what both do.
void check_counts_and_outputs() {
  MaskParams p = params_of(Case{80, 48, 64, 1, 2, 0, 0.0F, 0});
  const std::vector<float> protos(16 * 16, 1.0F);
  const std::vector<float> boxes{0, 0, 80, 48, 0, 0, 80, 48};
  const Masks none = run_masks(protos, {1.0F, 1.0F}, boxes, 0, p);
  VERIFY(std::all_of(none.masks.begin(), none.masks.end(), [](auto v) { return v == 0; }));
  VERIFY(std::all_of(none.labels.begin(), none.labels.end(), [](auto v) { return v == 0; }));
  const Masks over = run_masks(protos, {1.0F, 1.0F}, boxes, 7, p);
  VERIFY(std::all_of(over.masks.begin(), over.masks.end(), [](auto v) { return v == 1; }));
  VERIFY(std::all_of(over.labels.begin(), over.labels.end(), [](auto v) { return v == 1; }));
  const Masks masks_only = run_masks(protos, {1.0F, 1.0F}, boxes, 1, p, true, false);
  const Masks labels_only = run_masks(protos, {1.0F, 1.0F}, boxes, 1, p, false, true);
  const Masks both = run_masks(protos, {1.0F, 1.0F}, boxes, 1, p);
  VERIFY(masks_only.masks == both.masks && labels_only.labels == both.labels);
}

// launch_segmentation_decode() keeps what launch_decode() keeps of the rows'
// first six fields, and each kept row's coefficients go to its slot.
void check_segmentation_decode() {
  constexpr int kRows = 700;
  constexpr int kChannels = 32;
  constexpr int kCapacity = 50;
  std::mt19937 random(11);
  std::uniform_real_distribution<float> unit(0.0F, 1.0F);
  std::vector<float> rows;
  std::vector<float> detection_rows;
  for (int r = 0; r < kRows; ++r) {
    const float x1 = unit(random) * 600.0F;
    const float y1 = unit(random) * 600.0F;
    const std::vector<float> head{x1,
                                  y1,
                                  x1 + 5.0F + unit(random) * 100.0F,
                                  y1 + 5.0F + unit(random) * 100.0F,
                                  unit(random),
                                  static_cast<float>(random() % 80U)};
    rows.insert(rows.end(), head.begin(), head.end());
    detection_rows.insert(detection_rows.end(), head.begin(), head.end());
    // Coefficient m of row r is r + m / 100, so a slot shows its row.
    for (int m = 0; m < kChannels; ++m) rows.push_back(static_cast<float>(r) + m / 100.0F);
  }
  const DecodeParams params{.rows = kRows,
                            .score_threshold = 0.6F,
                            .num_classes = 80,
                            .capacity = kCapacity,
                            .letterbox = make_letterbox(1280, 720, 640)};
  DeviceBuffer<float> d_rows(rows);
  DeviceBuffer<float> d_detection_rows(detection_rows);
  struct Out {
    DeviceBuffer<float> boxes{kCapacity * 4};
    DeviceBuffer<float> scores{kCapacity};
    DeviceBuffer<std::int32_t> class_ids{kCapacity};
    DeviceBuffer<std::int32_t> count{1};
    DeviceBuffer<std::int32_t> passed{1};
    DecodeOutputs outputs() const {
      return {boxes.get(), scores.get(), class_ids.get(), count.get(), passed.get()};
    }
  };
  Out seg;
  Out det;
  DeviceBuffer<float> coefficients(kCapacity * kChannels);
  VERIFY(launch_segmentation_decode(d_rows.get(), params, kChannels, seg.outputs(),
                                    coefficients.get(), nullptr) == cudaSuccess);
  VERIFY(launch_decode(d_detection_rows.get(), params, det.outputs(), nullptr) == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  const auto count = seg.count.read(1)[0];
  VERIFY(count == det.count.read(1)[0] && count == kCapacity);
  VERIFY(seg.passed.read(1) == det.passed.read(1));
  VERIFY(seg.boxes.read(kCapacity * 4) == det.boxes.read(kCapacity * 4));
  VERIFY(seg.scores.read(kCapacity) == det.scores.read(kCapacity));
  VERIFY(seg.class_ids.read(kCapacity) == det.class_ids.read(kCapacity));
  // Each slot's coefficients are its row's: find the row by its score.
  const auto scores = seg.scores.read(kCapacity);
  const auto gathered = coefficients.read(kCapacity * kChannels);
  for (int k = 0; k < kCapacity; ++k) {
    int row = -1;
    for (int r = 0; r < kRows; ++r) {
      if (rows[static_cast<std::size_t>(r) * (6 + kChannels) + 4] == scores[k]) row = r;
    }
    for (int m = 0; m < kChannels; ++m) {
      VERIFY_MSG(gathered[k * kChannels + m] == static_cast<float>(row) + m / 100.0F,
                 "slot " + std::to_string(k) + " coefficient " + std::to_string(m));
    }
  }
  // Slots past the count are zeroed: one row kept of two.
  const std::vector<float> two{10, 10, 50, 50, 0.9F, 1, 7, 7, 10, 10, 50, 50, 0.1F, 1, 8, 8};
  DeviceBuffer<float> d_two(two);
  DecodeParams small = params;
  small.rows = 2;
  small.capacity = 3;
  small.letterbox = make_letterbox(640, 640, 640);
  DeviceBuffer<float> few(3 * 2);
  VERIFY(launch_segmentation_decode(d_two.get(), small, 2, seg.outputs(), few.get(), nullptr) ==
         cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  VERIFY(few.read(6) == (std::vector<float>{7, 7, 0, 0, 0, 0}));
}

void check_rejects_bad_arguments() {
  DeviceBuffer<float> f(4096);
  DeviceBuffer<std::int32_t> i(4096);
  DeviceBuffer<std::uint8_t> b(4096);
  const DecodeParams decode{.rows = 1,
                            .score_threshold = 0.5F,
                            .num_classes = 1,
                            .capacity = 1,
                            .letterbox = make_letterbox(32, 32, 32)};
  const DecodeOutputs outputs{f.get(), f.get(), i.get(), i.get()};
  VERIFY(launch_segmentation_decode(f.get(), decode, 4, outputs, f.get(), nullptr) == cudaSuccess);
  VERIFY(launch_segmentation_decode(f.get(), decode, 0, outputs, f.get(), nullptr) ==
         cudaErrorInvalidValue);
  VERIFY(launch_segmentation_decode(f.get(), decode, kMaxMaskChannels + 1, outputs, f.get(),
                                    nullptr) == cudaErrorInvalidValue);
  VERIFY(launch_segmentation_decode(f.get(), decode, 4, outputs, nullptr, nullptr) ==
         cudaErrorInvalidValue);

  const MaskInputs in{f.get(), f.get(), f.get(), i.get()};
  const MaskOutputs out{b.get(), i.get(), f.get()};
  MaskParams p{.capacity = 1,
               .mask_channels = 1,
               .proto_width = 8,
               .proto_height = 8,
               .logit_threshold = 0.0F,
               .letterbox = make_letterbox(32, 32, 32)};
  VERIFY(launch_masks(in, p, out, nullptr) == cudaSuccess);
  VERIFY(launch_masks(MaskInputs{nullptr, f.get(), f.get(), i.get()}, p, out, nullptr) ==
         cudaErrorInvalidValue);
  VERIFY(launch_masks(in, p, MaskOutputs{b.get(), i.get(), nullptr}, nullptr) ==
         cudaErrorInvalidValue);
  MaskParams bad = p;
  bad.capacity = 0;
  VERIFY(launch_masks(in, bad, out, nullptr) == cudaErrorInvalidValue);
  bad = p;
  bad.mask_channels = kMaxMaskChannels + 1;
  VERIFY(launch_masks(in, bad, out, nullptr) == cudaErrorInvalidValue);
  bad = p;
  bad.proto_width = 0;
  VERIFY(launch_masks(in, bad, out, nullptr) == cudaErrorInvalidValue);
  bad = p;
  bad.logit_threshold = std::nanf("");
  VERIFY(launch_masks(in, bad, out, nullptr) == cudaErrorInvalidValue);
  bad = p;
  bad.letterbox = Letterbox{};
  VERIFY(launch_masks(in, bad, out, nullptr) == cudaErrorInvalidValue);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
}

}  // namespace
}  // namespace holoscan::yolo26

int main() {
  namespace yd = holoscan::yolo26;
  using yd::Case;
  return yd::test::run({
      {"reference: wide image, YOLO26 shapes",
       [] { yd::check_against_reference(Case{1280, 720, 640, 32, 20, 12, 0.0F, 1}); }},
      {"reference: tall image, odd padding",
       [] { yd::check_against_reference(Case{333, 1000, 640, 32, 8, 8, 0.0F, 2}); }},
      {"reference: odd wide image, input 320",
       [] { yd::check_against_reference(Case{1001, 333, 320, 32, 6, 5, 0.0F, 3}); }},
      {"reference: small image, upscaled",
       [] { yd::check_against_reference(Case{100, 64, 640, 32, 4, 4, 0.0F, 4}); }},
      {"reference: other threshold and channels",
       [] { yd::check_against_reference(Case{640, 480, 640, 7, 5, 5, 0.85F, 5}); }},
      {"reference: more boxes than one shared chunk",
       [] { yd::check_against_reference(Case{320, 240, 320, 4, 300, 290, -0.2F, 6}); }},
      {"reference: no detections",
       [] { yd::check_against_reference(Case{320, 240, 640, 32, 4, 0, 0.0F, 7}); }},
      {"letterbox inversion, wide", [] { yd::check_letterbox_inversion(1280, 720); }},
      {"letterbox inversion, tall, odd", [] { yd::check_letterbox_inversion(333, 1000); }},
      {"letterbox inversion, upscaled", [] { yd::check_letterbox_inversion(200, 120); }},
      {"overlapping masks", yd::check_overlap},
      {"counts and optional outputs", yd::check_counts_and_outputs},
      {"segmentation decode", yd::check_segmentation_decode},
      {"rejects bad arguments", yd::check_rejects_bad_arguments},
  });
}
