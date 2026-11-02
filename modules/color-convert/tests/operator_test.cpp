// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ColorConvertOp and ResizeOp through the SDK's operator test harness: images
// from host and device memory, packed and planar, converted and resized as
// the CPU reference does; the descriptors they publish (encoding, plane
// layouts, colour space, header, ROI) and the capture time and frame id they
// carry; the input's colour space; NV16 as CUSTOM; and unusable inputs and
// configurations refused.

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/color_convert/color_convert_op.hpp>
#include <holoscan/color_convert/converter.hpp>
#include <holoscan/color_convert/reference.hpp>
#include <holoscan/color_convert/resize_op.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "test_support.hpp"
#include "textbook.hpp"

namespace holoscan::color_convert {
namespace {

namespace schema = holoscan::schema;
using std::chrono_literals::operator""ms;
using test::Image;
using ImageSnapshot = holoscan::testing::Snapshot<schema::ImageT>;

// A tensor of `shape` holding `bytes`, in host or device memory.
holoscan::Tensor make_tensor(const std::vector<std::int64_t>& shape, DLDataType dtype,
                             const std::vector<std::uint8_t>& bytes, bool device) {
  auto tensor = holoscan::make_tensor(
      shape, dtype, std::nullopt,
      device ? holoscan::MemoryStorageType::kDevice : holoscan::MemoryStorageType::kHost,
      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) {
    throw std::runtime_error(tensor.error().message);
  }
  auto destination = tensor->mutable_data();
  if (!destination) {
    throw std::runtime_error(destination.error().message);
  }
  if (cudaMemcpy(*destination, bytes.data(), bytes.size(), cudaMemcpyDefault) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return std::move(*tensor);
}

// `img` as an image message, laid out as the operators publish it.
schema::ImageT message(const Image& img, bool device,
                       schema::ColorSpace space = schema::ColorSpace_UNSPECIFIED) {
  const FormatInfo info = format_info(img.format);
  std::vector<std::int64_t> shape;
  if (info.planes > 1) {
    shape = {static_cast<std::int64_t>(img.bytes.size())};
  } else if (info.channels == 1) {
    shape = {img.height, img.width};
  } else {
    shape = {img.height, img.width, info.channels};
  }
  const DLDataType dtype{kDLUInt, static_cast<std::uint8_t>(8 * info.bytes_per_sample), 1};
  schema::ImageT image;
  image.width = img.width;
  image.height = img.height;
  image.encoding = img.format == PixelFormat::kNv16     ? schema::ImageEncoding_CUSTOM
                   : img.format == PixelFormat::kNv12   ? schema::ImageEncoding_NV12
                   : img.format == PixelFormat::kI420   ? schema::ImageEncoding_I420
                   : img.format == PixelFormat::kYuyv   ? schema::ImageEncoding_YUYV
                   : img.format == PixelFormat::kUyvy   ? schema::ImageEncoding_UYVY
                   : img.format == PixelFormat::kRgb8   ? schema::ImageEncoding_RGB8
                   : img.format == PixelFormat::kBgr8   ? schema::ImageEncoding_BGR8
                   : img.format == PixelFormat::kRgba8  ? schema::ImageEncoding_RGBA8
                   : img.format == PixelFormat::kBgra8  ? schema::ImageEncoding_BGRA8
                   : img.format == PixelFormat::kMono8  ? schema::ImageEncoding_MONO8
                   : img.format == PixelFormat::kMono16 ? schema::ImageEncoding_MONO16
                                                        : schema::ImageEncoding_RGB16;
  image.color_space = space;
  for (int plane = 0; info.planes > 1 && plane < info.planes; ++plane) {
    const PlaneSize size = plane_size(img.format, img.width, img.height, plane);
    image.plane_layouts.emplace_back(img.plane_offset(plane),
                                     static_cast<std::uint32_t>(size.row_bytes),
                                     size.row_bytes * static_cast<std::uint64_t>(size.rows));
  }
  image.data = std::make_shared<holoscan::Tensor>(make_tensor(shape, dtype, img.bytes, device));
  return image;
}

// A captured image's pixels, after checking its descriptor says `format`,
// `width` x `height` and, for planar formats, packed plane layouts.
Image pixels(const ImageSnapshot& snapshot, PixelFormat format, int width, int height,
             schema::ImageEncoding encoding) {
  const schema::ImageT descriptor = snapshot.metadata();
  VERIFY(descriptor.encoding == encoding);
  VERIFY(descriptor.width == width && descriptor.height == height);
  Image img(format, width, height);
  const FormatInfo info = format_info(format);
  VERIFY(descriptor.plane_layouts.size() == (info.planes > 1 ? std::size_t(info.planes) : 0U));
  for (std::size_t plane = 0; plane < descriptor.plane_layouts.size(); ++plane) {
    const PlaneSize size = plane_size(format, width, height, static_cast<int>(plane));
    VERIFY(descriptor.plane_layouts[plane].offset_bytes() ==
           img.plane_offset(static_cast<int>(plane)));
    VERIFY(descriptor.plane_layouts[plane].row_stride_bytes() == size.row_bytes);
  }
  const auto& tensor = snapshot.tensor(schema::ImageTensorFields::data);
  VERIFY(tensor.bytes().size() == img.bytes.size());
  if (tensor.bytes().size() == img.bytes.size()) {
    std::memcpy(img.bytes.data(), tensor.bytes().data(), img.bytes.size());
  }
  return img;
}

Image reference(const Image& src, PixelFormat format, int width, int height,
                const ConversionParams& params) {
  Image in = src;
  Image out(format, width, height);
  reference_convert(in.view(), out.view(), params);
  return out;
}

Image pattern(PixelFormat format, int width, int height, std::uint32_t frame,
              ColorMatrix matrix = ColorMatrix::kBt601) {
  Image rgb(PixelFormat::kRgb8, width, height);
  draw_test_pattern(rgb.view(), frame);
  return reference(rgb, format, width, height, ConversionParams{.matrix = matrix});
}

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

// Converts two pattern frames from `from` to `to`, from host or device memory,
// and checks the pixels, the descriptor and the provenance.
void check_convert(PixelFormat from, PixelFormat to, schema::ImageEncoding encoding, bool device) {
  constexpr int kWidth = 96;
  constexpr int kHeight = 54;
  ColorConvertConfig config{.output_format = to, .max_width = 128, .max_height = 128};
  if (from == PixelFormat::kNv16) {
    config.custom_format = PixelFormat::kNv16;
  }
  auto h = holoscan::testing::create_operator_test<ColorConvertOp>(config);
  std::vector<holoscan::testing::TestSample<schema::ImageT>> samples;
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    schema::ImageT image = message(pattern(from, kWidth, kHeight, frame), device);
    image.header = std::make_shared<schema::HeaderT>();
    image.header->frame_id = "camera";
    image.roi_offset_x = 8;
    image.roi_offset_y = 4;
    image.exposure_time_ns = 1000 + frame;
    auto sample = holoscan::testing::sample_at(std::move(image), capture_time(frame));
    sample.metadata.frame_id = 7U + frame;
    samples.push_back(std::move(sample));
  }
  h.push_samples("image", std::move(samples));
  if (device) {
    h.input_source_device("image", holoscan::DeviceId{0});
  }
  h.output_device("converted", holoscan::DeviceId{0});
  h.capture_snapshot<schema::ImageT>("converted");
  const auto result = h.run();
  const std::string what = std::string(format_name(from)) + " -> " + std::string(format_name(to)) +
                           (device ? " (device)" : " (host)");
  VERIFY_MSG(result.ok(), what + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto images = result.snapshots<schema::ImageT>("converted");
  VERIFY_MSG(images.size() == 2, what);
  for (std::uint32_t frame = 0; frame < images.size(); ++frame) {
    const Image got = pixels(images[frame], to, kWidth, kHeight, encoding);
    const Image want = reference(pattern(from, kWidth, kHeight, frame), to, kWidth, kHeight,
                                 ConversionParams{.mode = ResizeMode::kNearest});
    VERIFY_MSG(got.bytes == want.bytes,
               what + ": frame " + std::to_string(frame) + " differs from the reference");
    const schema::ImageT descriptor = images[frame].metadata();
    VERIFY(descriptor.header && descriptor.header->frame_id == "camera");
    VERIFY(descriptor.roi_offset_x == 8 && descriptor.roi_offset_y == 4);
    VERIFY(descriptor.exposure_time_ns == 1000 + frame);
    VERIFY(descriptor.color_space ==
           (is_yuv(to) ? schema::ColorSpace_REC601 : schema::ColorSpace_UNSPECIFIED));
    const holoscan::SampleMetadata& metadata = images[frame].sample_metadata();
    VERIFY(metadata.capture_timestamp_ns == capture_time(frame).count());
    VERIFY(metadata.frame_id == 7U + frame);
  }
}

void check_conversions() {
  check_convert(PixelFormat::kNv12, PixelFormat::kRgb8, schema::ImageEncoding_RGB8, true);
  check_convert(PixelFormat::kNv12, PixelFormat::kRgb8, schema::ImageEncoding_RGB8, false);
  check_convert(PixelFormat::kYuyv, PixelFormat::kBgra8, schema::ImageEncoding_BGRA8, true);
  check_convert(PixelFormat::kRgb8, PixelFormat::kNv12, schema::ImageEncoding_NV12, false);
  check_convert(PixelFormat::kBgr8, PixelFormat::kI420, schema::ImageEncoding_I420, true);
  check_convert(PixelFormat::kUyvy, PixelFormat::kMono16, schema::ImageEncoding_MONO16, true);
  check_convert(PixelFormat::kRgb16, PixelFormat::kYuyv, schema::ImageEncoding_YUYV, false);
  check_convert(PixelFormat::kNv16, PixelFormat::kRgba8, schema::ImageEncoding_RGBA8, true);
  check_convert(PixelFormat::kI420, PixelFormat::kNv16, schema::ImageEncoding_CUSTOM, true);
  check_convert(PixelFormat::kMono8, PixelFormat::kUyvy, schema::ImageEncoding_UYVY, false);
}

// A YCbCr image's colour space chooses its matrix, unless told not to.
void check_input_color_space() {
  constexpr int kWidth = 64;
  constexpr int kHeight = 32;
  const Image yuyv = pattern(PixelFormat::kYuyv, kWidth, kHeight, 0, ColorMatrix::kBt709);
  for (const bool use : {true, false}) {
    auto h = holoscan::testing::create_operator_test<ColorConvertOp>(
        ColorConvertConfig{.output_format = PixelFormat::kRgb8,
                           .use_input_color_space = use,
                           .max_width = kWidth,
                           .max_height = kHeight});
    h.push("image", message(yuyv, true, schema::ColorSpace_REC709));
    h.input_source_device("image", holoscan::DeviceId{0});
    h.output_device("converted", holoscan::DeviceId{0});
    h.capture_snapshot<schema::ImageT>("converted");
    const auto result = h.run();
    VERIFY_MSG(result.ok(), result.describe_failure());
    if (!result.ok()) return;
    const Image got = pixels(result.snapshot<schema::ImageT>("converted"), PixelFormat::kRgb8,
                             kWidth, kHeight, schema::ImageEncoding_RGB8);
    const Image want =
        reference(yuyv, PixelFormat::kRgb8, kWidth, kHeight,
                  ConversionParams{.matrix = use ? ColorMatrix::kBt709 : ColorMatrix::kBt601,
                                   .mode = ResizeMode::kNearest});
    VERIFY_MSG(got.bytes == want.bytes, use ? "REC709 input" : "configured BT.601");
  }
}

// Resizing, fused with a conversion or keeping the input's format.
void check_resize(PixelFormat from, std::optional<PixelFormat> to, ResizeConfig config,
                  schema::ImageEncoding encoding, bool device) {
  constexpr int kWidth = 160;
  constexpr int kHeight = 90;
  config.output_format = to;
  const PixelFormat out = to.value_or(from);
  auto h = holoscan::testing::create_operator_test<ResizeOp>(config);
  schema::ImageT image = message(pattern(from, kWidth, kHeight, 3), device);
  image.roi_offset_x = 2;
  h.push("image", std::move(image));
  if (device) {
    h.input_source_device("image", holoscan::DeviceId{0});
  }
  h.output_device("resized", holoscan::DeviceId{0});
  h.capture_snapshot<schema::ImageT>("resized");
  const auto result = h.run();
  const std::string what = std::string(format_name(from)) + " -> " + std::string(format_name(out)) +
                           " " + std::to_string(config.width) + "x" +
                           std::to_string(config.height) + " " +
                           std::string(resize_mode_name(config.mode));
  VERIFY_MSG(result.ok(), what + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto snapshot = result.snapshot<schema::ImageT>("resized");
  const Image got = pixels(snapshot, out, config.width, config.height, encoding);
  const Image want = reference(pattern(from, kWidth, kHeight, 3), out, config.width, config.height,
                               ConversionParams{.matrix = config.matrix,
                                                .range = config.range,
                                                .mode = config.mode,
                                                .crop = config.crop,
                                                .letterbox = config.letterbox,
                                                .fill_rgb = config.fill_rgb});
  VERIFY_MSG(got.bytes == want.bytes, what + ": differs from the reference");
  VERIFY(snapshot.metadata().roi_offset_x == 0);
}

void check_resizes() {
  check_resize(PixelFormat::kYuyv, PixelFormat::kRgb8,
               ResizeConfig{.width = 96, .height = 96, .letterbox = true, .fill_rgb = {0, 64, 0}},
               schema::ImageEncoding_RGB8, true);
  check_resize(PixelFormat::kNv12, std::nullopt,
               ResizeConfig{.width = 64, .height = 36, .mode = ResizeMode::kArea},
               schema::ImageEncoding_NV12, true);
  check_resize(PixelFormat::kRgba8, PixelFormat::kNv12,
               ResizeConfig{.width = 50,
                            .height = 50,
                            .mode = ResizeMode::kNearest,
                            .crop = Rect{20, 10, 100, 60},
                            .letterbox = true},
               schema::ImageEncoding_NV12, false);
  check_resize(PixelFormat::kI420, PixelFormat::kBgr8,
               ResizeConfig{.width = 333, .height = 181, .mode = ResizeMode::kBilinear},
               schema::ImageEncoding_BGR8, true);
}

// Runs `config`'s operator on `image` and checks the failure names `what`.
template <typename Op, typename Config>
void check_rejects(const Config& config, schema::ImageT image, const char* output,
                   const std::string& what) {
  auto h = holoscan::testing::create_operator_test<Op>(config);
  h.push("image", std::move(image));
  h.output_device(output, holoscan::DeviceId{0});
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

void check_rejected_inputs() {
  const Image rgb = pattern(PixelFormat::kRgb8, 64, 48, 0);
  schema::ImageT bayer = message(rgb, false);
  bayer.encoding = schema::ImageEncoding_BAYER_RGGB;
  check_rejects<ColorConvertOp>(ColorConvertConfig{}, bayer, "converted",
                                "unsupported image encoding");
  check_rejects<ColorConvertOp>(ColorConvertConfig{},
                                message(pattern(PixelFormat::kNv16, 64, 48, 0), false), "converted",
                                "set custom_format");
  check_rejects<ColorConvertOp>(ColorConvertConfig{.max_width = 32, .max_height = 32},
                                message(rgb, false), "converted", "raise max_width");
  check_rejects<ResizeOp>(ResizeConfig{.width = 32, .height = 32, .crop = Rect{40, 0, 40, 40}},
                          message(rgb, false), "resized", "is not inside");
  // The schema lets a 16-bit image travel as an opaque byte buffer; the
  // operators want its words.
  Image mono16 = pattern(PixelFormat::kMono16, 64, 48, 0);
  schema::ImageT bytes = message(mono16, false);
  bytes.data = std::make_shared<holoscan::Tensor>(
      make_tensor({48, 64, 2}, DLDataType{kDLUInt, 8, 1}, mono16.bytes, false));
  check_rejects<ColorConvertOp>(ColorConvertConfig{}, bytes, "converted",
                                "must be [height, width] or [height, width, 1]");
}

void check_rejected_configs() {
  VERIFY(test::throws<std::invalid_argument>(
      [] { ColorConvertOp op(ColorConvertConfig{.max_width = 0}); }));
  VERIFY(test::throws<std::invalid_argument>(
      [] { ColorConvertOp op(ColorConvertConfig{.custom_format = PixelFormat::kNv12}); }));
  VERIFY(test::throws<std::invalid_argument>(
      [] { ResizeOp op(ResizeConfig{.width = 33, .output_format = PixelFormat::kYuyv}); }));
  VERIFY(test::throws<std::invalid_argument>([] { ResizeOp op(ResizeConfig{.height = 0}); }));
  VERIFY(test::throws<std::invalid_argument>(
      [] { ResizeOp op(ResizeConfig{.crop = Rect{0, 0, 0, 10}}); }));
  // The vic backend needs a VIC: refused at construction where there is none.
  if (!probe_platform().has_vic) {
    VERIFY(test::throws<std::invalid_argument>(
        [] { ColorConvertOp op(ColorConvertConfig{.backend = Backend::kVic}); }));
  }
}

}  // namespace
}  // namespace holoscan::color_convert

int main() {
  namespace cc = holoscan::color_convert;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "no CUDA device; skipping\n";
    return 77;
  }
  return cc::test::run({
      {"ColorConvertOp conversions", cc::check_conversions},
      {"the input's colour space", cc::check_input_color_space},
      {"ResizeOp resizes", cc::check_resizes},
      {"unusable inputs rejected", cc::check_rejected_inputs},
      {"unusable configurations rejected", cc::check_rejected_configs},
  });
}
