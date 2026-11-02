// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// VideoEncoderOp and VideoDecoderOp through the SDK's operator test harness, on the backend `auto`
// picks: images of each encoding, in host and device memory, encoded with their headers and
// timestamps carried, keyframes at the GOP, and the raw stream written to a file; access units
// decoded back to RGB and NV12 images of the right size, in order and close to their sources; a
// change of size started as a new stream, or refused; empty and undecodable access units giving
// nothing; and unusable settings refused. Exits 77 (skipped) when no backend can encode and
// decode here.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/codec/bitstream.hpp>
#include <holoscan/codec/test_video.hpp>
#include <holoscan/codec/video_codec.hpp>
#include <holoscan/codec/video_decoder_op.hpp>
#include <holoscan/codec/video_encoder_op.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "test_support.hpp"

namespace holoscan::codec {
namespace {

using Bytes = std::vector<std::uint8_t>;
using ImageSnapshot = holoscan::testing::Snapshot<holoscan::schema::ImageT>;
using FrameSnapshot = holoscan::testing::Snapshot<schema::EncodedVideoFrameT>;

constexpr std::int64_t kFirstCapture = 1'000'000'000;
constexpr std::int64_t kFramePeriod = 33'333'333;

// A host tensor owning `bytes`, with `shape` (packed).
std::shared_ptr<holoscan::Tensor> host_tensor(Bytes bytes, std::vector<std::int64_t> shape) {
  struct Storage {
    DLManagedTensor managed{};
    std::vector<std::int64_t> shape;
    Bytes bytes;
  };
  auto storage = std::make_unique<Storage>();
  storage->shape = std::move(shape);
  storage->bytes = std::move(bytes);
  storage->managed.dl_tensor = {.data = storage->bytes.data(),
                                .device = {kDLCPU, 0},
                                .ndim = static_cast<std::int32_t>(storage->shape.size()),
                                .dtype = {kDLUInt, 8, 1},
                                .shape = storage->shape.data(),
                                .strides = nullptr,
                                .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* value) {
    delete static_cast<Storage*>(value->manager_ctx);
  };
  return std::make_shared<holoscan::Tensor>(&storage.release()->managed);
}

holoscan::schema::ImageEncoding encoding_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kBgr8:
      return holoscan::schema::ImageEncoding_BGR8;
    case PixelFormat::kRgba8:
      return holoscan::schema::ImageEncoding_RGBA8;
    case PixelFormat::kBgra8:
      return holoscan::schema::ImageEncoding_BGRA8;
    case PixelFormat::kNv12:
      return holoscan::schema::ImageEncoding_NV12;
    case PixelFormat::kRgb8:
      break;
  }
  return holoscan::schema::ImageEncoding_RGB8;
}

// BT.709 limited-range NV12 of an RGB8 image, on the host, as a reference for NV12 input.
Bytes to_nv12(const Bytes& rgb, int width, int height) {
  const auto clamp = [](double v) {
    return static_cast<std::uint8_t>(std::clamp(v + 0.5, 0.0, 255.0));
  };
  const int cw = (width + 1) / 2;
  const int ch = (height + 1) / 2;
  Bytes nv12(static_cast<std::size_t>(width) * height + static_cast<std::size_t>(cw) * 2 * ch);
  const auto pixel = [&](int x, int y, int c) {
    x = std::min(x, width - 1);
    y = std::min(y, height - 1);
    return static_cast<double>(rgb[(static_cast<std::size_t>(y) * width + x) * 3 + c]);
  };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const double l = 0.2126 * pixel(x, y, 0) + 0.7152 * pixel(x, y, 1) + 0.0722 * pixel(x, y, 2);
      nv12[static_cast<std::size_t>(y) * width + x] = clamp(16.0 + 219.0 / 255.0 * l);
    }
  }
  std::uint8_t* uv = nv12.data() + static_cast<std::size_t>(width) * height;
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      double r = 0, g = 0, b = 0;
      for (int d = 0; d < 4; ++d) {
        r += pixel(2 * x + d % 2, 2 * y + d / 2, 0) / 4;
        g += pixel(2 * x + d % 2, 2 * y + d / 2, 1) / 4;
        b += pixel(2 * x + d % 2, 2 * y + d / 2, 2) / 4;
      }
      const double l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      uv[(static_cast<std::size_t>(y) * cw + x) * 2] =
          clamp(128.0 + 224.0 / 255.0 * (b - l) / 1.8556);
      uv[(static_cast<std::size_t>(y) * cw + x) * 2 + 1] =
          clamp(128.0 + 224.0 / 255.0 * (r - l) / 1.5748);
    }
  }
  return nv12;
}

// Frame `index` of the test video as an image of `format`, with a header whose capture time
// and frame id identify it.
holoscan::schema::ImageT test_image(int width, int height, std::uint32_t index,
                                    PixelFormat format = PixelFormat::kRgb8) {
  const Bytes rgb = render_test_frame(width, height, index);
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding_of(format);
  image.header = std::make_shared<holoscan::schema::HeaderT>();
  image.header->capture_timestamp_ns = kFirstCapture + kFramePeriod * index;
  image.header->frame_id = "camera_" + std::to_string(index);
  image.header->device_sequence = index + 1U;
  if (format == PixelFormat::kNv12) {
    Bytes nv12 = to_nv12(rgb, width, height);
    const std::uint64_t luma = static_cast<std::uint64_t>(width) * height;
    const auto chroma_row = static_cast<std::uint32_t>(2 * ((width + 1) / 2));
    image.plane_layouts = {
        holoscan::schema::ImagePlaneLayout(0, static_cast<std::uint32_t>(width), luma),
        holoscan::schema::ImagePlaneLayout(luma, chroma_row, nv12.size() - luma)};
    const auto size = static_cast<std::int64_t>(nv12.size());
    image.data = host_tensor(std::move(nv12), {size});
    return image;
  }
  const int channels = bytes_per_pixel(format);
  Bytes pixels(static_cast<std::size_t>(width) * height * channels);
  for (std::size_t p = 0; p < static_cast<std::size_t>(width) * height; ++p) {
    const bool bgr = format == PixelFormat::kBgr8 || format == PixelFormat::kBgra8;
    pixels[p * channels + 0] = rgb[p * 3 + (bgr ? 2 : 0)];
    pixels[p * channels + 1] = rgb[p * 3 + 1];
    pixels[p * channels + 2] = rgb[p * 3 + (bgr ? 0 : 2)];
    if (channels == 4) pixels[p * channels + 3] = 255;
  }
  image.data = host_tensor(std::move(pixels), {height, width, channels});
  return image;
}

Bytes bytes_of(const holoscan::testing::TensorSnapshot& tensor) {
  Bytes bytes(tensor.bytes().size());
  std::memcpy(bytes.data(), tensor.bytes().data(), bytes.size());
  return bytes;
}

// Runs an encoder over `images`; the encoded frames, or none when the run fails (and
// `expect_failure` says whether it should).
std::vector<FrameSnapshot> run_encoder(const VideoEncoderConfig& config,
                                       const std::vector<holoscan::schema::ImageT>& images,
                                       bool device_input = false, bool expect_failure = false) {
  auto h = holoscan::testing::create_operator_test<VideoEncoderOp>(config);
  for (const auto& image : images) {
    if (!device_input) {
      h.push("image", image);
      continue;
    }
    // The same image with its pixels in device memory on GPU 0.
    holoscan::schema::ImageT on_device = image;
    auto copy = image.data->copy_to(DLDevice{kDLCUDA, 0});
    VERIFY_MSG(copy.has_value(), "copy to the device: " + (copy ? "" : copy.error().message));
    VERIFY(cudaDeviceSynchronize() == cudaSuccess);
    if (!copy) return {};
    on_device.data = std::make_shared<holoscan::Tensor>(std::move(*copy));
    h.push("image", on_device);
  }
  if (device_input) h.input_source_device("image", holoscan::DeviceId{0});
  h.capture_snapshot<schema::EncodedVideoFrameT>("bitstream");
  const auto result = h.run();
  if (expect_failure) {
    VERIFY_MSG(!result.ok(), "the encoder run should have failed");
    return {};
  }
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return {};
  return result.snapshots<schema::EncodedVideoFrameT>("bitstream");
}

// The encoded frames as decoder input.
std::vector<schema::EncodedVideoFrameT> as_input(const std::vector<FrameSnapshot>& frames) {
  std::vector<schema::EncodedVideoFrameT> inputs;
  for (const FrameSnapshot& frame : frames) {
    schema::EncodedVideoFrameT input = frame.metadata();
    Bytes bytes = bytes_of(frame.tensor(schema::EncodedVideoFrameTensorFields::data));
    const auto size = static_cast<std::int64_t>(bytes.size());
    input.data = host_tensor(std::move(bytes), {size});
    inputs.push_back(std::move(input));
  }
  return inputs;
}

std::vector<ImageSnapshot> run_decoder(const VideoDecoderConfig& config,
                                       const std::vector<schema::EncodedVideoFrameT>& frames) {
  auto h = holoscan::testing::create_operator_test<VideoDecoderOp>(config);
  for (const auto& frame : frames) h.push("bitstream", frame);
  h.output_device("image", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("image");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return {};
  return result.snapshots<holoscan::schema::ImageT>("image");
}

VideoEncoderConfig encoder_config(Codec codec, std::uint32_t gop = 30) {
  VideoEncoderConfig config;
  config.settings.codec = codec;
  config.settings.rate_control = RateControl::kConstQp;
  config.settings.qp = 22;
  config.settings.gop = gop;
  return config;
}

// RGB8 host images in, frames out with the images' headers, pts, keyframes at the GOP, and the
// raw stream in the file; decoded back to RGB8 and NV12 images close to their sources.
void check_round_trip(Codec codec) {
  constexpr std::uint32_t kFrames = 12;
  constexpr int kWidth = 640;
  constexpr int kHeight = 480;
  const auto file = std::filesystem::temp_directory_path() /
                    ("codec_operator_test." + std::string(codec_name(codec)));
  VideoEncoderConfig config = encoder_config(codec, 5);
  config.output_path = file;
  std::vector<holoscan::schema::ImageT> images;
  for (std::uint32_t i = 0; i < kFrames; ++i) images.push_back(test_image(kWidth, kHeight, i));
  const auto frames = run_encoder(config, images);
  VERIFY_MSG(frames.size() == kFrames, std::to_string(frames.size()) + " encoded frames");
  Bytes stream;
  for (std::uint32_t i = 0; i < frames.size(); ++i) {
    const schema::EncodedVideoFrameT meta = frames[i].metadata();
    const Bytes data = bytes_of(frames[i].tensor(schema::EncodedVideoFrameTensorFields::data));
    VERIFY(meta.codec ==
           (codec == Codec::kH264 ? schema::VideoCodec_H264 : schema::VideoCodec_H265));
    VERIFY(meta.width == kWidth && meta.height == kHeight);
    VERIFY(meta.frame_index == i);
    VERIFY(meta.pts == kFirstCapture + kFramePeriod * i);
    VERIFY(meta.keyframe == (i % 5 == 0));
    VERIFY(meta.keyframe == is_keyframe(codec, data));
    VERIFY(meta.header && meta.header->frame_id == "camera_" + std::to_string(i));
    VERIFY(meta.header && meta.header->device_sequence == i + 1U);
    stream.insert(stream.end(), data.begin(), data.end());
  }
  std::ifstream in(file, std::ios::binary);
  const Bytes written((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  VERIFY_MSG(written == stream, "the file holds " + std::to_string(written.size()) +
                                    " bytes, the frames " + std::to_string(stream.size()));
  std::filesystem::remove(file);

  for (const PixelFormat format : {PixelFormat::kRgb8, PixelFormat::kNv12, PixelFormat::kBgra8}) {
    VideoDecoderConfig decoder;
    decoder.output_format = format;
    const auto decoded = run_decoder(decoder, as_input(frames));
    VERIFY_MSG(decoded.size() == kFrames, std::string(pixel_format_name(format)) + ": " +
                                              std::to_string(decoded.size()) + " images");
    for (std::uint32_t i = 0; i < decoded.size(); ++i) {
      const holoscan::schema::ImageT meta = decoded[i].metadata();
      const holoscan::testing::TensorSnapshot& tensor =
          decoded[i].tensor(holoscan::schema::ImageTensorFields::data);
      VERIFY(meta.width == kWidth && meta.height == kHeight);
      VERIFY(meta.header && meta.header->capture_timestamp_ns == kFirstCapture + kFramePeriod * i);
      VERIFY(meta.header && meta.header->frame_id == "camera_" + std::to_string(i));
      VERIFY(tensor.source_device().device_type == kDLCUDA);
      const Bytes pixels = bytes_of(tensor);
      const Bytes source = render_test_frame(kWidth, kHeight, i);
      if (format == PixelFormat::kNv12) {
        VERIFY(meta.encoding == holoscan::schema::ImageEncoding_NV12);
        VERIFY(meta.plane_layouts.size() == 2);
        const Bytes expected = to_nv12(source, kWidth, kHeight);
        VERIFY(pixels.size() == expected.size());
        // Luma against the source's, as one-channel images.
        VERIFY(psnr(expected.data(), kWidth, pixels.data(), kWidth, kWidth, kHeight, 1) > 38.0);
      } else if (format == PixelFormat::kBgra8) {
        Bytes rgb(source.size());
        for (std::size_t p = 0; p < rgb.size() / 3; ++p) {
          rgb[p * 3] = pixels[p * 4 + 2];
          rgb[p * 3 + 1] = pixels[p * 4 + 1];
          rgb[p * 3 + 2] = pixels[p * 4];
          VERIFY(pixels[p * 4 + 3] == 255);
        }
        VERIFY(psnr(source, rgb, kWidth, kHeight, 3) > 35.0);
      } else {
        VERIFY(meta.encoding == holoscan::schema::ImageEncoding_RGB8);
        VERIFY(psnr(source, pixels, kWidth, kHeight, 3) > 35.0);
      }
    }
  }
}

// Each input encoding, in host or device memory, encodes to the same picture; an odd size is
// cropped back by the decoder.
void check_inputs(Codec codec) {
  struct Case {
    PixelFormat format;
    bool device;
    int width;
    int height;
  };
  const Case cases[] = {
      {PixelFormat::kBgr8, true, 640, 480},  {PixelFormat::kRgba8, false, 640, 480},
      {PixelFormat::kBgra8, true, 640, 480}, {PixelFormat::kNv12, false, 640, 480},
      {PixelFormat::kNv12, true, 640, 480},  {PixelFormat::kRgb8, true, 641, 361},
  };
  for (const Case& c : cases) {
    const std::string name = std::string(pixel_format_name(c.format)) +
                             (c.device ? " device " : " host ") + std::to_string(c.width) + "x" +
                             std::to_string(c.height);
    std::vector<holoscan::schema::ImageT> images;
    for (std::uint32_t i = 0; i < 3; ++i) {
      images.push_back(test_image(c.width, c.height, i, c.format));
    }
    const auto frames = run_encoder(encoder_config(codec), images, c.device);
    VERIFY_MSG(frames.size() == 3, name + ": " + std::to_string(frames.size()) + " frames");
    const auto decoded = run_decoder(VideoDecoderConfig{}, as_input(frames));
    VERIFY_MSG(decoded.size() == 3, name + ": " + std::to_string(decoded.size()) + " images");
    for (std::uint32_t i = 0; i < decoded.size(); ++i) {
      const holoscan::schema::ImageT meta = decoded[i].metadata();
      VERIFY(meta.width == c.width && meta.height == c.height);
      const Bytes pixels = bytes_of(decoded[i].tensor(holoscan::schema::ImageTensorFields::data));
      const double quality =
          psnr(render_test_frame(c.width, c.height, i), pixels, c.width, c.height, 3);
      VERIFY_MSG(quality > 34.0, name + ": PSNR " + std::to_string(quality));
    }
  }
}

// A change of size starts a new stream with a keyframe, which the decoder follows; refused when
// allow_resolution_change is off.
void check_size_change(Codec codec) {
  std::vector<holoscan::schema::ImageT> images;
  for (std::uint32_t i = 0; i < 4; ++i) images.push_back(test_image(640, 480, i));
  for (std::uint32_t i = 4; i < 8; ++i) images.push_back(test_image(800, 600, i));
  const auto frames = run_encoder(encoder_config(codec, 100), images);
  VERIFY(frames.size() == 8);
  for (std::uint32_t i = 0; i < frames.size(); ++i) {
    const auto meta = frames[i].metadata();
    VERIFY(meta.keyframe == (i == 0 || i == 4));
    VERIFY(meta.frame_index == i % 4);
    VERIFY(meta.width == (i < 4 ? 640 : 800));
  }
  const auto decoded = run_decoder(VideoDecoderConfig{}, as_input(frames));
  VERIFY_MSG(decoded.size() == 8, std::to_string(decoded.size()) + " images across the change");
  for (std::uint32_t i = 0; i < decoded.size(); ++i) {
    VERIFY(decoded[i].metadata().width == (i < 4 ? 640 : 800));
    VERIFY(decoded[i].metadata().header->capture_timestamp_ns == kFirstCapture + kFramePeriod * i);
  }
  VideoEncoderConfig strict = encoder_config(codec, 100);
  strict.allow_resolution_change = false;
  (void)run_encoder(strict, images, false, true);
}

// Empty access units and those before the first keyframe give no image.
void check_stream_edges(Codec codec) {
  std::vector<holoscan::schema::ImageT> images;
  for (std::uint32_t i = 0; i < 8; ++i) images.push_back(test_image(640, 480, i));
  auto inputs = as_input(run_encoder(encoder_config(codec, 4), images));
  VERIFY(inputs.size() == 8);
  if (inputs.size() != 8) return;
  // Join at frame 1: frames 1 to 3 cannot be decoded; an empty access unit in the middle.
  inputs.erase(inputs.begin());
  schema::EncodedVideoFrameT empty = inputs[4];
  empty.data = host_tensor({}, {0});
  inputs.insert(inputs.begin() + 4, empty);
  const auto decoded = run_decoder(VideoDecoderConfig{}, inputs);
  VERIFY_MSG(decoded.size() == 4, std::to_string(decoded.size()) + " images");
  for (std::uint32_t i = 0; i < decoded.size(); ++i) {
    VERIFY(decoded[i].metadata().header->capture_timestamp_ns ==
           kFirstCapture + kFramePeriod * (i + 4));
  }
}

void check_configs() {
  const auto encoder_rejects = [](VideoEncoderConfig config) {
    return test::throws<std::invalid_argument>([&] { VideoEncoderOp op(config); });
  };
  VideoEncoderConfig config;
  config.max_width = 100;
  VERIFY(encoder_rejects(config));
  config = {};
  config.settings.gop = 0;
  VERIFY(encoder_rejects(config));
  VideoDecoderConfig decoder;
  decoder.max_height = kMaxFrameSide + 1;
  VERIFY(test::throws<std::invalid_argument>([&] { VideoDecoderOp op(decoder); }));
}

bool usable(Codec codec) {
  const auto any = [](const std::vector<BackendStatus>& statuses) {
    return std::any_of(statuses.begin(), statuses.end(),
                       [](const BackendStatus& s) { return s.available; });
  };
  return any(probe_encoder_backends(codec)) && any(probe_decoder_backends(codec));
}

}  // namespace
}  // namespace holoscan::codec

int main() {
  using namespace holoscan::codec;  // NOLINT(build/namespaces)
  std::vector<test::TestCase> cases{{"configs", check_configs}};
  for (const Codec codec : {Codec::kH264, Codec::kH265}) {
    if (!usable(codec)) {
      std::cout << codec_name(codec) << ": no backend encodes and decodes it here\n";
      continue;
    }
    const std::string name(codec_name(codec));
    cases.push_back({name + ": round trip, headers, file", [=] { check_round_trip(codec); }});
    cases.push_back({name + ": input encodings and memory", [=] { check_inputs(codec); }});
    cases.push_back({name + ": size change", [=] { check_size_change(codec); }});
    cases.push_back({name + ": empty and undecodable", [=] { check_stream_edges(codec); }});
  }
  if (cases.size() == 1) {
    std::cout << "SKIPPED: no backend encodes and decodes here\n";
    return test::kSkipped;
  }
  return test::run(cases);
}
