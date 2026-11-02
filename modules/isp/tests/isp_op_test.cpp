// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// IspOp through the SDK's operator test harness, every output checked bit for
// bit against IspPipeline's CPU backend run on the same raw frames: BAYER_*
// input in device and host memory; MONO16 with the configured pattern;
// packed RAW10; every output format; the
// histogram and statistics ports; the cpu and auto backends; the capture
// time, frame id, header, exposure and gain carried to the outputs; and
// unusable inputs and settings rejected.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/isp/isp_op.hpp>
#include <holoscan/isp/isp_pipeline.hpp>
#include <holoscan/isp/synthetic_raw.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "scene_support.hpp"
#include "tensor_support.hpp"
#include "test_support.hpp"

namespace holoscan::isp {
namespace {

using test::make;
using test::throws;
using ImageSnapshot = holoscan::testing::Snapshot<holoscan::schema::ImageT>;
using TensorSnapshot = holoscan::testing::Snapshot<holoscan::Tensor>;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};
constexpr int kWidth = 96;
constexpr int kHeight = 64;
constexpr std::int64_t kExposureNs = 8'000'000;
constexpr float kGain = 2.0F;

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

holoscan::schema::ImageEncoding bayer(BayerPattern pattern) {
  switch (pattern) {
    case BayerPattern::kBGGR:
      return holoscan::schema::ImageEncoding_BAYER_BGGR;
    case BayerPattern::kGRBG:
      return holoscan::schema::ImageEncoding_BAYER_GRBG;
    case BayerPattern::kGBRG:
      return holoscan::schema::ImageEncoding_BAYER_GBRG;
    case BayerPattern::kRGGB:
      break;
  }
  return holoscan::schema::ImageEncoding_BAYER_RGGB;
}

// A raw image descriptor of a camera called "camera", with an exposure and gain.
holoscan::schema::ImageT raw_image(holoscan::Tensor tensor, int width, int height,
                                   holoscan::schema::ImageEncoding encoding, int bits) {
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding;
  image.significant_bits = static_cast<std::uint8_t>(bits);
  image.data = std::make_shared<holoscan::Tensor>(std::move(tensor));
  image.header = std::make_shared<holoscan::schema::HeaderT>();
  image.header->frame_id = "camera";
  image.exposure_time_ns = kExposureNs;
  image.gain = kGain;
  return image;
}

SyntheticRawConfig camera(BayerPattern pattern, RawFormat format, int bits, std::uint32_t frame) {
  SyntheticRawConfig config;
  config.width = kWidth;
  config.height = kHeight;
  config.pattern = pattern;
  config.format = format;
  config.bit_depth = bits;
  const float black = static_cast<float>(1 << (bits - 4));
  config.black_levels = {black, black, black, black};
  config.vignetting = 0.3F;
  config.frame = frame;
  return config;
}

// The frames' expected outputs: the CPU backend over them, in order.
std::vector<test::Result> reference(const IspParams& params,
                                    const std::vector<test::RawFrame>& frames,
                                    OutputFormat format) {
  IspPipeline pipeline(Backend::kCpu, params);
  std::vector<test::Result> results;
  for (const test::RawFrame& frame : frames) {
    results.push_back(test::run(pipeline, frame, format, true,
                                FrameExposure{.exposure_ns = kExposureNs, .gain = kGain}));
  }
  return results;
}

template <typename Snapshot>
std::vector<unsigned char> bytes_of(const Snapshot& tensor) {
  std::vector<unsigned char> bytes(tensor.bytes().size());
  std::memcpy(bytes.data(), tensor.bytes().data(), bytes.size());
  return bytes;
}

// Checks the image, histogram and statistics of every frame against `expected`.
void check_outputs(const holoscan::testing::TestResult& result,
                   const std::vector<test::Result>& expected, OutputFormat format, bool stats) {
  const auto images = result.snapshots<holoscan::schema::ImageT>("image");
  VERIFY_MSG(images.size() == expected.size(), std::to_string(images.size()) + " images");
  for (std::size_t i = 0; i < images.size() && i < expected.size(); ++i) {
    const holoscan::schema::ImageT descriptor = images[i].metadata();
    const auto& tensor = images[i].tensor(holoscan::schema::ImageTensorFields::data);
    VERIFY(descriptor.width == kWidth && descriptor.height == kHeight);
    VERIFY(tensor.shape() == (std::vector<std::int64_t>{kHeight, kWidth, output_channels(format)}));
    VERIFY_MSG(bytes_of(tensor) == expected[i].image,
               "frame " + std::to_string(i) + " differs from the CPU backend");
    // The camera's header and capture settings, and the sample's capture time and frame id.
    VERIFY(descriptor.header && descriptor.header->frame_id == "camera");
    VERIFY(descriptor.exposure_time_ns == kExposureNs && descriptor.gain == kGain);
    const holoscan::SampleMetadata& metadata = images[i].sample_metadata();
    VERIFY(metadata.capture_timestamp_ns == capture_time(static_cast<std::uint32_t>(i)).count());
    VERIFY(metadata.frame_id == 7U + i);
  }
  if (!stats) {
    return;
  }
  const auto histograms = result.snapshots<holoscan::Tensor>("histogram");
  const auto statistics = result.snapshots<holoscan::Tensor>("stats");
  VERIFY(histograms.size() == expected.size() && statistics.size() == expected.size());
  for (std::size_t i = 0; i < histograms.size() && i < statistics.size() && i < expected.size();
       ++i) {
    VERIFY(histograms[i].shape() ==
           (std::vector<std::int64_t>{kHistogramChannels, kHistogramBins}));
    std::vector<std::uint32_t> histogram(expected[i].histogram.size());
    VERIFY(histograms[i].bytes().size() == histogram.size() * 4U);
    std::memcpy(histogram.data(), histograms[i].bytes().data(),
                std::min(histograms[i].bytes().size(), histogram.size() * 4U));
    VERIFY_MSG(histogram == expected[i].histogram, "histograms differ");
    std::vector<float> values(stats::kStatsSize);
    VERIFY(statistics[i].bytes().size() == values.size() * 4U);
    std::memcpy(values.data(), statistics[i].bytes().data(),
                std::min(statistics[i].bytes().size(), values.size() * 4U));
    for (int k = 0; k < stats::kStatsSize; ++k) {
      VERIFY_MSG(std::abs(values[k] - expected[i].stats[k]) <=
                     1e-5F * std::max(1.0F, std::abs(expected[i].stats[k])),
                 "stats[" + std::to_string(k) + "] differs");
    }
    VERIFY(statistics[i].sample_metadata().frame_id == 7U + i);
  }
}

struct Input {
  holoscan::schema::ImageT image;
  bool device{};
};

// Runs IspOp on `inputs`, numbered from frame id 7, and checks against the CPU backend.
void run_and_check(const IspOpConfig& config, const std::vector<test::RawFrame>& frames,
                   std::vector<Input> inputs, const std::string& what) {
  auto h = holoscan::testing::create_operator_test<IspOp>(config);
  std::vector<holoscan::testing::TestSample<holoscan::schema::ImageT>> samples;
  bool device = false;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    device = device || inputs[i].device;
    auto sample = holoscan::testing::sample_at(std::move(inputs[i].image),
                                               capture_time(static_cast<std::uint32_t>(i)));
    sample.metadata.frame_id = 7U + i;
    samples.push_back(std::move(sample));
  }
  h.push_samples("raw", std::move(samples));
  if (device) {
    h.input_source_device("raw", holoscan::DeviceId{0});
  }
  const bool cuda = resolve_backend(config.backend) == Backend::kCuda;
  if (cuda) {
    h.output_device("image", holoscan::DeviceId{0});
  }
  h.capture_snapshot<holoscan::schema::ImageT>("image");
  if (config.stats) {
    if (cuda) {
      h.output_device("histogram", holoscan::DeviceId{0});
      h.output_device("stats", holoscan::DeviceId{0});
    }
    h.capture_snapshot<holoscan::Tensor>("histogram");
    h.capture_snapshot<holoscan::Tensor>("stats");
  }
  const auto result = h.run();
  VERIFY_MSG(result.ok(), what + ": " + result.describe_failure());
  if (!result.ok()) return;
  check_outputs(result, reference(config.params, frames, config.output_format),
                config.output_format, config.stats);
}

// Synthetic frames as uint16 or uint8 tensors, in device or pageable host memory.
void bayer_input(Backend backend, bool device, RawFormat format, int bits, OutputFormat output,
                 bool stats) {
  const BayerPattern pattern =
      format == RawFormat::kUint8 ? BayerPattern::kBGGR : BayerPattern::kGRBG;
  std::vector<test::RawFrame> frames;
  std::vector<Input> inputs;
  for (std::uint32_t f = 0; f < 3; ++f) {
    frames.push_back(test::render(camera(pattern, format, bits, f)));
    const test::RawFrame& frame = frames.back();
    const bool u8 = format == RawFormat::kUint8;
    inputs.push_back(Input{raw_image(make({kHeight, kWidth}, u8 ? kUInt8 : kUInt16,
                                          frame.bytes.data(), frame.bytes.size(), device),
                                     kWidth, kHeight, bayer(pattern), bits),
                           device});
  }
  IspOpConfig config;
  config.backend = backend;
  config.params = synthetic_isp_params(frames[0].config, 9);
  config.params.white_balance = WhiteBalanceMode::kGrayWorld;
  config.params.sharpen = 0.5F;
  config.output_format = output;
  config.stats = stats;
  config.max_width = 128;
  config.max_height = 128;
  run_and_check(config, frames, std::move(inputs),
                std::string(backend_name(backend)) + (device ? " device" : " host"));
}

// MONO16 with the pattern configured and 10 significant bits, on the GPU.
void mono16() {
  const test::RawFrame frame = test::render(camera(BayerPattern::kGBRG, RawFormat::kUint16, 10, 0));
  IspOpConfig config;
  config.params = synthetic_isp_params(frame.config, 9);
  config.pattern = BayerPattern::kGBRG;
  config.output_format = OutputFormat::kRgb16;
  config.max_width = kWidth;
  config.max_height = kHeight;
  run_and_check(config, {frame},
                {Input{raw_image(make({kHeight, kWidth, 1}, kUInt16, frame.bytes.data(),
                                      frame.bytes.size(), true),
                                 kWidth, kHeight, holoscan::schema::ImageEncoding_MONO16, 10),
                       true}},
                "MONO16");
}

// MIPI RAW10 in a uint8 [H, row bytes] tensor on the GPU, to float RGB.
void packed_raw10() {
  std::vector<test::RawFrame> frames;
  std::vector<Input> inputs;
  for (std::uint32_t f = 0; f < 2; ++f) {
    frames.push_back(test::render(camera(BayerPattern::kRGGB, RawFormat::kRaw10, 10, f)));
    const test::RawFrame& frame = frames.back();
    inputs.push_back(
        Input{raw_image(make({kHeight, static_cast<std::int64_t>(frame.pitch)}, kUInt8,
                             frame.bytes.data(), frame.bytes.size(), true),
                        kWidth, kHeight, holoscan::schema::ImageEncoding_BAYER_RGGB, 8),
              true});
  }
  IspOpConfig config;
  config.params = synthetic_isp_params(frames[0].config, 9);
  config.params.demosaic = DemosaicMethod::kBilinear;
  config.packing = Packing::kMipiRaw10;
  config.output_format = OutputFormat::kRgb32f;
  config.stats = true;
  config.max_width = kWidth;
  config.max_height = kHeight;
  run_and_check(config, frames, std::move(inputs), "packed RAW10");
}

void auto_backend() {
  IspOpConfig config;
  config.backend = Backend::kAuto;
  int devices = 0;
  const bool gpu = cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
  VERIFY(IspOp(config).backend() == (gpu ? Backend::kCuda : Backend::kCpu));
  config.backend = Backend::kCpu;
  VERIFY(IspOp(config).backend() == Backend::kCpu);
}

// One frame that must fail the run.
void rejected(const IspOpConfig& config, holoscan::schema::ImageT image, const std::string& what) {
  auto h = holoscan::testing::create_operator_test<IspOp>(config);
  std::vector<holoscan::testing::TestSample<holoscan::schema::ImageT>> samples;
  samples.push_back(holoscan::testing::sample_at(std::move(image), capture_time(0)));
  h.push_samples("raw", std::move(samples));
  h.output_device("image", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("image");
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " was accepted");
}

void rejects() {
  const test::RawFrame frame = test::render(camera(BayerPattern::kRGGB, RawFormat::kUint16, 12, 0));
  const auto tensor = [&] {
    return make({kHeight, kWidth}, kUInt16, frame.bytes.data(), frame.bytes.size(), false);
  };
  IspOpConfig config;
  config.max_width = kWidth;
  config.max_height = kHeight;
  rejected(config, raw_image(tensor(), kWidth, kHeight, holoscan::schema::ImageEncoding_MONO16, 12),
           "MONO16 without a configured pattern");
  rejected(
      config,
      raw_image(make({kHeight, kWidth, 3}, kUInt8, frame.bytes.data(), 3 * kWidth * kHeight, false),
                kWidth, kHeight, holoscan::schema::ImageEncoding_RGB8, 8),
      "an RGB8 image");
  IspOpConfig other = config;
  other.pattern = BayerPattern::kBGGR;
  rejected(other,
           raw_image(tensor(), kWidth, kHeight, holoscan::schema::ImageEncoding_BAYER_RGGB, 12),
           "a pattern that disagrees with the configured one");
  IspOpConfig small = config;
  small.max_width = kWidth / 2;
  rejected(small,
           raw_image(tensor(), kWidth, kHeight, holoscan::schema::ImageEncoding_BAYER_RGGB, 12),
           "an image over max_width");
  IspOpConfig dark = config;
  dark.params.black_level = 5000.0F;
  rejected(dark,
           raw_image(tensor(), kWidth, kHeight, holoscan::schema::ImageEncoding_BAYER_RGGB, 12),
           "a black level above 12 bits' white");

  const auto bad = [](auto tweak) {
    IspOpConfig c;
    tweak(c);
    return throws<std::invalid_argument>([&] { IspOp op(c); });
  };
  VERIFY(bad([](IspOpConfig& c) { c.max_width = 1; }));
  VERIFY(bad([](IspOpConfig& c) { c.bit_depth = 17; }));
  VERIFY(bad([](IspOpConfig& c) {
    c.packing = Packing::kMipiRaw10;
    c.bit_depth = 12;
  }));
  VERIFY(bad([](IspOpConfig& c) {
    c.params.gamma = 0.0F;
    c.params.tone_curve = ToneCurve::kGamma;
  }));
}

}  // namespace
}  // namespace holoscan::isp

int main() {
  using holoscan::isp::Backend;
  using holoscan::isp::OutputFormat;
  using holoscan::isp::RawFormat;
  using holoscan::isp::test::TestCase;
  return holoscan::isp::test::run({
      TestCase{"cuda: 12-bit BAYER_GRBG from device memory, RGB8, statistics",
               [] {
                 holoscan::isp::bayer_input(Backend::kCuda, true, RawFormat::kUint16, 12,
                                            OutputFormat::kRgb8, true);
               }},
      TestCase{"cuda: 8-bit BAYER_BGGR from pageable host memory, RGBA8",
               [] {
                 holoscan::isp::bayer_input(Backend::kCuda, false, RawFormat::kUint8, 8,
                                            OutputFormat::kRgba8, false);
               }},
      TestCase{"cuda: MONO16 [H, W, 1] with a configured pattern, RGB16", holoscan::isp::mono16},
      TestCase{"cuda: packed RAW10 from device memory, RGB32F, statistics",
               holoscan::isp::packed_raw10},
      TestCase{"cpu: 12-bit BAYER_GRBG from device memory, statistics in host memory",
               [] {
                 holoscan::isp::bayer_input(Backend::kCpu, true, RawFormat::kUint16, 12,
                                            OutputFormat::kRgb8, true);
               }},
      TestCase{"cpu: 16-bit from host memory, RGB16",
               [] {
                 holoscan::isp::bayer_input(Backend::kCpu, false, RawFormat::kUint16, 16,
                                            OutputFormat::kRgb16, false);
               }},
      TestCase{"auto picks cuda where there is a GPU", holoscan::isp::auto_backend},
      TestCase{"unusable inputs and settings are rejected", holoscan::isp::rejects},
  });
}
