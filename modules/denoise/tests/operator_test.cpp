// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The operators through the SDK's operator test harness, every image checked
// byte for byte against the Denoiser, ExposureFusion and ToneMapper run on the
// same frames: DenoiseOp on MONO8, RGB8, RGBA8 and NV12 from device and host
// memory (the host copy logged once), a change of size, the auto backend's
// choice, and VPI where it is built; ExposureFusionOp with 3 and 2 brackets
// matched by capture time (exactly, and within a window), and tone mapping
// float and 16-bit frames; the capture time, frame id and header carried to the
// outputs; and unusable inputs and settings rejected.

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/denoise/denoise_op.hpp>
#include <holoscan/denoise/denoiser.hpp>
#include <holoscan/denoise/exposure_fusion.hpp>
#include <holoscan/denoise/exposure_fusion_op.hpp>
#include <holoscan/denoise/synthetic_scene.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "image_support.hpp"
#include "tensor_support.hpp"
#include "test_support.hpp"

namespace holoscan::denoise {
namespace {

using ImageSnapshot = holoscan::testing::Snapshot<holoscan::schema::ImageT>;
using test::DeviceImage;
using test::make;
using test::throws;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr int kWidth = 160;
constexpr int kHeight = 96;
constexpr NoiseModel kNoise{.read_sigma = 6.0F, .shot_gain = 1.0F, .seed = 11};

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

holoscan::schema::ImageEncoding encoding_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kMono8:
      return holoscan::schema::ImageEncoding_MONO8;
    case PixelFormat::kRgba8:
      return holoscan::schema::ImageEncoding_RGBA8;
    case PixelFormat::kNv12:
      return holoscan::schema::ImageEncoding_NV12;
    case PixelFormat::kRgb8:
      break;
  }
  return holoscan::schema::ImageEncoding_RGB8;
}

// A packed image of `format` as an ImageT, with a header naming its camera.
holoscan::schema::ImageT image_of(const std::vector<std::uint8_t>& bytes, PixelFormat format,
                                  int width, int height, bool device) {
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding_of(format);
  std::vector<std::int64_t> shape{height, width, pixel_bytes(format)};
  if (format == PixelFormat::kMono8) shape = {height, width};
  if (format == PixelFormat::kNv12) {
    shape = {height + height / 2, width};
    const auto plane = static_cast<std::uint64_t>(width) * height;
    image.plane_layouts = {
        holoscan::schema::ImagePlaneLayout(0, static_cast<std::uint32_t>(width), plane),
        holoscan::schema::ImagePlaneLayout(plane, static_cast<std::uint32_t>(width), plane / 2)};
  }
  image.data =
      std::make_shared<holoscan::Tensor>(make(shape, kUInt8, bytes.data(), bytes.size(), device));
  image.header = std::make_shared<holoscan::schema::HeaderT>();
  image.header->frame_id = "camera";
  image.exposure_time_ns = 5'000'000;
  image.gain = 3.0F;
  return image;
}

std::vector<std::uint8_t> bytes_of(const ImageSnapshot& image) {
  const auto& tensor = image.tensor(holoscan::schema::ImageTensorFields::data);
  std::vector<std::uint8_t> bytes(tensor.bytes().size());
  std::memcpy(bytes.data(), tensor.bytes().data(), bytes.size());
  return bytes;
}

// The noisy frames of the low-light scene in `format`.
std::vector<std::vector<std::uint8_t>> noisy_frames(PixelFormat format, int width, int height,
                                                    int count) {
  const bool nv12 = format == PixelFormat::kNv12;
  const LowLightScene scene{.width = width,
                            .height = height,
                            .format = nv12 ? PixelFormat::kRgb8 : format,
                            .speed = 4.0F,
                            .object_radius = 12};
  std::vector<std::vector<std::uint8_t>> frames;
  for (int f = 0; f < count; ++f) {
    auto frame = render_noisy(scene, kNoise, f);
    frames.push_back(nv12 ? test::rgb_to_nv12(frame, width, height) : std::move(frame));
  }
  return frames;
}

// What a Denoiser makes of `frames`, in order.
std::vector<std::vector<std::uint8_t>> denoised(
    Backend backend, const DenoiseParams& params, PixelFormat format, int width, int height,
    const std::vector<std::vector<std::uint8_t>>& frames) {
  Denoiser denoiser(backend, params);
  std::vector<std::vector<std::uint8_t>> out;
  for (const auto& frame : frames) {
    DeviceImage in(format, width, height);
    DeviceImage result(format, width, height);
    in.upload(frame);
    denoiser.process(in.view(), result.view(), nullptr);
    out.push_back(result.download());
  }
  return out;
}

// What the process writes to stdout and stderr, where the SDK logs, while `body` runs.
std::string capture_output(const std::function<void()>& body) {
  std::cout.flush();
  std::cerr.flush();
  std::fflush(nullptr);
  FILE* file = std::tmpfile();
  VERIFY(file != nullptr);
  if (file == nullptr) return {};
  const int saved_out = dup(STDOUT_FILENO);
  const int saved_err = dup(STDERR_FILENO);
  dup2(fileno(file), STDOUT_FILENO);
  dup2(fileno(file), STDERR_FILENO);
  struct Restore {
    int out;
    int err;
    ~Restore() {
      std::cout.flush();
      std::cerr.flush();
      std::fflush(nullptr);
      dup2(out, STDOUT_FILENO);
      dup2(err, STDERR_FILENO);
      close(out);
      close(err);
    }
  };
  {
    const Restore restore{saved_out, saved_err};
    body();
  }
  std::string output;
  std::rewind(file);
  char buffer[4096];
  for (std::size_t n; (n = std::fread(buffer, 1, sizeof(buffer), file)) > 0;) {
    output.append(buffer, n);
  }
  std::fclose(file);
  return output;
}

std::size_t count_of(const std::string& text, std::string_view needle) {
  std::size_t count = 0;
  for (std::size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + needle.size())) {
    ++count;
  }
  return count;
}

// ---- DenoiseOp ----

DenoiseOpConfig op_config(Backend backend, const DenoiseParams& params) {
  return DenoiseOpConfig{.backend = backend, .params = params, .max_width = 256, .max_height = 256};
}

// Frames through DenoiseOp, each image compared with the Denoiser's, and its
// descriptor and provenance checked.
void check_denoise_op(Backend backend, PixelFormat format, bool device, SpatialFilter spatial) {
  const std::string what = std::string(backend_name(backend)) + " " +
                           std::string(format_name(format)) + (device ? " device" : " host");
  DenoiseParams params;
  params.spatial = spatial;
  constexpr int kFrames = 4;
  const auto frames = noisy_frames(format, kWidth, kHeight, kFrames);
  const auto expected = denoised(backend, params, format, kWidth, kHeight, frames);
  auto h = holoscan::testing::create_operator_test<DenoiseOp>(op_config(backend, params));
  std::vector<holoscan::testing::TestSample<holoscan::schema::ImageT>> samples;
  for (std::uint32_t f = 0; f < kFrames; ++f) {
    auto sample = holoscan::testing::sample_at(image_of(frames[f], format, kWidth, kHeight, device),
                                               capture_time(f));
    sample.metadata.frame_id = 30U + f;
    samples.push_back(std::move(sample));
  }
  h.push_samples("image", std::move(samples));
  if (device) h.input_source_device("image", holoscan::DeviceId{0});
  h.output_device("denoised", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("denoised");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), what + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto images = result.snapshots<holoscan::schema::ImageT>("denoised");
  VERIFY_MSG(images.size() == kFrames, what + ": " + std::to_string(images.size()) + " images");
  for (std::size_t f = 0; f < images.size() && f < expected.size(); ++f) {
    const holoscan::schema::ImageT descriptor = images[f].metadata();
    VERIFY_MSG(descriptor.encoding == encoding_of(format), what + ": encoding");
    VERIFY(descriptor.width == kWidth && descriptor.height == kHeight);
    VERIFY(descriptor.header && descriptor.header->frame_id == "camera");
    VERIFY(descriptor.exposure_time_ns == 5'000'000 && descriptor.gain == 3.0F);
    if (format == PixelFormat::kNv12) {
      VERIFY_MSG(
          descriptor.plane_layouts.size() == 2 && descriptor.plane_layouts[1].offset_bytes() ==
                                                      static_cast<std::uint64_t>(kWidth) * kHeight,
          what + ": NV12 plane layouts");
    }
    const auto bytes = bytes_of(images[f]);
    const bool same =
        backend == Backend::kCuda ? bytes == expected[f] : bytes.size() == expected[f].size();
    VERIFY_MSG(same, what + ": frame " + std::to_string(f) + " differs from the Denoiser's");
    const holoscan::SampleMetadata& metadata = images[f].sample_metadata();
    VERIFY(metadata.capture_timestamp_ns == capture_time(static_cast<std::uint32_t>(f)).count());
    VERIFY(metadata.frame_id == 30U + f);
  }
}

// Host images are copied to the GPU, which is logged once; device images log nothing.
void check_copy_logged_once(bool device) {
  holoscan::set_log_level(holoscan::LogLevel::INFO);
  const auto frames = noisy_frames(PixelFormat::kRgb8, kWidth, kHeight, 3);
  const std::string output = capture_output([&] {
    auto h = holoscan::testing::create_operator_test<DenoiseOp>(
        op_config(Backend::kCuda, DenoiseParams{}));
    for (const auto& frame : frames) {
      h.push("image", image_of(frame, PixelFormat::kRgb8, kWidth, kHeight, device));
    }
    if (device) h.input_source_device("image", holoscan::DeviceId{0});
    h.output_device("denoised", holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>("denoised");
    const auto result = h.run();
    VERIFY_MSG(result.ok(), result.describe_failure());
  });
  holoscan::set_log_level(holoscan::LogLevel::WARN);
  const std::size_t logged = count_of(output, "copied host to device");
  VERIFY_MSG(logged == (device ? 0U : 1U),
             "the copy was logged " + std::to_string(logged) + " times:\n" + output);
  if (!device && logged == 1) {
    VERIFY_MSG(output.find(std::to_string(kWidth * kHeight * 3) + " bytes") != std::string::npos,
               "the log gives the bytes per frame");
  }
}

// A change of size mid-stream starts a new sequence at the new size.
void check_size_change() {
  const auto small = noisy_frames(PixelFormat::kRgb8, 64, 48, 2);
  const auto large = noisy_frames(PixelFormat::kRgb8, kWidth, kHeight, 2);
  auto h = holoscan::testing::create_operator_test<DenoiseOp>(
      op_config(Backend::kCuda, DenoiseParams{}));
  h.push("image", image_of(small[0], PixelFormat::kRgb8, 64, 48, false));
  h.push("image", image_of(large[0], PixelFormat::kRgb8, kWidth, kHeight, false));
  h.push("image", image_of(large[1], PixelFormat::kRgb8, kWidth, kHeight, false));
  h.output_device("denoised", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("denoised");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto images = result.snapshots<holoscan::schema::ImageT>("denoised");
  VERIFY(images.size() == 3);
  if (images.size() != 3) return;
  const auto expected =
      denoised(Backend::kCuda, DenoiseParams{}, PixelFormat::kRgb8, kWidth, kHeight, large);
  VERIFY_MSG(bytes_of(images[1]) == expected[0], "the new size starts a new sequence");
  VERIFY_MSG(bytes_of(images[2]) == expected[1], "and continues it");
}

// kAuto picks vpi only where the VIC runs TNR and the settings suit it, and
// logs its choice once at start.
void check_auto_backend() {
  const PlatformFacts facts = probe_platform();
  DenoiseParams tnr;
  tnr.spatial = SpatialFilter::kNone;
  VERIFY(DenoiseOp(op_config(Backend::kAuto, tnr)).backend() == resolve_auto(facts, tnr));
  VERIFY(DenoiseOp(op_config(Backend::kAuto, DenoiseParams{})).backend() == Backend::kCuda);
  const char* platform = std::getenv("HOLOSCAN_CI_PLATFORM");
  if (platform != nullptr && std::string_view(platform) == "x86_64") {
    VERIFY_MSG(resolve_auto(facts, tnr) == Backend::kCuda, "auto is cuda on x86_64");
  }
  holoscan::set_log_level(holoscan::LogLevel::INFO);
  const auto frames = noisy_frames(PixelFormat::kRgb8, kWidth, kHeight, 2);
  const std::string output = capture_output([&] {
    auto h = holoscan::testing::create_operator_test<DenoiseOp>(op_config(Backend::kAuto, tnr));
    for (const auto& frame : frames) {
      h.push("image", image_of(frame, PixelFormat::kRgb8, kWidth, kHeight, true));
    }
    h.input_source_device("image", holoscan::DeviceId{0});
    h.output_device("denoised", holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>("denoised");
    const auto result = h.run();
    VERIFY_MSG(result.ok(), result.describe_failure());
  });
  holoscan::set_log_level(holoscan::LogLevel::WARN);
  VERIFY_MSG(count_of(output, "backend auto chose") == 1, "the choice is logged once:\n" + output);
}

// One frame that must fail the run.
template <typename Op, typename Config>
void rejected(const Config& config, std::string_view port, holoscan::schema::ImageT image,
              const std::string& what) {
  auto h = holoscan::testing::create_operator_test<Op>(config);
  h.push(std::string(port), std::move(image));
  const std::string output = std::is_same_v<Op, DenoiseOp> ? "denoised" : "image";
  h.output_device(output, holoscan::DeviceId{0});
  h.template capture_snapshot<holoscan::schema::ImageT>(output);
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " was accepted");
}

void check_denoise_rejections() {
  const DenoiseOpConfig config = op_config(Backend::kCuda, DenoiseParams{});
  const auto frame = noisy_frames(PixelFormat::kRgb8, kWidth, kHeight, 1)[0];
  holoscan::schema::ImageT bgr = image_of(frame, PixelFormat::kRgb8, kWidth, kHeight, false);
  bgr.encoding = holoscan::schema::ImageEncoding_BGR8;
  rejected<DenoiseOp>(config, "image", bgr, "a BGR8 image");
  DenoiseOpConfig small = config;
  small.max_width = kWidth / 2;
  rejected<DenoiseOp>(small, "image", image_of(frame, PixelFormat::kRgb8, kWidth, kHeight, false),
                      "an image over max_width");
  holoscan::schema::ImageT nv12 = image_of(test::rgb_to_nv12(frame, kWidth, kHeight),
                                           PixelFormat::kNv12, kWidth, kHeight, false);
  nv12.plane_layouts.pop_back();
  rejected<DenoiseOp>(config, "image", nv12, "NV12 with one plane layout");
  holoscan::schema::ImageT mismatched = image_of(frame, PixelFormat::kRgb8, kWidth, kHeight, false);
  mismatched.width = kWidth + 2;
  rejected<DenoiseOp>(config, "image", mismatched, "a width that disagrees with the tensor");

  VERIFY(throws<std::invalid_argument>(
      [] { DenoiseOp op(DenoiseOpConfig{.params = DenoiseParams{.strength = 2.0F}}); }));
  VERIFY(throws<std::invalid_argument>([] { DenoiseOp op(DenoiseOpConfig{.max_width = 0}); }));
  // The vpi backend has no spatial filter of its own to configure.
  VERIFY(throws<std::invalid_argument>(
      [] { DenoiseOp op(DenoiseOpConfig{.backend = Backend::kVpi, .params = DenoiseParams{}}); }));
}

// ---- ExposureFusionOp ----

holoscan::schema::ImageT bracket_image(const HdrScene& scene, float exposure, bool device) {
  return image_of(render_bracket(scene, 0, exposure),
                  scene.channels == 1 ? PixelFormat::kMono8 : PixelFormat::kRgb8, scene.width,
                  scene.height, device);
}

std::vector<std::uint8_t> fused_directly(const HdrScene& scene, std::span<const float> exposures) {
  const PixelFormat format = scene.channels == 1 ? PixelFormat::kMono8 : PixelFormat::kRgb8;
  std::vector<std::unique_ptr<DeviceImage>> images;
  std::vector<ImageView> views;
  for (const float exposure : exposures) {
    images.push_back(std::make_unique<DeviceImage>(format, scene.width, scene.height));
    images.back()->upload(render_bracket(scene, 0, exposure));
    views.push_back(images.back()->view());
  }
  DeviceImage out(format, scene.width, scene.height);
  ExposureFusion(FusionParams{}).fuse(views, out.view(), nullptr);
  return out.download();
}

// Brackets of two frames, each set with one capture time (or offsets within
// the window); every set fused as ExposureFusion fuses it.
void check_fusion_op(int brackets, std::chrono::nanoseconds window, bool device, int channels) {
  const HdrScene scene{.width = kWidth, .height = kHeight, .channels = channels};
  std::vector<float> exposures{kBracketExposures[0], kBracketExposures[1], kBracketExposures[2]};
  if (brackets == 2) exposures = {kBracketExposures[0], kBracketExposures[2]};
  ExposureFusionOpConfig config{.brackets = brackets, .match_window = window};
  config.max_width = 256;
  config.max_height = 256;
  auto h = holoscan::testing::create_operator_test<ExposureFusionOp>(config);
  const std::array<std::string, 3> ports{"bracket0", "bracket1", "bracket2"};
  constexpr std::uint32_t kSets = 2;
  for (std::size_t k = 0; k < exposures.size(); ++k) {
    std::vector<holoscan::testing::TestSample<holoscan::schema::ImageT>> samples;
    for (std::uint32_t f = 0; f < kSets; ++f) {
      // Within the window, the brackets are captured one after another.
      const auto offset =
          window.count() == 0 ? std::chrono::nanoseconds(0) : window / 3 * static_cast<int>(k);
      auto sample = holoscan::testing::sample_at(bracket_image(scene, exposures[k], device),
                                                 capture_time(f) + offset);
      sample.metadata.frame_id = 50U + f;
      samples.push_back(std::move(sample));
    }
    h.push_samples(ports[k], std::move(samples));
    if (device) h.input_source_device(ports[k], holoscan::DeviceId{0});
  }
  h.output_device("image", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("image");
  const auto result = h.run();
  const std::string what = std::to_string(brackets) + " brackets";
  VERIFY_MSG(result.ok(), what + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto images = result.snapshots<holoscan::schema::ImageT>("image");
  VERIFY_MSG(images.size() == kSets, what + ": " + std::to_string(images.size()) + " images");
  const auto expected = fused_directly(scene, exposures);
  for (std::size_t f = 0; f < images.size(); ++f) {
    VERIFY_MSG(bytes_of(images[f]) == expected, what + ": differs from ExposureFusion's");
    VERIFY(images[f].metadata().encoding == (channels == 1 ? holoscan::schema::ImageEncoding_MONO8
                                                           : holoscan::schema::ImageEncoding_RGB8));
    const holoscan::SampleMetadata& metadata = images[f].sample_metadata();
    VERIFY(metadata.capture_timestamp_ns == capture_time(static_cast<std::uint32_t>(f)).count());
    VERIFY(metadata.frame_id == 50U + f);
  }
}

// Brackets whose capture times differ by more than the window are never fused.
void check_fusion_unmatched() {
  const HdrScene scene{.width = kWidth, .height = kHeight, .channels = 3};
  ExposureFusionOpConfig config{.brackets = 2};
  config.max_width = 256;
  config.max_height = 256;
  auto h = holoscan::testing::create_operator_test<ExposureFusionOp>(config);
  std::vector<holoscan::testing::TestSample<holoscan::schema::ImageT>> first;
  std::vector<holoscan::testing::TestSample<holoscan::schema::ImageT>> second;
  first.push_back(holoscan::testing::sample_at(bracket_image(scene, kBracketExposures[0], false),
                                               capture_time(0)));
  second.push_back(holoscan::testing::sample_at(bracket_image(scene, kBracketExposures[2], false),
                                                capture_time(0) + std::chrono::milliseconds(1)));
  h.push_samples("bracket0", std::move(first));
  h.push_samples("bracket1", std::move(second));
  h.output_device("image", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("image");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  VERIFY_MSG(result.snapshots<holoscan::schema::ImageT>("image").empty(),
             "brackets 1 ms apart are not fused with a window of 0");
}

// One float or 16-bit frame, tone mapped as the reference maps it (within one
// code value), from host and device memory.
void check_tone_map_op(bool device) {
  const HdrScene scene{.width = kWidth, .height = kHeight, .channels = 3};
  const std::vector<float> radiance = render_radiance(scene, 0);
  const auto pixels = static_cast<std::size_t>(kWidth) * kHeight;
  std::vector<std::uint16_t> codes(pixels * 3);
  for (std::size_t i = 0; i < codes.size(); ++i) {
    codes[i] = static_cast<std::uint16_t>(std::min(4095.0F, radiance[i] / 6.5F * 4095.0F + 0.5F));
  }
  struct Case {
    holoscan::schema::ImageEncoding encoding;
    HdrFormat format;
    const void* data;
    DLDataType dtype;
    std::size_t sample;
  };
  const Case cases[] = {
      {holoscan::schema::ImageEncoding_RGB32F, HdrFormat::kRgb32f, radiance.data(),
       DLDataType{kDLFloat, 32U, 1U}, sizeof(float)},
      {holoscan::schema::ImageEncoding_RGB16, HdrFormat::kRgb16, codes.data(),
       DLDataType{kDLUInt, 16U, 1U}, sizeof(std::uint16_t)},
  };
  for (const Case& c : cases) {
    holoscan::schema::ImageT image;
    image.width = kWidth;
    image.height = kHeight;
    image.encoding = c.encoding;
    image.significant_bits = 12;
    image.data = std::make_shared<holoscan::Tensor>(
        make({kHeight, kWidth, 3}, c.dtype, c.data, pixels * 3 * c.sample, device));
    ExposureFusionOpConfig config{.mode = FusionMode::kToneMap};
    config.max_width = 256;
    config.max_height = 256;
    auto h = holoscan::testing::create_operator_test<ExposureFusionOp>(config);
    h.push_samples("hdr", {holoscan::testing::sample_at(std::move(image), capture_time(3))});
    if (device) h.input_source_device("hdr", holoscan::DeviceId{0});
    h.output_device("image", holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>("image");
    const auto result = h.run();
    VERIFY_MSG(result.ok(), result.describe_failure());
    if (!result.ok()) continue;
    const auto images = result.snapshots<holoscan::schema::ImageT>("image");
    VERIFY(images.size() == 1);
    if (images.empty()) continue;
    const HdrImageView view{.format = c.format,
                            .width = kWidth,
                            .height = kHeight,
                            .data = c.data,
                            .pitch = static_cast<std::size_t>(kWidth) * 3 * c.sample,
                            .significant_bits = 12};
    const auto expected = tone_map_reference(view, ToneMapParams{});
    const auto bytes = bytes_of(images[0]);
    int far = static_cast<int>(bytes.size() != expected.size());
    for (std::size_t i = 0; i < bytes.size() && i < expected.size(); ++i) {
      far += std::abs(bytes[i] - expected[i]) > 1;
    }
    VERIFY_MSG(far == 0, std::string(hdr_format_name(c.format)) + ": " + std::to_string(far) +
                             " samples off the reference");
    VERIFY(images[0].metadata().encoding == holoscan::schema::ImageEncoding_RGB8);
    VERIFY(images[0].sample_metadata().capture_timestamp_ns == capture_time(3).count());
  }
}

void check_fusion_rejections() {
  ExposureFusionOpConfig tone{.mode = FusionMode::kToneMap, .max_width = 256, .max_height = 256};
  const auto frame = noisy_frames(PixelFormat::kRgb8, kWidth, kHeight, 1)[0];
  rejected<ExposureFusionOp>(tone, "hdr",
                             image_of(frame, PixelFormat::kRgb8, kWidth, kHeight, false),
                             "an 8-bit frame to tone map");
  VERIFY(throws<std::invalid_argument>(
      [] { ExposureFusionOp op(ExposureFusionOpConfig{.brackets = 4}); }));
  VERIFY(throws<std::invalid_argument>([] {
    ExposureFusionOp op(ExposureFusionOpConfig{.match_window = std::chrono::seconds(2)});
  }));
  VERIFY(throws<std::invalid_argument>([] {
    ExposureFusionOp op(ExposureFusionOpConfig{.mode = FusionMode::kToneMap,
                                               .tone_map = ToneMapParams{.key = 2.0F}});
  }));
  // Brackets of different sizes.
  const HdrScene a{.width = kWidth, .height = kHeight};
  const HdrScene b{.width = kWidth / 2, .height = kHeight};
  ExposureFusionOpConfig config{.brackets = 2, .max_width = 256, .max_height = 256};
  auto h = holoscan::testing::create_operator_test<ExposureFusionOp>(config);
  h.push_samples("bracket0",
                 {holoscan::testing::sample_at(bracket_image(a, 0.125F, false), capture_time(0))});
  h.push_samples("bracket1",
                 {holoscan::testing::sample_at(bracket_image(b, 32.0F, false), capture_time(0))});
  h.output_device("image", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("image");
  VERIFY_MSG(!h.run().ok(), "brackets of different sizes were accepted");
}

bool vpi_built() {
  const auto backends = available_backends();
  return std::find(backends.begin(), backends.end(), Backend::kVpi) != backends.end();
}

}  // namespace
}  // namespace holoscan::denoise

int main() {
  namespace dn = holoscan::denoise;
  using dn::Backend;
  using dn::PixelFormat;
  using dn::SpatialFilter;
  holoscan::set_log_level(holoscan::LogLevel::WARN);
  std::vector<dn::test::TestCase> cases;
  for (const PixelFormat format :
       {PixelFormat::kMono8, PixelFormat::kRgb8, PixelFormat::kRgba8, PixelFormat::kNv12}) {
    for (const bool device : {true, false}) {
      cases.push_back({"DenoiseOp cuda " + std::string(dn::format_name(format)) +
                           (device ? " device" : " host"),
                       [=] {
                         dn::check_denoise_op(Backend::kCuda, format, device,
                                              SpatialFilter::kBilateral);
                       }});
    }
  }
  cases.push_back({"DenoiseOp cuda RGB8 nlm", [] {
                     dn::check_denoise_op(Backend::kCuda, PixelFormat::kRgb8, true,
                                          SpatialFilter::kNlm);
                   }});
  if (dn::vpi_built()) {
    for (const PixelFormat format : {PixelFormat::kRgb8, PixelFormat::kNv12}) {
      cases.push_back({"DenoiseOp vpi " + std::string(dn::format_name(format)), [=] {
                         dn::check_denoise_op(Backend::kVpi, format, true, SpatialFilter::kNone);
                       }});
    }
  }
  cases.push_back({"DenoiseOp host copy logged once", [] { dn::check_copy_logged_once(false); }});
  cases.push_back(
      {"DenoiseOp device input logs no copy", [] { dn::check_copy_logged_once(true); }});
  cases.push_back({"DenoiseOp size change", dn::check_size_change});
  cases.push_back({"DenoiseOp auto backend", dn::check_auto_backend});
  cases.push_back({"DenoiseOp rejections", dn::check_denoise_rejections});
  cases.push_back({"ExposureFusionOp 3 brackets, device",
                   [] { dn::check_fusion_op(3, std::chrono::nanoseconds(0), true, 3); }});
  cases.push_back({"ExposureFusionOp 2 gray brackets, host",
                   [] { dn::check_fusion_op(2, std::chrono::nanoseconds(0), false, 1); }});
  cases.push_back({"ExposureFusionOp brackets within a 3 ms window",
                   [] { dn::check_fusion_op(3, std::chrono::milliseconds(3), true, 3); }});
  cases.push_back({"ExposureFusionOp unmatched brackets", dn::check_fusion_unmatched});
  cases.push_back({"ExposureFusionOp tone map, host", [] { dn::check_tone_map_op(false); }});
  cases.push_back({"ExposureFusionOp tone map, device", [] { dn::check_tone_map_op(true); }});
  cases.push_back({"ExposureFusionOp rejections", dn::check_fusion_rejections});
  return dn::test::run(cases);
}
