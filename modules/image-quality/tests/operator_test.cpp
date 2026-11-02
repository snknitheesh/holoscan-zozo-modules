// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ImageQualityOp on its own, through the SDK's operator test harness: the
// metrics and flags of host and device images match the CPU reference's, the
// optional tile and overlay outputs, provenance carried to every output, and
// unusable inputs and configurations rejected.

#include <cuda_runtime_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/image_quality/image_quality_op.hpp>
#include <holoscan/image_quality/quality.hpp>
#include <holoscan/image_quality/reference.hpp>
#include <holoscan/image_quality/synthetic.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "test_support.hpp"

namespace holoscan::image_quality {
namespace {

using std::chrono_literals::operator""ms;

constexpr int kWidth = 160;
constexpr int kHeight = 120;

holoscan::schema::ImageEncoding encoding_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kMono8:
      return holoscan::schema::ImageEncoding_MONO8;
    case PixelFormat::kRgb8:
      return holoscan::schema::ImageEncoding_RGB8;
    case PixelFormat::kBgr8:
      return holoscan::schema::ImageEncoding_BGR8;
    case PixelFormat::kRgba8:
      return holoscan::schema::ImageEncoding_RGBA8;
    case PixelFormat::kBgra8:
      return holoscan::schema::ImageEncoding_BGRA8;
  }
  return holoscan::schema::ImageEncoding_MONO8;
}

// `image` as an ImageT, its tensor in host or device memory.
holoscan::schema::ImageT image_message(const HostImage& image, bool device) {
  const int ch = channels(image.format);
  std::vector<std::int64_t> shape{image.height, image.width};
  if (ch != 1) shape.push_back(ch);
  auto tensor = holoscan::make_tensor(
      shape, DLDataType{kDLUInt, 8, 1}, std::nullopt,
      device ? holoscan::MemoryStorageType::kDevice : holoscan::MemoryStorageType::kHost,
      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) throw std::runtime_error(tensor.error().message);
  auto destination = tensor->mutable_data();
  if (!destination) throw std::runtime_error(destination.error().message);
  if (cudaMemcpy(*destination, image.pixels.data(), image.pixels.size(), cudaMemcpyDefault) !=
      cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  holoscan::schema::ImageT message;
  message.width = image.width;
  message.height = image.height;
  message.encoding = encoding_of(image.format);
  message.data = std::make_shared<holoscan::Tensor>(std::move(*tensor));
  return message;
}

std::chrono::nanoseconds capture_time(std::size_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

template <typename T>
std::vector<T> values_of(const holoscan::testing::TensorSnapshot& snapshot) {
  std::vector<T> values(snapshot.bytes().size() / sizeof(T));
  std::memcpy(values.data(), snapshot.bytes().data(), values.size() * sizeof(T));
  return values;
}

// The reports the CPU reference gives for `frames`.
std::vector<QualityReport> expected_reports(const std::vector<HostImage>& frames,
                                            const ImageQualityConfig& config) {
  ReferenceStats stats(config.stats);
  QualityEvaluator evaluator(config.thresholds);
  std::vector<QualityReport> reports;
  for (const HostImage& frame : frames)
    reports.push_back(evaluator.evaluate(stats.gather(frame.view())));
  return reports;
}

// Frames through the operator: the metrics, flags and (when configured) tiles
// and overlay of every frame equal the reference's, with the frame's provenance.
void check_frames(PixelFormat format, bool device, bool extra_outputs) {
  ImageQualityConfig config;
  config.tiles_output = extra_outputs;
  config.overlay_output = extra_outputs;
  config.overlay_cell = 4;
  config.thresholds.obstruction_frames = 2;
  // Tiles of 40x40 pixels, each with the scene's edges.
  config.stats.tiles_x = 4;
  config.stats.tiles_y = 3;
  // A repeat (frozen), then a smudge over two frames (obstructed on the second).
  std::vector<HostImage> frames{render_scene(kWidth, kHeight, 0, format),
                                render_scene(kWidth, kHeight, 0, format)};
  for (std::uint32_t i = 1; i <= 2; ++i) {
    HostImage smudged = render_scene(kWidth, kHeight, i, format);
    apply_smudge(smudged, Smudge{80, 60, 70, 50});
    frames.push_back(std::move(smudged));
  }
  const std::vector<QualityReport> expected = expected_reports(frames, config);
  VERIFY(expected[1].flags == kFrozen && expected[3].flags == kObstructed);

  auto h = holoscan::testing::create_operator_test<ImageQualityOp>(config);
  std::vector<holoscan::testing::TestSample<holoscan::schema::ImageT>> samples;
  for (std::size_t i = 0; i < frames.size(); ++i) {
    holoscan::testing::TestSample<holoscan::schema::ImageT> sample{
        .data = image_message(frames[i], device), .capture_coordinate = capture_time(i)};
    sample.metadata.frame_id = 70U + i;
    samples.push_back(std::move(sample));
  }
  h.push_samples("image", std::move(samples));
  if (device) h.input_source_device("image", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::Tensor>("metrics");
  h.capture_snapshot<holoscan::Tensor>("flags");
  if (extra_outputs) {
    h.capture_snapshot<holoscan::Tensor>("tiles");
    h.capture_snapshot<holoscan::schema::ImageT>("overlay");
  }
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto metrics = result.snapshots<holoscan::Tensor>("metrics");
  const auto flags = result.snapshots<holoscan::Tensor>("flags");
  VERIFY(metrics.size() == frames.size() && flags.size() == frames.size());
  for (std::size_t i = 0; i < metrics.size() && i < flags.size(); ++i) {
    const std::string what = "frame " + std::to_string(i);
    VERIFY(metrics[i].shape() ==
           std::vector<std::int64_t>{static_cast<std::int64_t>(kMetricCount)});
    VERIFY(metrics[i].dtype().code == kDLFloat && metrics[i].dtype().bits == 32);
    VERIFY_MSG(std::memcmp(metrics[i].bytes().data(), expected[i].metrics.data(),
                           sizeof(expected[i].metrics)) == 0,
               what + ": metrics differ from the reference");
    VERIFY(flags[i].shape() == std::vector<std::int64_t>{1});
    VERIFY(flags[i].dtype().code == kDLUInt && flags[i].dtype().bits == 32);
    VERIFY_MSG(values_of<std::uint32_t>(flags[i]) == std::vector<std::uint32_t>{expected[i].flags},
               what + ": flags differ from the reference");
    for (const auto* metadata : {&metrics[i].sample_metadata(), &flags[i].sample_metadata()}) {
      VERIFY_MSG(metadata->capture_timestamp_ns == capture_time(i).count() &&
                     metadata->frame_id == 70U + i,
                 what + ": provenance not carried");
    }
  }
  if (!extra_outputs) return;
  const auto tiles = result.snapshots<holoscan::Tensor>("tiles");
  const auto overlays = result.snapshots<holoscan::schema::ImageT>("overlay");
  VERIFY(tiles.size() == frames.size() && overlays.size() == frames.size());
  for (std::size_t i = 0; i < tiles.size() && i < overlays.size(); ++i) {
    const QualityReport& report = expected[i];
    VERIFY(tiles[i].shape() ==
           (std::vector<std::int64_t>{report.tiles_y, report.tiles_x,
                                      static_cast<std::int64_t>(kTileChannelCount)}));
    VERIFY(values_of<float>(tiles[i]).size() == report.tiles.size());
    VERIFY(std::memcmp(tiles[i].bytes().data(), report.tiles.data(),
                       report.tiles.size() * sizeof(float)) == 0);
    VERIFY(overlays[i].metadata().encoding == holoscan::schema::ImageEncoding_RGBA8);
    const auto& pixels = overlays[i].tensor(holoscan::schema::ImageTensorFields::data);
    const int w = report.tiles_x * 4;
    const int ht = report.tiles_y * 4;
    VERIFY(pixels.shape() == (std::vector<std::int64_t>{ht, w, 4}));
    std::vector<std::uint8_t> drawn(static_cast<std::size_t>(w) * ht * 4);
    draw_overlay(report, 4, drawn.data());
    VERIFY(pixels.bytes().size() == drawn.size() &&
           std::memcmp(pixels.bytes().data(), drawn.data(), drawn.size()) == 0);
    VERIFY(overlays[i].sample_metadata().capture_timestamp_ns == capture_time(i).count());
  }
}

// The overlay of an obstructed tile is magenta; a clear tile's is transparent.
void check_overlay_colors() {
  QualityReport report;
  report.tiles_x = 2;
  report.tiles_y = 1;
  report.tiles.assign(2 * kTileChannelCount, 0.0F);
  report.tiles[static_cast<std::size_t>(TileChannel::kFlags)] = kTileLowTexture | kTileObstructed;
  std::vector<std::uint8_t> rgba(2 * 4 * 4 * 4);
  draw_overlay(report, 4, rgba.data());
  const std::uint8_t* first = rgba.data();
  VERIFY(first[0] == 0xFF && first[1] == 0x00 && first[2] == 0xFF && first[3] == 0x8C);
  const std::uint8_t* gap = rgba.data() + 3 * 4;  // The tile's last column stays clear.
  VERIFY(gap[3] == 0);
  const std::uint8_t* second = rgba.data() + 4 * 4;
  VERIFY(second[3] == 0);
}

void check_rejects_unsupported_encoding() {
  auto h = holoscan::testing::create_operator_test<ImageQualityOp>(ImageQualityConfig{});
  holoscan::schema::ImageT image = image_message(render_scene(kWidth, kHeight, 0), false);
  image.encoding = holoscan::schema::ImageEncoding_YUYV;
  h.push("image", image);
  h.capture_snapshot<holoscan::Tensor>("metrics");
  h.capture_snapshot<holoscan::Tensor>("flags");
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "a YUYV image is rejected");
  VERIFY(result.describe_failure().find("unsupported image encoding") != std::string::npos);
}

void check_rejects_config() {
  const auto rejects = [](auto change) {
    ImageQualityConfig config;
    change(config);
    return test::throws<std::invalid_argument>([&] { ImageQualityOp op(config); });
  };
  VERIFY(rejects([](ImageQualityConfig& c) { c.stats.tiles_x = 0; }));
  VERIFY(rejects([](ImageQualityConfig& c) { c.thresholds.obstruction_frames = 0; }));
  VERIFY(rejects([](ImageQualityConfig& c) { c.overlay_cell = 0; }));
  VERIFY(rejects([](ImageQualityConfig& c) { c.log_interval = std::chrono::milliseconds(-1); }));
  VERIFY(!rejects([](ImageQualityConfig&) {}));
}

}  // namespace
}  // namespace holoscan::image_quality

int main() {
  using holoscan::image_quality::PixelFormat;
  using holoscan::image_quality::test::TestCase;
  namespace iq = holoscan::image_quality;
  std::vector<TestCase> cases;
  for (const PixelFormat format : {PixelFormat::kMono8, PixelFormat::kRgb8, PixelFormat::kBgra8}) {
    for (const bool device : {false, true}) {
      cases.push_back({"frames format " + std::to_string(static_cast<int>(format)) +
                           (device ? " device" : " host"),
                       [=] { iq::check_frames(format, device, false); }});
    }
  }
  cases.push_back({"frames with tiles and overlay host",
                   [] { iq::check_frames(PixelFormat::kRgb8, false, true); }});
  cases.push_back({"frames with tiles and overlay device",
                   [] { iq::check_frames(PixelFormat::kRgba8, true, true); }});
  cases.push_back({"overlay colors", iq::check_overlay_colors});
  cases.push_back({"rejects unsupported encoding", iq::check_rejects_unsupported_encoding});
  cases.push_back({"rejects config", iq::check_rejects_config});
  return holoscan::image_quality::test::run(cases);
}
