// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "report_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/codec/test_video.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

namespace codec_app {
namespace {

using std::chrono_literals::operator""ns;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
// Text runs: the two labels and the statistics line, each at most kTextBytes long.
constexpr std::int64_t kTextRuns = 3;
constexpr std::int64_t kTextBytes = 96;
using holoscan::visualization::kTextRunHeaderBytes;
constexpr std::int64_t kTextRow = static_cast<std::int64_t>(kTextRunHeaderBytes) + kTextBytes;

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, message};
}

// Copies a device tensor's bytes to `host`.
holoscan::expected<void, holoscan::Error> to_host(const holoscan::Tensor& tensor,
                                                  std::vector<std::uint8_t>& host) {
  if (auto ready = tensor.synchronize_for_host(); !ready) {
    return ready;
  }
  host.resize(static_cast<std::size_t>(tensor.nbytes()));
  const DLDevice device = tensor.device();
  const bool on_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  if (!on_device) {
    std::memcpy(host.data(), tensor.data(), host.size());
    return {};
  }
  if (const cudaError_t status =
          cudaMemcpy(host.data(), tensor.data(), host.size(), cudaMemcpyDeviceToHost);
      status != cudaSuccess) {
    return holoscan::make_unexpected(failure(cudaGetErrorString(status)));
  }
  return {};
}

void write_run(std::uint8_t* row, float x, float y, float size, std::uint32_t color,
               const std::string& text) {
  holoscan::visualization::TextRunHeader header{.x = x, .y = y, .size_px = size, .color = color};
  std::memcpy(row, &header, sizeof(header));
  std::uint8_t* bytes = row + sizeof(header);
  std::memset(bytes, 0, kTextBytes);
  std::memcpy(bytes, text.data(), std::min<std::size_t>(text.size(), kTextBytes));
}

std::string format(const char* pattern, double a, double b) {
  std::array<char, 64> buffer{};
  std::snprintf(buffer.data(), buffer.size(), pattern, a, b);
  return buffer.data();
}

// PSNR of one half of a rendered RGBA frame (`width` x `height` at column `x0`) against an RGB
// source frame, over the rows between the labels and the statistics line, every other pixel.
double half_psnr(const std::vector<std::uint8_t>& rgba, int width, int height, int x0,
                 const std::vector<std::uint8_t>& rgb) {
  const std::size_t stride = static_cast<std::size_t>(2 * width) * 4;
  double squared = 0.0;
  std::size_t count = 0;
  for (int y = height / 10; y < height * 85 / 100; y += 2) {
    for (int x = 0; x < width; x += 2) {
      const std::uint8_t* r = rgba.data() + y * stride + static_cast<std::size_t>(x0 + x) * 4;
      const std::uint8_t* s = rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3;
      for (int c = 0; c < 3; ++c) {
        const double d = static_cast<double>(r[c]) - static_cast<double>(s[c]);
        squared += d * d;
      }
      count += 3;
    }
  }
  const double mse = squared / static_cast<double>(count);
  return mse == 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

}  // namespace

CodecReportOp::CodecReportOp(CodecReportConfig config) : config_(std::move(config)) {}

void CodecReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(bitstream, "bitstream").queue_depth(4U);
  spec.input(decoded, "decoded").queue_depth(4U);
  if (config_.text) {
    spec.output(stats, "stats")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
                    .dtype = kUInt8,
                    .rank = 2U,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                },
            .bounds = holoscan::tensor_bounds(static_cast<std::uint64_t>(kTextRuns * kTextRow)),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
  }
}

holoscan::Contract CodecReportOp::contract() const {
  // An image and its access unit carry the same capture time.
  holoscan::Contract result;
  result.trigger(
      holoscan::OnAligned{.ports = holoscan::PortList{{bitstream, decoded}}, .window = 0ns});
  result.input(bitstream, holoscan::Exact{});
  result.input(decoded, holoscan::Exact{});
  return result;
}

void CodecReportOp::start() {
  seen_ = 0;
  last_sequence_ = 0;
  recent_.clear();
}

holoscan::expected<void, holoscan::Error> CodecReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto unit = bitstream.receive();
  if (!unit) {
    return holoscan::make_unexpected(std::move(unit).error());
  }
  auto image = decoded.receive();
  if (!image) {
    return holoscan::make_unexpected(std::move(image).error());
  }
  CodecReport& report = *config_.report;
  const holoscan::codec::schema::EncodedVideoFrameT& frame = unit->data;
  const holoscan::schema::ImageT& picture = image->data;
  const std::uint64_t bytes = frame.data ? static_cast<std::uint64_t>(frame.data->nbytes()) : 0U;
  const std::uint64_t sequence = picture.header ? picture.header->device_sequence : 0U;
  const auto index = static_cast<std::uint32_t>(sequence - 1U);

  if (sequence == 0 || !picture.data || picture.width != config_.width ||
      picture.height != config_.height ||
      picture.encoding != holoscan::schema::ImageEncoding_RGB8) {
    report.fail("frame " + std::to_string(seen_) + ": not an RGB8 image of the source's size");
  } else if (frame.pts != picture.header->capture_timestamp_ns ||
             (frame.header && frame.header->device_sequence != sequence)) {
    report.fail("frame " + std::to_string(index) + ": its pts or header did not come through");
  }
  if (config_.lossless && sequence != last_sequence_ + 1U) {
    report.fail("frame " + std::to_string(sequence) + " came after " +
                std::to_string(last_sequence_));
  }
  if (sequence <= last_sequence_) {
    report.fail("frame " + std::to_string(sequence) + " is out of order");
  }
  last_sequence_ = sequence;

  double quality = 0.0;
  if (picture.data && sequence != 0) {
    if (auto copied = to_host(*picture.data, host_); !copied) {
      return copied;
    }
    const auto source = holoscan::codec::render_test_frame(config_.width, config_.height, index);
    quality = host_.size() == source.size()
                  ? holoscan::codec::psnr(source, host_, config_.width, config_.height, 3)
                  : 0.0;
    if (config_.min_psnr > 0.0 && quality < config_.min_psnr) {
      report.fail("frame " + std::to_string(index) + ": PSNR " + std::to_string(quality) +
                  " dB, below " + std::to_string(config_.min_psnr));
    }
  }

  // Bitrate over the last second of frames.
  const std::int64_t now = frame.pts;
  recent_.emplace_back(now, bytes);
  std::uint64_t window_bytes = 0;
  while (!recent_.empty() && recent_.front().first < now - 1'000'000'000) recent_.pop_front();
  for (const auto& [time, size] : recent_) window_bytes += size;
  const double span_s =
      recent_.size() > 1 ? static_cast<double>(now - recent_.front().first) / 1e9 : 0.0;
  const double mbps =
      span_s > 0.0 ? static_cast<double>(window_bytes - recent_.front().second) * 8.0 / span_s / 1e6
                   : 0.0;
  {
    const std::scoped_lock lock(report.mutex);
    report.psnr_sum += quality;
    report.psnr_min = std::min(report.psnr_min, quality);
    report.bytes += bytes;
    report.keyframes += frame.keyframe ? 1U : 0U;
  }
  ++seen_;
  report.latest.store(index);
  report.frames.fetch_add(1U, std::memory_order_release);

  if (config_.text) {
    auto loan = stats.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{kTextRuns, kTextRow}, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    const std::string codec = config_.codec == holoscan::codec::Codec::kH264 ? "H.264" : "H.265";
    const std::string line = codec + " on " + config_.backend +
                             format("   %.2f Mbit/s   PSNR %.1f dB", mbps, quality) + "   frame " +
                             std::to_string(index) + (frame.keyframe ? "   keyframe" : "");
    auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
      auto* rows = writer.data_as<std::uint8_t>();
      write_run(rows, 0.01F, 0.015F, 22.0F, 0xFFFFFFFFU, "original");
      write_run(rows + kTextRow, 0.51F, 0.015F, 22.0F, 0xFFFFFFFFU, "decoded");
      write_run(rows + 2 * kTextRow, 0.01F, 0.93F, 22.0F, 0xFF40FFFFU, line);
      return {};
    });
    if (!committed) {
      return committed;
    }
    // With the frame's capture time, by which HoloViz picks the newest text for a frame.
    if (auto sent = stats.emit(std::move(*loan), holoscan::sensor_io::derive_from(*image)); !sent) {
      return sent;
    }
  }
  if (config_.expected_frames != 0 &&
      (config_.lossless ? seen_ == config_.expected_frames : sequence >= config_.expected_frames)) {
    context.request_completion();
  }
  return {};
}

RenderCheckOp::RenderCheckOp(RenderCheckConfig config) : config_(std::move(config)) {}

void RenderCheckOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(rendered, "rendered").queue_depth(2U);
}

holoscan::Contract RenderCheckOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{rendered});
  return result;
}

void RenderCheckOp::stop() {
  if (config_.image_file.empty() || host_.empty()) {
    return;
  }
  std::ofstream out(config_.image_file, std::ios::binary);
  out << "P6\n" << 2 * config_.width << ' ' << config_.height << "\n255\n";
  for (std::size_t p = 0; p + 3 < host_.size(); p += 4) {
    out.write(reinterpret_cast<const char*>(host_.data() + p), 3);
  }
}

holoscan::expected<void, holoscan::Error> RenderCheckOp::compute(
    holoscan::ExecutionContext& /*context*/) {
  auto frame = rendered.receive();
  if (!frame) {
    return holoscan::make_unexpected(std::move(frame).error());
  }
  CodecReport& report = *config_.report;
  const holoscan::schema::ImageT& image = frame->data;
  if (!image.data || image.width != 2 * config_.width || image.height != config_.height) {
    report.fail("a rendered frame is not " + std::to_string(2 * config_.width) + "x" +
                std::to_string(config_.height));
    return {};
  }
  if (auto copied = to_host(*image.data, host_); !copied) {
    return copied;
  }
  // The decoded frame is the latest reported or one just before it; the original is the one
  // HoloViz paired with it, or one shortly before or after.
  const auto latest = static_cast<std::int64_t>(report.latest.load());
  double best_original = 0.0;
  double best_decoded = 0.0;
  for (std::int64_t candidate = std::max<std::int64_t>(0, latest - 8); candidate <= latest + 6;
       ++candidate) {
    const auto source = holoscan::codec::render_test_frame(config_.width, config_.height,
                                                           static_cast<std::uint32_t>(candidate));
    best_original =
        std::max(best_original, half_psnr(host_, config_.width, config_.height, 0, source));
    best_decoded = std::max(best_decoded,
                            half_psnr(host_, config_.width, config_.height, config_.width, source));
  }
  // The original half is the source frame, unchanged; the decoded half as good as the stream.
  // HoloViz may pair a decoded frame with an original from outside the window searched (at the
  // start, before the first original, and at the end), so main() asks most frames to match.
  const bool matched = best_original >= 40.0 && best_decoded >= config_.min_psnr;
  if (!matched) {
    HOLOSCAN_LOG_INFO(
        "RenderCheckOp: a rendered frame: original half {:.1f} dB, decoded half "
        "{:.1f} dB",
        best_original, best_decoded);
  }
  const std::scoped_lock lock(report.mutex);
  ++report.rendered;
  report.rendered_matched += matched ? 1U : 0U;
  return {};
}

}  // namespace codec_app
