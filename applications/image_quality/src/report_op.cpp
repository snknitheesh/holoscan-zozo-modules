// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "report_op.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

namespace image_quality_app {
namespace {

namespace iq = holoscan::image_quality;
using std::chrono_literals::operator""ns;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr std::int64_t kTextHeaderBytes = holoscan::visualization::kTextRunHeaderBytes;
constexpr std::int64_t kTextRowBytes = kTextHeaderBytes + kTextColumns;

holoscan::Error invalid(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, std::move(message)};
}

// A frame's metrics and flags, the frame id they carry, after checking them.
struct Received {
  std::array<float, iq::kMetricCount> metrics{};
  std::uint32_t flags{};
  std::uint64_t frame{};
  holoscan::EmitOptions provenance;
};

holoscan::expected<Received, holoscan::Error> receive(holoscan::Input<holoscan::Tensor>& metrics,
                                                      holoscan::Input<holoscan::Tensor>& flags) {
  auto values = metrics.receive();
  if (!values) {
    return holoscan::make_unexpected(std::move(values).error());
  }
  auto bits = flags.receive();
  if (!bits) {
    return holoscan::make_unexpected(std::move(bits).error());
  }
  const holoscan::Tensor& m = values->data;
  const holoscan::Tensor& f = bits->data;
  if (m.dtype().code != kDLFloat || m.dtype().bits != 32 ||
      m.size() != static_cast<std::int64_t>(iq::kMetricCount) || f.dtype().code != kDLUInt ||
      f.dtype().bits != 32 || f.size() != 1) {
    return holoscan::make_unexpected(invalid("metrics or flags of the wrong type or size"));
  }
  if (auto ready = m.synchronize_for_host(); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  if (auto ready = f.synchronize_for_host(); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  Received result;
  std::memcpy(result.metrics.data(), m.data(), sizeof(result.metrics));
  std::memcpy(&result.flags, f.data(), sizeof(result.flags));
  result.frame = values->metadata.frame_id;
  result.provenance = holoscan::sensor_io::derive_from(*values);
  return result;
}

// Fires when both inputs of one frame (the same capture time) are there.
holoscan::Contract aligned(const holoscan::Input<holoscan::Tensor>& metrics,
                           const holoscan::Input<holoscan::Tensor>& flags) {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{.ports = holoscan::ports(metrics, flags), .window = 0ns});
  result.input(metrics, holoscan::Exact{});
  result.input(flags, holoscan::Exact{});
  return result;
}

}  // namespace

QualityReportOp::QualityReportOp(QualityReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("QualityReportOp needs a report");
  }
}

void QualityReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(metrics, "metrics").queue_depth(2U);
  spec.input(flags, "flags").queue_depth(2U);
}

holoscan::Contract QualityReportOp::contract() const { return aligned(metrics, flags); }

holoscan::expected<void, holoscan::Error> QualityReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto received = receive(metrics, flags);
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  RunReport& report = *config_.report;
  const Segment segment = config_.schedule.segment(received->frame);
  SegmentTally& tally = report.segments[static_cast<std::size_t>(segment)];
  const std::uint32_t expected = config_.schedule.expected_flags(received->frame);
  ++tally.frames;
  tally.flags |= received->flags;
  const auto metric = [&](iq::Metric which) {
    return static_cast<double>(received->metrics[static_cast<std::size_t>(which)]);
  };
  tally.focus += metric(iq::Metric::kFocus);
  tally.mean_luma += metric(iq::Metric::kMeanLuma);
  tally.noise_sigma += metric(iq::Metric::kNoiseSigma);
  tally.color_cast += metric(iq::Metric::kColorCast);
  if (received->flags != expected) {
    ++tally.wrong;
    if (report.mismatches.size() < 10) {
      report.mismatches.push_back(
          "frame " + std::to_string(received->frame) + " (" + std::string(segment_name(segment)) +
          "): flags " + iq::flag_names(received->flags) + ", expected " + iq::flag_names(expected));
    }
  }
  if (++report.frames == config_.expected_frames) {
    context.request_completion();
  }
  return {};
}

std::array<std::string, kTextRows> text_lines(std::uint64_t frame, Segment segment,
                                              std::uint32_t flags,
                                              const std::array<float, iq::kMetricCount>& m) {
  const auto get = [&m](iq::Metric which) {
    return static_cast<double>(m[static_cast<std::size_t>(which)]);
  };
  std::array<std::string, kTextRows> lines;
  char line[kTextColumns + 1];
  lines[0] = "frame " + std::to_string(frame) + "  camera: " + std::string(segment_name(segment));
  std::snprintf(line, sizeof(line), "flags: %s", iq::flag_names(flags).c_str());
  lines[1] = line;
  std::snprintf(line, sizeof(line), "focus %.2f  laplacian var %.0f  tenengrad %.0f",
                get(iq::Metric::kFocus), get(iq::Metric::kLaplacianVariance),
                get(iq::Metric::kTenengrad));
  lines[2] = line;
  std::snprintf(line, sizeof(line), "luma %.0f (p1 %.0f, p99 %.0f)  noise %.1f",
                get(iq::Metric::kMeanLuma), get(iq::Metric::kLumaP1), get(iq::Metric::kLumaP99),
                get(iq::Metric::kNoiseSigma));
  lines[3] = line;
  std::snprintf(line, sizeof(line), "diff %.1f  obstructed %.0f%%  cast %.2f",
                get(iq::Metric::kFrameDifference), 100.0 * get(iq::Metric::kObstructedTileFraction),
                get(iq::Metric::kColorCast));
  lines[4] = line;
  return lines;
}

QualityTextOp::QualityTextOp(Schedule schedule) : schedule_(std::move(schedule)) {}

void QualityTextOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(metrics, "metrics").queue_depth(2U);
  spec.input(flags, "flags").queue_depth(2U);
  spec.output(text, "text")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kHost,
                  .dtype = kUInt8,
                  .rank = 2U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(static_cast<std::uint64_t>(kTextRows * kTextRowBytes)),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract QualityTextOp::contract() const { return aligned(metrics, flags); }

holoscan::expected<void, holoscan::Error> QualityTextOp::compute(holoscan::ExecutionContext&) {
  auto received = receive(metrics, flags);
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  const auto lines = text_lines(received->frame, schedule_.segment(received->frame),
                                received->flags, received->metrics);
  const std::array<std::int64_t, 2> shape{kTextRows, kTextRowBytes};
  auto loan = text.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                 -> holoscan::expected<void, holoscan::Error> {
    auto* out = writer.data_as<std::uint8_t>();
    std::memset(out, 0, static_cast<std::size_t>(kTextRows * kTextRowBytes));
    for (std::int64_t row = 0; row < kTextRows; ++row) {
      // White, or red for the flags line of a failing frame.
      const bool alert = row == 1 && received->flags != 0U;
      const holoscan::visualization::TextRunHeader header{
          .x = 0.01F,
          .y = 0.01F + 0.045F * static_cast<float>(row),
          .size_px = 20.0F,
          .color = alert ? 0xFF4040FFU : 0xFFFFFFFFU};
      std::uint8_t* record = out + row * kTextRowBytes;
      std::memcpy(record, &header, sizeof(header));
      const std::string& line = lines[static_cast<std::size_t>(row)];
      std::memcpy(record + holoscan::visualization::kTextRunHeaderBytes, line.data(),
                  std::min<std::size_t>(line.size(), kTextColumns));
    }
    return {};
  });
  if (!written) {
    return holoscan::make_unexpected(std::move(written).error());
  }
  return text.emit(std::move(*loan), received->provenance);
}

}  // namespace image_quality_app
