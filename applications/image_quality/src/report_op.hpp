// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's checks on ImageQualityOp's outputs: QualityReportOp compares every
// frame's flags with what its segment should give and requests completion after
// the last frame; QualityTextOp writes the metrics as a HoloViz text layer.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/image_quality/quality.hpp>

#include "segments.hpp"

namespace image_quality_app {

// One segment's frames: how many, how many had other flags than expected, the
// union of the flags they had, and the sums of a few metrics.
struct SegmentTally {
  std::uint64_t frames{};
  std::uint64_t wrong{};
  std::uint32_t flags{};
  double focus{};
  double mean_luma{};
  double noise_sigma{};
  double color_cast{};
};

// Written by the reporter during the run and read by main() after it stopped.
struct RunReport {
  std::uint64_t frames{};
  std::array<SegmentTally, kSegmentCount> segments{};
  // The first few frames with other flags than expected.
  std::vector<std::string> mismatches;
};

struct QualityReportConfig {
  std::shared_ptr<RunReport> report;
  Schedule schedule;
  // Frames after which the reporter requests completion.
  std::uint64_t expected_frames{};
};

class QualityReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `report` is missing.
  explicit QualityReportOp(QualityReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // ImageQualityOp's metrics and flags of one frame, with its capture time.
  holoscan::Input<holoscan::Tensor> metrics;
  holoscan::Input<holoscan::Tensor> flags;

 private:
  QualityReportConfig config_;
};

// Lines of text the layer holds, and bytes per line.
inline constexpr std::int64_t kTextRows = 5;
inline constexpr std::int64_t kTextColumns = 64;

class QualityTextOp final : public holoscan::Operator<> {
 public:
  explicit QualityTextOp(Schedule schedule);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> metrics;
  holoscan::Input<holoscan::Tensor> flags;
  // uint8 [kTextRows, 16 + kTextColumns] HoloViz text records, in host memory.
  holoscan::Output<holoscan::Tensor> text;

 private:
  Schedule schedule_;
};

// The text lines for one frame's metrics and flags.
[[nodiscard]] std::array<std::string, kTextRows> text_lines(
    std::uint64_t frame, Segment segment, std::uint32_t flags,
    const std::array<float, holoscan::image_quality::kMetricCount>& metrics);

}  // namespace image_quality_app
