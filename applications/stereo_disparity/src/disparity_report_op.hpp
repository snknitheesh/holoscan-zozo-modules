// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// DisparityReportOp: the demo's sink, one per backend. Counts frames, valid
// pixels and latency; checks synthetic frames against the scene's truth, and a
// recording's computed depth against the camera's own; and, with the other
// reporters, requests completion once every one has its frames.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace stereo_disparity_app {

// Relative depth errors are counted in bins this wide; the last bin holds
// everything from 100 % up.
inline constexpr double kErrorBin = 0.001;
inline constexpr std::size_t kErrorBins = 1001;
inline constexpr double kTruthBin = 0.01;
inline constexpr std::size_t kTruthBins = 1001;

// A run's results for one backend. Written by its reporter during the run and
// read by main() after the run has stopped.
struct DisparityReport {
  std::uint32_t frames{};
  std::uint32_t frame_gaps{};     // Frames missing between two received ones.
  std::vector<float> latency_ms;  // Capture to report, per frame.
  std::uint64_t pixels{};
  std::uint64_t valid_pixels{};
  std::uint64_t whole_pixels{};  // Valid disparities that are whole pixels.
  // Synthetic truth: pixels both cameras see, those with a disparity, those
  // within 1 px of the truth, and their absolute error.
  bool verified{};
  std::uint64_t truth_pixels{};
  std::uint64_t matched_pixels{};
  std::uint64_t within_1px{};
  double abs_error_sum{};
  // Their absolute errors in bins of kTruthBin pixels, the last for
  // everything from kTruthBins bins up.
  std::array<std::uint64_t, kTruthBins> truth_bins{};
  // The camera's own depth: pixels where it has one, those where the computed
  // depth has one too, and the relative error |computed - camera| / camera of
  // those, binned.
  bool compared{};
  std::uint64_t camera_pixels{};
  std::uint64_t both_pixels{};
  std::array<std::uint64_t, kErrorBins> error_bins{};
  // VPI's confidence (0..65535) summed over the pixels with a disparity.
  std::uint64_t confidence_pixels{};
  double confidence_sum{};

  // The `fraction` quantile of the relative errors, to a bin's width.
  [[nodiscard]] double error_quantile(double fraction) const;
  // The share of compared pixels within `relative` of the camera's depth.
  [[nodiscard]] double share_within(double relative) const;
  // The median absolute error against the truth, to a bin's width.
  [[nodiscard]] double truth_median() const;
};

struct DisparityReportConfig {
  std::shared_ptr<DisparityReport> report;
  // Frames after which this reporter is done; when every reporter sharing
  // `remaining` is, the last one requests completion. 0 never finishes.
  std::uint32_t expected_frames{};
  std::shared_ptr<std::atomic<int>> remaining;
  // Check each frame against make_scene(frame_id - 1, width, height).
  bool truth{false};
  int width{};
  int height{};
  // Take the computed `depth` and the camera's `camera_depth` (MONO16, in
  // units of this scale, 0 where it has none) and compare them.
  std::optional<double> camera_depth_scale;
  // Take the `confidence` map (MONO16) too.
  bool confidence{false};
};

class DisparityReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument without a report or `remaining`.
  explicit DisparityReportOp(DisparityReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::ImageT> disparity;
  holoscan::Input<holoscan::schema::ImageT> depth;
  holoscan::Input<holoscan::schema::ImageT> camera_depth;
  holoscan::Input<holoscan::schema::ImageT> confidence;

 private:
  DisparityReportConfig config_;
  bool finished_{};
  std::uint64_t last_frame_id_{};
  std::vector<float> disparity_;
  std::vector<float> depth_;
  std::vector<std::uint16_t> camera_depth_;
  std::vector<std::uint16_t> confidence_;
};

}  // namespace stereo_disparity_app
