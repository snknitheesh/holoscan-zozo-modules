// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// DisparityReportOp: the demo's sink. Counts frames, valid pixels and
// latency; checks synthetic frames against the scene's truth, and a
// recording's computed depth against the camera's own; and requests
// completion once it has its frames.

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
#include <holoscan/foundation_stereo/synthetic_scene.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace foundation_stereo_app {

// Relative depth errors are counted in bins this wide; the last bin holds
// everything from 100 % up.
inline constexpr double kErrorBin = 0.001;
inline constexpr std::size_t kErrorBins = 1001;
inline constexpr double kTruthBin = 0.01;
inline constexpr std::size_t kTruthBins = 1001;

// A run's results. Written by the reporter during the run and
// read by main() after the run has stopped.
struct DisparityReport {
  std::uint32_t frames{};
  std::uint32_t frame_gaps{};     // Frames missing between two received ones.
  std::vector<float> latency_ms;  // Capture to report, per frame.
  std::uint64_t pixels{};
  std::uint64_t valid_pixels{};
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
  // With the stub model: pixels compared with the CPU reference of the whole
  // chain, those off by more than the tolerance, and the largest difference.
  bool stub_checked{};
  std::uint64_t stub_pixels{};
  std::uint64_t stub_wrong{};
  double stub_worst{};

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
  // With the stub model: check each synthetic frame against the CPU
  // reference of the operator's chain for a stub of this input size,
  // relative tolerance `stub_tolerance`, from RGB8 (`color`) or MONO8 images.
  std::optional<std::array<int, 2>> stub_model_size;
  double stub_tolerance{5e-3};
  bool color{true};
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

 private:
  // Compares the frame's disparity with the stub's reference for `scene`.
  void check_stub(const holoscan::foundation_stereo::SyntheticScene& scene,
                  DisparityReport& report);

  DisparityReportConfig config_;
  bool finished_{};
  std::uint64_t last_frame_id_{};
  std::vector<float> disparity_;
  std::vector<float> depth_;
  std::vector<std::uint16_t> camera_depth_;
};

}  // namespace foundation_stereo_app
