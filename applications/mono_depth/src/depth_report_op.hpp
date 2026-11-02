// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// DepthReportOp: the application's sink. It takes each image's depth from
// MonoDepthOp, checks that frame ids run 1, 2, 3, ... without a gap (in real
// time, that they go up), measures the time from capture to depth on the
// host, and then, as configured:
//
// - with the stub model: checks every value against the host reference of
//   the whole pipeline, computed from the very image the estimator saw,
//   which it takes too;
// - with a real model on synthetic frames: checks that the depth ranks the
//   scene as its true depth does (Spearman's correlation, and the regions'
//   order);
// - with image files: prints each file's depth range the first time it is
//   seen.
//
// It takes the metric depth too when the estimator gives it, checked against
// the reference as well, and optionally the colorized depth, to write it as a
// PPM file, and asks the run to complete after its expected frames.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/mono_depth/mono_depth_op.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace mono_depth_app {

// What one run saw. Written by the reporter during the run and read by main()
// after it has stopped; only `frames` may be read while it runs.
struct DepthReport {
  std::atomic<std::uint32_t> frames{};
  // Frames whose id is not the one after the previous frame's; in real time,
  // only those whose id is not above it.
  std::uint32_t frame_gaps{};
  // In real time: the frames passed over for newer ones, and the last id seen.
  std::uint64_t skipped{};
  std::uint64_t last_frame_id{};
  // The stub's reference: frames checked, frames off by more than the
  // tolerance, and the largest error, relative to max(1, |value|).
  std::uint32_t reference_checked{};
  std::uint32_t reference_failed{};
  float max_reference_error{};
  // The synthetic scene's truth: frames checked, frames below the minimum
  // correlation or with their regions out of order, and the correlations.
  std::uint32_t ordering_checked{};
  std::uint32_t ordering_failed{};
  double min_spearman{1.0};
  double sum_spearman{};
  // The inverse depth's range over the run, and the metric depth's.
  float min_depth{std::numeric_limits<float>::infinity()};
  float max_depth{-std::numeric_limits<float>::infinity()};
  float min_meters{std::numeric_limits<float>::infinity()};
  float max_meters{-std::numeric_limits<float>::infinity()};
  std::uint32_t files_written{};
  // Capture to depth on the host, per frame, in milliseconds.
  std::vector<double> latencies_ms;

  [[nodiscard]] bool passed() const noexcept {
    return frame_gaps == 0 && reference_failed == 0 && ordering_failed == 0;
  }
};

// How the reporter checks the depth.
enum class Verify : std::uint8_t { kNone, kReference, kOrdering };

struct DepthReportConfig {
  // Where the reporter adds what it sees; one report per run.
  std::shared_ptr<DepthReport> report;
  // Frames after which the run completes; 0 never completes. In real time,
  // the run completes at the frame with this id.
  std::uint32_t expected_frames{};
  // The inputs keep only the newest sample (see edges.hpp): frames come with
  // ids that go up, but not one by one, as those that came while the reporter
  // was busy are passed over and counted as skipped.
  bool realtime{false};
  // Time spent on each frame on top of the work: a stand-in, in tests, for a
  // consumer slower than the source.
  std::chrono::nanoseconds delay{};
  Verify verify{Verify::kNone};
  // The estimator's settings, which the reference follows; its metric depth
  // is taken when it gives one.
  holoscan::mono_depth::MonoDepthConfig estimator;
  // kReference: the largest error accepted, relative to max(1, |value|).
  float tolerance{2e-5F};
  // kOrdering: the synthetic scene's size, whether it is still, and the
  // smallest correlation accepted.
  int scene_width{};
  int scene_height{};
  bool still{false};
  double min_spearman{0.85};
  // The images of a file source, frame f being files[(f - 1) % size]: named
  // when printing.
  std::vector<std::filesystem::path> files;
  // Print each file's depth range when it is first seen.
  bool print{false};
  // Declare the `colorized` input and write each frame's colorized depth as
  // `image_dir`/depth_NNNNNN.ppm, numbered by frame id.
  std::filesystem::path image_dir;
};

class DepthReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if the report is missing or the settings
  // unusable.
  explicit DepthReportOp(DepthReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MonoDepthOp's depth.
  holoscan::Input<holoscan::schema::ImageT> depth;
  // MonoDepthOp's metric depth; declared when it gives one.
  holoscan::Input<holoscan::schema::ImageT> metric_depth;
  // The image the estimator saw; declared for kReference only.
  holoscan::Input<holoscan::schema::ImageT> image;
  // MonoDepthOp's colorized depth; declared with `image_dir` only.
  holoscan::Input<holoscan::schema::ImageT> colorized;

 private:
  [[nodiscard]] holoscan::expected<void, holoscan::Error> process(
      holoscan::ExecutionContext& context);

  DepthReportConfig config_;
  std::uint64_t last_frame_id_{};
  std::uint32_t seen_{};
  std::vector<float> depth_;
  std::vector<float> meters_;
  std::vector<std::uint8_t> pixels_;
};

}  // namespace mono_depth_app
