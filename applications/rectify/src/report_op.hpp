// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's reporter: takes each raw pair with its rectified pair and
// geometry, checks them, and optionally draws them for HoloViz: the raw pair
// above the rectified one, with horizontal epipolar lines across both.

#pragma once

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
#include <holoscan/rectify/rectification.hpp>
#include <holoscan/rectify/stereo_rectify_op.hpp>
#include <holoscan/rectify/synthetic_stereo.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace rectify_app {

// What one rectifier did over the run. Written by its ReportOp, read after the run.
struct Report {
  // Pairs reported; the source reads it while the run goes on.
  std::atomic<std::uint32_t> frames{};
  // Synthetic: the dots measured, those not found, and their errors in pixels.
  int dots{};
  int missing{};
  double max_row_error{};
  double sum_row_error2{};
  double max_position_error{};
  // Recording: the largest change of a pixel between a raw and a rectified image.
  int max_pixel_change{};
  // Every geometry message matched the rectification computed at start.
  bool geometry_ok{true};
  // The backend that remapped the pairs, from the geometry messages.
  std::string backend;

  [[nodiscard]] double rms_row_error() const;
};

// The tolerances a run must meet: rows and positions in pixels, for the
// synthetic pair; and for a recording, which is rectified already, at most
// one gray level of change.
inline constexpr double kMaxRowError = 0.1;
inline constexpr double kMaxPositionError = 0.15;
inline constexpr int kMaxPixelChange = 1;

struct ReportConfig {
  std::shared_ptr<Report> report;
  // Counts down the reporters still running; the last one to see
  // `expected_frames` frames completes the run.
  std::shared_ptr<std::atomic<int>> remaining;
  std::uint32_t expected_frames{};
  // Synthetic: the scene, to check each dot against. Without it, the pair is
  // a recording, which must come out unchanged.
  std::optional<holoscan::rectify::SyntheticStereo> scene;
  holoscan::rectify::StereoRectification rectification;
  // Declare the `view` output.
  bool view_output{};
};

class ReportOp final : public holoscan::Operator<> {
 public:
  explicit ReportOp(ReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The raw pair, the rectified pair and the geometry, of one capture time.
  holoscan::Input<holoscan::schema::ImageT> raw_left;
  holoscan::Input<holoscan::schema::ImageT> raw_right;
  holoscan::Input<holoscan::schema::ImageT> rectified_left;
  holoscan::Input<holoscan::schema::ImageT> rectified_right;
  holoscan::Input<holoscan::rectify::schema::StereoGeometryT> geometry;
  // With `view_output`: RGBA8 in host memory, the raw pair above the
  // rectified pair, with green epipolar lines every 40 rows and the valid
  // regions of the rectified images outlined in yellow.
  holoscan::Output<holoscan::schema::ImageT> view;

 private:
  [[nodiscard]] holoscan::expected<void, holoscan::Error> send_view(
      const std::vector<std::uint8_t>& raw_l, const std::vector<std::uint8_t>& raw_r,
      const std::vector<std::uint8_t>& rect_l, const std::vector<std::uint8_t>& rect_r,
      const holoscan::EmitOptions& emit);

  ReportConfig config_;
};

}  // namespace rectify_app
