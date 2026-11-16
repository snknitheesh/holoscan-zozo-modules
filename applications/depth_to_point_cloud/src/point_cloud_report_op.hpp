// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// PointCloudReportOp: the application's sink. Counts valid points and their
// depth range, checks that frame ids run without a gap from the first cloud
// to the last frame (frames before the first camera info give no cloud),
// checks every point (and color) against the synthetic camera's ground truth
// when the input is synthetic, optionally writes PLY files, and asks the run
// to complete at the last frame's cloud.

#pragma once

#include <cstdint>
#include <filesystem>
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
#include <holoscan/messages/point_cloud.hpp>

namespace depth_to_point_cloud_app {

// What one run saw. Written by the reporter during the run and read by main()
// after the run has stopped.
struct CloudReport {
  // Clouds received, and the frame id of the first: the frames before it came
  // before the first camera info.
  std::uint32_t frames{};
  std::uint64_t first_frame_id{};
  std::uint64_t points{};
  std::uint64_t valid_points{};
  float min_depth{};
  float max_depth{};
  std::uint32_t ply_files{};
  // Clouds whose frame id is not the one after the previous cloud's.
  std::uint32_t frame_gaps{};
  // Ground-truth checks; zero when the input is not synthetic.
  bool verified{};
  std::uint64_t wrong_points{};  // Valid where invalid, invalid where valid, or misplaced.
  std::uint64_t wrong_colors{};
  double max_error_m{};  // Largest distance of a valid point from the truth.

  [[nodiscard]] bool passed() const noexcept {
    return frames != 0 && frame_gaps == 0 &&
           (!verified || (wrong_points == 0 && wrong_colors == 0));
  }
  // The frames before the first cloud, which came before the first camera
  // info; 0 without a cloud.
  [[nodiscard]] std::uint64_t before_camera_info() const noexcept {
    return frames == 0 ? 0U : first_frame_id - 1U;
  }
  // Whether the run accounts for all `expected` frames: it passed, at most
  // `max_before` frames came before the first camera info, and every frame
  // from it on gave a cloud.
  [[nodiscard]] bool complete(std::uint32_t expected, std::uint64_t max_before) const noexcept {
    return passed() && before_camera_info() <= max_before &&
           before_camera_info() + frames == expected;
  }
};

// The synthetic camera and operator settings the clouds should match.
struct CloudTruth {
  int width{};
  int height{};
  bool millimeters{};
  // The camera info came with every frame, its intrinsics varying by frame.
  bool vary_intrinsics{};
  float depth_scale{};
  float depth_min{};
  float depth_max{};
};

struct PointCloudReportConfig {
  std::shared_ptr<CloudReport> report;
  // The frame id of the last frame, whose cloud completes the run; 0 never
  // completes.
  std::uint32_t expected_frames{};
  // The clouds carry colors.
  bool colors{false};
  // The camera's frame, which every cloud's header names.
  std::string frame_id;
  std::optional<CloudTruth> truth;
  // Where to write each cloud as frame_NNNNNN.ply, NNNNNN its frame from 0
  // (frame id 1); none when empty.
  std::filesystem::path ply_dir;
};

// Largest distance, in meters, a verified point may be from the truth.
inline constexpr double kMaxPointError = 1e-4;

class PointCloudReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `report` is missing.
  explicit PointCloudReportOp(PointCloudReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::messages::schema::PointCloudT> point_cloud;

 private:
  // Whether the message is the organized cloud of the camera's frame that the
  // operator sends; a cloud that is not fails the run.
  [[nodiscard]] bool well_formed(const holoscan::messages::schema::PointCloudT& cloud) const;
  void verify(std::uint32_t frame, const std::vector<float>& xyz,
              const std::vector<std::uint8_t>* rgb, int width, int height);

  PointCloudReportConfig config_;
  std::vector<float> xyz_;
  std::vector<std::uint8_t> rgb_;
  std::uint64_t last_frame_id_{};
};

}  // namespace depth_to_point_cloud_app
