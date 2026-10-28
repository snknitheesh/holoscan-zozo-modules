// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// FrameReportOp: takes each frame's ports from StereoCameraOp together, by
// capture time, and checks them: the eyes have the same size and format as each
// other and as the calibration, the RGB and depth images have theirs, and the
// frame ids, capture times and the camera's timestamps increase. For a
// synthetic recording it also checks every image against the synthetic camera.
// It prints the calibration of the first frame, and asks the run to complete
// after its expected frames.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/stereo_camera/calibration_message.hpp>
#include <holoscan/stereo_camera/image.hpp>

namespace stereo_camera_app {

// What the reporter found, for main() to read once the run is over.
struct FrameReport {
  std::atomic<std::uint32_t> frames{0};
  std::atomic<std::uint64_t> last_frame_id{0};
  // Set from the first frame.
  std::mutex mutex;
  std::optional<holoscan::stereo_camera::StereoCalibration> calibration;
  std::string backend;
  std::string eyes;
  std::string rgb;
  std::string depth;
  // Problems, each counted once per frame.
  std::uint32_t wrong_sizes{};
  std::uint32_t not_increasing{};
  std::uint32_t frame_gaps{};
  std::uint32_t calibration_changes{};
  std::uint32_t wrong_images{};
  // Images checked against the synthetic camera.
  std::uint32_t verified{};
  // The share of depth pixels with a depth, over every frame.
  std::uint64_t depth_pixels{};
  std::uint64_t valid_depth{};

  [[nodiscard]] bool passed() const noexcept {
    return wrong_sizes == 0 && not_increasing == 0 && frame_gaps == 0 && calibration_changes == 0 &&
           wrong_images == 0;
  }
};

struct FrameReportConfig {
  std::shared_ptr<FrameReport> report;
  bool rgb{};
  bool depth{};
  // Frames after which the run completes; 0 never completes. In real time, the
  // run completes at the frame with this id, and skipped frames are no gap.
  std::uint32_t expected_frames{};
  bool realtime{};
  // The images are a synthetic recording's: check them against the camera.
  bool synthetic{};
};

class FrameReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument without a report.
  explicit FrameReportOp(FrameReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::ImageT> left;
  holoscan::Input<holoscan::schema::ImageT> right;
  holoscan::Input<holoscan::schema::ImageT> rgb;
  holoscan::Input<holoscan::schema::ImageT> depth;
  holoscan::Input<holoscan::stereo_camera::schema::StereoCalibrationT> calibration;

 private:
  FrameReportConfig config_;
  std::optional<std::int64_t> last_capture_;
  std::optional<std::int64_t> last_timestamp_;
  std::optional<std::uint64_t> last_index_;
  std::string first_calibration_;
  holoscan::stereo_camera::Image host_[4];
  holoscan::stereo_camera::Image truth_[4];
};

}  // namespace stereo_camera_app
