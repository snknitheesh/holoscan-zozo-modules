// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's operators: stereo frames on a clock, a consumer that takes them
// with the camera info and checks every calibration it gets, and a synthetic
// point cloud source and its checker.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/memory_kind.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/schemas/header_schema_traits.hpp>

namespace messages_demo {

namespace hm = holoscan::messages;
// A frame of the stereo pair: its header stands in for the images, which
// the demo does not need.
using Frame = holoscan::schema::HeaderT;

// What the checkers found; read by main() while the run goes on.
struct Report {
  std::atomic<std::uint32_t> frames{};
  std::atomic<std::uint32_t> frames_with_camera_info{};
  std::atomic<std::uint32_t> wrong_camera_info{};
  std::atomic<std::uint32_t> clouds{};
  std::atomic<std::uint32_t> wrong_clouds{};
  // The largest difference from the true extrinsics, rotation and translation (m).
  std::atomic<double> max_extrinsics_error{};
  // Checkers still running; the last one to finish ends the run.
  std::atomic<int> running{2};
};

// `count` stereo frames, one every `period`, with capture time = activation time.
class FrameSourceOp final : public holoscan::Operator<> {
 public:
  FrameSourceOp(std::uint32_t count, std::chrono::nanoseconds period)
      : count_(count), period_(period) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<Frame> left;
  holoscan::Output<Frame> right;

 private:
  std::uint32_t count_;
  std::chrono::nanoseconds period_;
  std::uint32_t sent_{};
  // The pair being sent, and whether its left frame went out before the
  // right one met a busy checker.
  std::optional<holoscan::EmitOptions> pending_;
  bool left_sent_{};
};

struct CalibrationCheckConfig {
  // The calibration the files were written from.
  hm::StereoCameraInfo truth;
  std::uint32_t frames{};
  std::shared_ptr<Report> report;
};

// Takes the stereo frames and the camera info with the recommended contract,
// and checks each new calibration against the truth: both cameras field by
// field, the extrinsics recovered from R and P to 1e-9.
class CalibrationCheckOp final : public holoscan::Operator<> {
 public:
  explicit CalibrationCheckOp(CalibrationCheckConfig config) : config_(std::move(config)) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<Frame> left;
  holoscan::Input<Frame> right;
  holoscan::Input<hm::schema::StereoCameraInfoT> camera_info;

 private:
  CalibrationCheckConfig config_;
  hm::LatestCameraInfo<hm::schema::StereoCameraInfoT> info_;
  bool calibration_ok_{};
};

// The demo's clouds: organized, with colors and normals, without intensity.
inline constexpr std::int64_t kCloudHeight = 48;
inline constexpr std::int64_t kCloudWidth = 64;

struct CloudConfig {
  std::uint32_t count{};
  std::chrono::nanoseconds period{};
  holoscan::MemoryKind memory{holoscan::MemoryKind::kCudaDevice};
  std::shared_ptr<Report> report;
};

// Synthetic clouds whose every value follows from the frame number.
class CloudSourceOp final : public holoscan::Operator<> {
 public:
  explicit CloudSourceOp(CloudConfig config) : config_(std::move(config)) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<hm::schema::PointCloudT> cloud;

 private:
  CloudConfig config_;
  std::uint32_t sent_{};
};

// Checks each cloud, every value, against the one its frame number gives.
class CloudCheckOp final : public holoscan::Operator<> {
 public:
  explicit CloudCheckOp(CloudConfig config) : config_(std::move(config)) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<hm::schema::PointCloudT> cloud;

 private:
  CloudConfig config_;
};

}  // namespace messages_demo
