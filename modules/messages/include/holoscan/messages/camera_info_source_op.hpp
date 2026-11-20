// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CameraInfoSourceOp: the first operator of a chain that needs calibration.
// It reads one camera_info YAML file (a single camera) or two (a stereo
// pair's left and right) when it is constructed, and sends the calibration
// periodically, as CameraInfo on `camera_info` or as StereoCameraInfo on
// `stereo_camera_info`, stamped with the time it is sent. Consumers take it
// with camera_info_contract() and LatestCameraInfo (camera_info_input.hpp).

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info_schema_traits.hpp>
#include <holoscan/messages/stereo_camera_info_schema_traits.hpp>

namespace holoscan::messages {

struct CameraInfoSourceConfig {
  // Required: a single camera's camera_info file, or a stereo pair's left one.
  std::filesystem::path camera_info_path;
  // A stereo pair's right camera_info file; empty for a single camera.
  std::filesystem::path right_camera_info_path;
  // How often the calibration is sent, > 0. The first one goes out when the
  // run starts, so a consumer's frames wait at most this long for it at the
  // start; keep it below the consumer's max_age (10 s by default). Each one is
  // stamped with its send time, and a frame takes only one captured at or
  // before it: keep the period above the time frames take from their capture
  // to the consumer, or connect it with latest_camera_info_connection(depth)
  // (camera_info_input.hpp).
  std::chrono::nanoseconds period{std::chrono::seconds{1}};
  // How many to send, then stay idle; 0 sends until the run ends.
  std::uint64_t count{0};
};

class CameraInfoSourceOp final : public holoscan::Operator<> {
 public:
  // Reads and validates the file(s). The SDK constructs an operator again
  // when a run starts, so the files are read again then and must still be
  // there.
  // @throws std::invalid_argument naming the file and what is wrong, or for
  // a period that is not > 0.
  explicit CameraInfoSourceOp(CameraInfoSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // A single camera's calibration; declared without right_camera_info_path.
  holoscan::Output<schema::CameraInfoT> camera_info;
  // A stereo pair's; declared with right_camera_info_path.
  holoscan::Output<schema::StereoCameraInfoT> stereo_camera_info;

 private:
  CameraInfoSourceConfig config_;
  // The message read at construction; each send stamps its headers.
  std::optional<schema::CameraInfoT> mono_;
  std::optional<schema::StereoCameraInfoT> stereo_;
  std::uint64_t sent_{};
};

}  // namespace holoscan::messages
