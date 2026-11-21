// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// SyntheticCameraOp: a camera that sends the module's test pattern (moving
// colour bars over gradients) as NV12 or YUYV in CUDA device memory, BT.709
// limited range, as many cameras do. Each frame is drawn on the GPU and
// converted to the camera's format with the module's Converter. Its first
// frame goes out on its second tick, after the camera info that a
// CameraInfoSourceOp sends when the run starts.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>

#include <holoscan/color_convert/converter.hpp>
#include <holoscan/color_convert/pixel_format.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace color_convert_app {

struct SyntheticCameraConfig {
  holoscan::color_convert::PixelFormat format{holoscan::color_convert::PixelFormat::kNv12};
  int width{1920};
  int height{1080};
  // Frames to send, then none; 0 sends until the run is stopped.
  std::uint32_t count{0};
  std::chrono::milliseconds period{33};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const SyntheticCameraConfig& config);

// The matrix and range the camera encodes with.
inline constexpr auto kCameraMatrix = holoscan::color_convert::ColorMatrix::kBt709;
inline constexpr auto kCameraRange = holoscan::color_convert::ColorRange::kLimited;

class SyntheticCameraOp final : public holoscan::Operator<> {
 public:
  explicit SyntheticCameraOp(SyntheticCameraConfig config);
  ~SyntheticCameraOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // NV12 (a packed byte buffer with its plane layouts) or YUYV [H, W, 2], on
  // the GPU. The sample's frame id counts from 1 and its capture time is the
  // activation's. A frame the graph refuses (a full lossless queue, or every
  // slot of the output's pool still held) is sent again on the next tick.
  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  SyntheticCameraConfig config_;
  std::unique_ptr<holoscan::color_convert::Converter> converter_;
  void* rgb_{};
  bool started_{};
  std::uint32_t sent_{};
};

}  // namespace color_convert_app
