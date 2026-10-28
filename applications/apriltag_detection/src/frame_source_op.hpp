// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The application's image source: synthetic scenes on a clock.

#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/memory_kind.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace apriltag_app {

// Declares `image` as output `name`: width x height x channels uint8 frames in `memory`.
void image_output(holoscan::OperatorSpec& spec, holoscan::Output<holoscan::schema::ImageT>& image,
                  const char* name, holoscan::MemoryKind memory, int width, int height,
                  int channels);

struct FrameSourceConfig {
  // RGB8 renderings of holoscan::apriltag::make_scene(family, frame, width, height), each with
  // header.device_sequence frame + 1 (zero would mean unnumbered).
  std::string family{"tag36h11"};
  int width{1280};
  int height{720};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Emit the frames in CUDA device memory instead of host memory.
  bool device_output{false};
};

// A fixed number of synthetic frames on a clock, then idle.
class FrameSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for an unusable config.
  explicit FrameSourceOp(FrameSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  FrameSourceConfig config_;
  std::uint32_t emitted_{};
};

}  // namespace apriltag_app
