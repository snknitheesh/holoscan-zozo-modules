// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// RawSourceOp: frames of holoscan-isp's synthetic raw camera as a GMSL or CSI
// sensor delivers them: a Bayer mosaic in device memory, 8 to 16 bits per
// sample or packed MIPI RAW10 / RAW12, with the exposure time and gain of the
// capture; a fixed number on a clock, then idle.
//
// The ISP cannot send a refused output again, so a full queue must never meet
// it: with `finished`, the source keeps one frame in flight, sending the next
// only once the last consumer has counted the previous one.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/isp/synthetic_raw.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace isp_demo {

// The exposure every frame reports.
inline constexpr std::int64_t kExposureNs = 10'000'000;
inline constexpr float kGain = 1.0F;

struct RawSourceConfig {
  // The camera; `frame` is set per frame.
  holoscan::isp::SyntheticRawConfig camera;
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Frames the last consumer has finished; none sends on every tick.
  std::shared_ptr<const std::atomic<std::uint32_t>> finished;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const RawSourceConfig& config);

class RawSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit RawSourceOp(RawSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // BAYER_* uint16 or uint8 [H, W], or with packing uint8 [H, row bytes]; frame
  // ids count from 1, frame f + 1 is synthetic frame f.
  holoscan::Output<holoscan::schema::ImageT> raw;

 private:
  RawSourceConfig config_;
  std::uint32_t emitted_{};
};

}  // namespace isp_demo
