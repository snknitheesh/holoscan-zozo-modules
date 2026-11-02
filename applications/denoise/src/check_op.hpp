// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CheckOp: the demo's ground truth. Takes each noisy frame with its denoised
// and fused images, and measures:
// - the PSNR of the noisy and the denoised frame against the clean scene;
// - the trail: the error, behind the moving disc (where it was in the last 4
//   frames and no longer is), of the denoised frame against that of the noisy;
// - the fused image's shadow and window: samples clipped or crushed, and the
//   texture (standard deviation) each keeps.
// The last frame ends the run.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/denoise/synthetic_scene.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace denoise_demo {

struct Report {
  std::uint32_t frames{};
  std::uint32_t frame_gaps{};  // Frames that arrived out of order or not at all.
  // Over the frames past the first few of each sequence:
  std::uint32_t checked{};
  double noisy_psnr_sum{};
  double denoised_psnr_sum{};
  double worst_gain{99.0};
  // Squared errors and samples behind the disc.
  double trail_squared{};
  double trail_input_squared{};
  std::uint64_t trail_samples{};
  // The fused images.
  std::uint64_t fused_clipped{};  // Window samples at 254 or more.
  std::uint64_t fused_crushed{};  // Shadow samples at 1 or less.
  double worst_shadow_texture{999.0};
  double worst_window_texture{999.0};
};

struct CheckConfig {
  holoscan::denoise::LowLightScene scene;
  holoscan::denoise::HdrScene hdr;
  std::uint32_t expected_frames{};
  // Frames of a sequence (from the start, or a scene cut) before PSNR is judged.
  std::uint32_t settle_frames{4};
  std::shared_ptr<Report> report;
  // Counted up after each frame, for the source.
  std::shared_ptr<std::atomic<std::uint32_t>> finished;
};

class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(CheckConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // RGB8 [H, W, 3], all of one capture time.
  holoscan::Input<holoscan::schema::ImageT> noisy;
  holoscan::Input<holoscan::schema::ImageT> denoised;
  holoscan::Input<holoscan::schema::ImageT> fused;

 private:
  CheckConfig config_;
  std::vector<std::uint8_t> noisy_;
  std::vector<std::uint8_t> denoised_;
  std::vector<std::uint8_t> fused_;
};

}  // namespace denoise_demo
