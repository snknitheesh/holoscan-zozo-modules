// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// DenoiseOp: a noisy image in, the denoised image out, on the GPU, through a
// Denoiser (denoiser.hpp) on the configured backend. The temporal filter keeps
// its history between frames: a change of image size or format, or a restart
// of the operator, starts a new sequence, as does a scene cut.

#pragma once

#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/denoise/denoise_params.hpp>
#include <holoscan/denoise/denoiser.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::denoise {

struct DenoiseOpConfig {
  // kAuto chooses when the operator is built: see resolve_auto().
  Backend backend{Backend::kCuda};
  DenoiseParams params;
  VpiOptions vpi;
  // Largest image the output is sized for, 1..16384.
  int max_width{3840};
  int max_height{2160};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const DenoiseOpConfig& config);

class DenoiseOp final : public holoscan::Operator<> {
 public:
  // Resolves Backend::kAuto, and logs its choice at start.
  // @throws std::invalid_argument if `config` is not valid.
  explicit DenoiseOp(DenoiseOpConfig config);
  ~DenoiseOp() override;
  DenoiseOp(const DenoiseOp&) = delete;
  DenoiseOp& operator=(const DenoiseOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // kCuda or kVpi.
  [[nodiscard]] Backend backend() const noexcept { return backend_; }

  // MONO8, RGB8 or RGBA8, uint8 [H, W] or [H, W, channels], packed pixels, any
  // row pitch; or NV12 by its plane layouts, even sizes. Host or CUDA device
  // memory (host images are copied to the GPU, which is logged once).
  holoscan::Input<holoscan::schema::ImageT> image;
  // The input's format and size, uint8 [H, W, channels] (NV12: [H * 3 / 2, W, 1]
  // with plane layouts), packed, in CUDA device memory. The descriptor keeps the
  // input's header, colour space, exposure and gain; the sample its capture
  // time and frame id.
  holoscan::Output<holoscan::schema::ImageT> denoised;

 private:
  struct State;

  DenoiseOpConfig config_;
  Backend backend_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::denoise
