// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// DenoiseOp: a noisy image in, the denoised image out, on the GPU, through a
// Denoiser (denoiser.hpp) on the configured backend. The temporal filter keeps
// its history between frames: a change of image size or format, or a restart
// of the operator, starts a new sequence, as does a scene cut.

#pragma once

#include <atomic>
#include <cstdint>
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

// What a running DenoiseOp has done, for its owner to read at any time (the
// operator in a running graph is the SDK's copy, not the one graph.op()
// returned, so read this rather than that operator).
struct DenoiseStats {
  // The backend the operator runs: kAuto until it starts, then kCuda or kVpi.
  std::atomic<Backend> backend{Backend::kAuto};
  // What ran the latest frame, as Denoiser::engine(): "cuda", "vpi-vic" or
  // "vpi-cuda"; empty before the first frame.
  std::atomic<const char*> engine{""};
  std::atomic<std::uint64_t> frames{};
};

struct DenoiseOpConfig {
  // kAuto chooses when the operator starts, not when it is built: probing VPI
  // starts threads, and a plain holoscan::run(plan) admits only a process with
  // one. See resolve_auto().
  Backend backend{Backend::kCuda};
  DenoiseParams params;
  VpiOptions vpi;
  // Largest image the output is sized for, 1..16384.
  int max_width{3840};
  int max_height{2160};
  // Optional: filled in as the operator runs.
  std::shared_ptr<DenoiseStats> stats;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const DenoiseOpConfig& config);

class DenoiseOp final : public holoscan::Operator<> {
 public:
  // Probes nothing: Backend::kAuto is resolved, and its choice logged, at start.
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

  // The backend this operator runs: the configured one, or for kAuto the one
  // chosen at start, and kAuto until then. Safe from any thread.
  [[nodiscard]] Backend backend() const noexcept { return backend_.load(); }

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
  std::atomic<Backend> backend_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::denoise
