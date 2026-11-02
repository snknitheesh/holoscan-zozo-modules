// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Denoiser: the denoising pipeline without a Holoscan graph, on device images
// and a CUDA stream. DenoiseOp runs one per operator.

#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string_view>

#include <holoscan/denoise/denoise_params.hpp>

namespace holoscan::denoise {

struct VpiOptions {
  VpiEngine engine{VpiEngine::kAuto};
  VpiPreset preset{VpiPreset::kIndoorLowLight};
};

// Checks that `backend` is compiled into this build. kAuto always is.
// @throws std::invalid_argument naming what is missing.
void check_available(Backend backend);
// The facts about this process's platform that resolve_auto() needs; a fact
// that cannot be established is false.
[[nodiscard]] PlatformFacts probe_platform() noexcept;

class Denoiser {
 public:
  // Resolves kAuto (probe_platform(), resolve_auto()). Device memory is
  // allocated on the first frame, and again when the size or format changes.
  // @throws std::invalid_argument if the settings are unusable or the backend
  //         is not compiled in or cannot run them.
  // @throws std::runtime_error if VPI cannot be set up.
  Denoiser(Backend backend, const DenoiseParams& params, const VpiOptions& vpi = {});
  ~Denoiser();
  Denoiser(const Denoiser&) = delete;
  Denoiser& operator=(const Denoiser&) = delete;

  // kCuda or kVpi.
  [[nodiscard]] Backend backend() const noexcept;
  // What runs it: "cuda", "vpi-vic" or "vpi-cuda".
  [[nodiscard]] std::string_view engine() const noexcept;
  [[nodiscard]] const DenoiseParams& params() const noexcept;

  // Denoises `input` into `output`, both in CUDA device memory on the current
  // device, of the same format and size (NV12: even sizes). Work is queued on
  // `stream`; the vpi backend also waits for it on the host. A change of size or
  // format, like the first frame, starts a new sequence.
  // @throws std::invalid_argument for mismatched or unusable views.
  // @throws std::runtime_error on a CUDA or VPI failure.
  void process(const ImageView& input, const ImageView& output, cudaStream_t stream);

  // The next frame starts a new sequence, without history.
  void reset() noexcept;

  // The noise standard deviation (code values) the last frame was filtered for:
  // the configured one, or the estimate. Waits for `stream`. 0 before a frame.
  [[nodiscard]] float noise_sigma(cudaStream_t stream);

  class Impl;

 private:
  std::unique_ptr<Impl> impl_;
  DenoiseParams params_;
};

}  // namespace holoscan::denoise
