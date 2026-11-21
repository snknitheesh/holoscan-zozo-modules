// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The interface the denoiser's backends implement, their factories, and checks
// both share.

#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <string_view>

#include <holoscan/denoise/denoiser.hpp>

namespace holoscan::denoise {

class Denoiser::Impl {
 public:
  virtual ~Impl() = default;
  [[nodiscard]] virtual Backend backend() const noexcept = 0;
  [[nodiscard]] virtual std::string_view engine() const noexcept = 0;
  // Called with views already checked against each other.
  virtual void process(const ImageView& input, const ImageView& output, cudaStream_t stream) = 0;
  virtual void reset() noexcept = 0;
  [[nodiscard]] virtual float noise_sigma(cudaStream_t stream) = 0;
};

namespace detail {

[[nodiscard]] std::unique_ptr<Denoiser::Impl> make_cuda_denoiser(const DenoiseParams& params);
// Throws std::invalid_argument in a build without VPI.
[[nodiscard]] std::unique_ptr<Denoiser::Impl> make_vpi_denoiser(const DenoiseParams& params,
                                                                const VpiOptions& options);
// Whether VPI can run TNR on the VIC here; false in a build without VPI.
[[nodiscard]] bool vpi_vic_tnr_available() noexcept;

// Throws std::runtime_error naming `what` for a CUDA error.
void check_cuda(cudaError_t status, const char* what);

}  // namespace detail
}  // namespace holoscan::denoise
