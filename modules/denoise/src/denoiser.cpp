// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/denoise/denoiser.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include "denoiser_impl.hpp"

namespace holoscan::denoise {
namespace detail {

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

#if !HOLOSCAN_DENOISE_WITH_VPI
bool vpi_vic_tnr_available() noexcept { return false; }

std::unique_ptr<Denoiser::Impl> make_vpi_denoiser(const DenoiseParams&, const VpiOptions&) {
  throw std::invalid_argument("backend vpi: this build has no VPI; install VPI 4 and rebuild");
}
#endif

}  // namespace detail

namespace {

void check_view(const ImageView& view, const char* what) {
  if (view.width < 1 || view.height < 1) {
    throw std::invalid_argument(std::string(what) + " has no pixels");
  }
  const auto row = static_cast<std::size_t>(view.width) * pixel_bytes(view.format);
  if (view.planes[0].data == nullptr || view.planes[0].pitch < row) {
    throw std::invalid_argument(std::string(what) + ": no pixels, or rows shorter than a row");
  }
  if (view.format == PixelFormat::kNv12) {
    if (view.width % 2 != 0 || view.height % 2 != 0) {
      throw std::invalid_argument(std::string(what) + ": NV12 needs an even width and height");
    }
    if (view.planes[1].data == nullptr || view.planes[1].pitch < row) {
      throw std::invalid_argument(std::string(what) + ": NV12 needs its UV plane");
    }
  }
}

}  // namespace

void check_available(Backend backend) {
  if (backend == Backend::kVpi && !HOLOSCAN_DENOISE_WITH_VPI) {
    throw std::invalid_argument("backend vpi: this build has no VPI; install VPI 4 and rebuild");
  }
}

PlatformFacts probe_platform() noexcept {
  return PlatformFacts{.vpi_compiled = HOLOSCAN_DENOISE_WITH_VPI != 0,
                       .vic_tnr = detail::vpi_vic_tnr_available()};
}

Denoiser::Denoiser(Backend backend, const DenoiseParams& params, const VpiOptions& vpi) {
  validate_params(params);
  check_available(backend);
  const Backend resolved =
      backend == Backend::kAuto ? resolve_auto(probe_platform(), params) : backend;
  if (const std::string_view reason = unsupported_reason(resolved, params); !reason.empty()) {
    throw std::invalid_argument(std::string(reason));
  }
  impl_ = resolved == Backend::kVpi ? detail::make_vpi_denoiser(params, vpi)
                                    : detail::make_cuda_denoiser(params);
  params_ = params;
}

Denoiser::~Denoiser() = default;

Backend Denoiser::backend() const noexcept { return impl_->backend(); }
std::string_view Denoiser::engine() const noexcept { return impl_->engine(); }
const DenoiseParams& Denoiser::params() const noexcept { return params_; }

void Denoiser::process(const ImageView& input, const ImageView& output, cudaStream_t stream) {
  check_view(input, "the input");
  check_view(output, "the output");
  if (input.format != output.format || input.width != output.width ||
      input.height != output.height) {
    throw std::invalid_argument("the output must have the input's format and size");
  }
  impl_->process(input, output, stream);
}

void Denoiser::reset() noexcept { impl_->reset(); }

float Denoiser::noise_sigma(cudaStream_t stream) { return impl_->noise_sigma(stream); }

}  // namespace holoscan::denoise
