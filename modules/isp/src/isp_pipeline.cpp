// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/isp/isp_pipeline.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cpu_pipeline.hpp"
#include "cuda_pipeline.hpp"

namespace holoscan::isp {

PlatformFacts probe_platform() noexcept {
  int devices = 0;
  return PlatformFacts{.cuda_device = cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0};
}

Backend resolve_backend(Backend backend) noexcept {
  return backend == Backend::kAuto ? resolve_auto(probe_platform()) : backend;
}

struct IspPipeline::Impl {
  Backend backend{};
  IspParams params;
  std::vector<float> lut;
  // CPU.
  detail::AwbState awb{};
  detail::CpuScratch scratch;
  // CUDA.
  std::optional<detail::CudaPipeline> cuda;
};

IspPipeline::IspPipeline(Backend backend, IspParams params) : impl_(std::make_unique<Impl>()) {
  validate_params(params);
  impl_->backend = resolve_backend(backend);
  impl_->lut = tone_lut(params);
  impl_->awb = detail::initial_white_balance(params);
  impl_->params = std::move(params);
  if (impl_->backend == Backend::kCuda) {
    impl_->cuda.emplace(impl_->params, impl_->lut);
  }
}

IspPipeline::~IspPipeline() = default;

Backend IspPipeline::backend() const noexcept { return impl_->backend; }

MemoryLocation IspPipeline::location() const noexcept {
  return impl_->backend == Backend::kCuda ? MemoryLocation::kDevice : MemoryLocation::kHost;
}

const IspParams& IspPipeline::params() const noexcept { return impl_->params; }

void IspPipeline::reset() {
  impl_->awb = detail::initial_white_balance(impl_->params);
  if (impl_->cuda) {
    impl_->cuda->reset();
  }
}

void IspPipeline::process(const RawImageView& raw, const RgbImageView& out,
                          const StatsOutput& stats, const FrameExposure& exposure,
                          cudaStream_t stream) {
  validate_raw(raw, impl_->params);
  if (out.data == nullptr) {
    throw std::invalid_argument("output image has no data");
  }
  const std::size_t row = static_cast<std::size_t>(raw.width) * output_channels(out.format) *
                          output_channel_bytes(out.format);
  if (out.pitch < row || out.pitch % static_cast<std::size_t>(output_channel_bytes(out.format))) {
    throw std::invalid_argument("output image pitch is shorter than a row or misaligned");
  }
  const bool cuda = impl_->cuda.has_value();
  const detail::FrameParams p = detail::make_frame_params(
      impl_->params, raw, out, cuda ? nullptr : impl_->lut.data(),
      static_cast<int>(impl_->lut.size()),
      cuda || !impl_->params.lens_shading ? nullptr : impl_->params.lens_shading->gains.data());
  if (!cuda) {
    detail::cpu_process(p, impl_->params, stats, exposure, impl_->awb, impl_->scratch);
    return;
  }
  if (const cudaError_t status = impl_->cuda->process(p, stats, exposure, stream);
      status != cudaSuccess) {
    throw std::runtime_error(std::string("IspPipeline (cuda): ") + cudaGetErrorString(status));
  }
}

}  // namespace holoscan::isp
