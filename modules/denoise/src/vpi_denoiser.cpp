// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The vpi backend: VPI 4's Temporal Noise Reduction (version 3) on the VIC or on
// CUDA. TNR takes full-range NV12 only, so other formats are converted on the
// GPU before and after it (nv12_convert.cu), with odd sizes padded by one pixel.
//
// Per frame: the frame is converted into a VPI image locked for CUDA, together
// with this module's noise estimate and the count of changed pixels against
// the previous output (a scene cut makes TNR start over); the stream is then
// waited for, the images unlocked, TNR submitted and waited for, and its output,
// locked for CUDA, converted into the output view. That output stays locked
// until the next frame has waited for the stream, as the conversion may still
// read it.

// VPI's C API.
extern "C" {
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/TemporalNoiseReduction.h>
}

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "denoiser_impl.hpp"
#include "noise_estimate.hpp"
#include "nv12_convert.hpp"

namespace holoscan::denoise::detail {
namespace {

void check_vpi(VPIStatus status, const char* what) {
  if (status == VPI_SUCCESS) return;
  char message[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
  vpiGetLastStatusMessage(message, sizeof(message));
  throw std::runtime_error(std::string("VPI ") + what + ": " + vpiStatusGetName(status) + ": " +
                           message);
}

VPITNRPreset vpi_preset(VpiPreset preset) {
  switch (preset) {
    case VpiPreset::kOutdoorLowLight:
      return VPI_TNR_PRESET_OUTDOOR_LOW_LIGHT;
    case VpiPreset::kOutdoorMediumLight:
      return VPI_TNR_PRESET_OUTDOOR_MEDIUM_LIGHT;
    case VpiPreset::kOutdoorHighLight:
      return VPI_TNR_PRESET_OUTDOOR_HIGH_LIGHT;
    case VpiPreset::kIndoorLowLight:
      return VPI_TNR_PRESET_INDOOR_LOW_LIGHT;
    case VpiPreset::kIndoorMediumLight:
      return VPI_TNR_PRESET_INDOOR_MEDIUM_LIGHT;
    case VpiPreset::kIndoorHighLight:
      return VPI_TNR_PRESET_INDOOR_HIGH_LIGHT;
    case VpiPreset::kDefault:
      break;
  }
  return VPI_TNR_PRESET_DEFAULT;
}

struct ImageDestroy {
  void operator()(VPIImage image) const noexcept { vpiImageDestroy(image); }
};
using VpiImagePtr = std::unique_ptr<std::remove_pointer_t<VPIImage>, ImageDestroy>;

// The share of the noise variance left in TNR's output, for the scene cut test.
constexpr float kOutputVariance = 0.25F;

// The smallest frame VPI's TNR runs on the VIC (Jetson Orin, VPI 4.1); smaller
// ones fail when submitted.
constexpr int kVicMinWidth = 64;
constexpr int kVicMinHeight = 16;

class VpiDenoiser final : public Denoiser::Impl {
 public:
  VpiDenoiser(const DenoiseParams& params, const VpiOptions& options) : params_(params) {
    const bool vic = vpi_vic_tnr_available();
    if (options.engine == VpiEngine::kVic && !vic) {
      throw std::invalid_argument("vpi engine vic: VPI cannot run TNR on a VIC here");
    }
    engine_ = options.engine == VpiEngine::kCuda || !vic ? VPI_BACKEND_CUDA : VPI_BACKEND_VIC;
    vic_required_ = options.engine == VpiEngine::kVic;
    active_engine_ = engine_;
    check_vpi(vpiInitTemporalNoiseReductionParams(&tnr_params_), "TNR parameters");
    tnr_params_.preset = vpi_preset(options.preset);
    tnr_params_.strength = params.strength;
    check_vpi(vpiStreamCreate(engine_ | VPI_BACKEND_CUDA, &stream_), "vpiStreamCreate");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&sigma_), sizeof(float)), "cudaMalloc");
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&partials_), kNoiseBlocks * sizeof(std::uint64_t)),
        "cudaMalloc");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&changed_), sizeof(unsigned int)), "cudaMalloc");
    check_cuda(cudaMallocHost(reinterpret_cast<void**>(&changed_host_), sizeof(unsigned int)),
               "cudaMallocHost");
  }

  ~VpiDenoiser() override {
    // A conversion may still read the locked output; wait for every stream.
    cudaDeviceSynchronize();
    release_images();
    if (stream_ != nullptr) vpiStreamDestroy(stream_);
    cudaFree(sigma_);
    cudaFree(partials_);
    cudaFree(changed_);
    cudaFreeHost(changed_host_);
  }

  Backend backend() const noexcept override { return Backend::kVpi; }
  std::string_view engine() const noexcept override {
    return active_engine_ == VPI_BACKEND_VIC ? "vpi-vic" : "vpi-cuda";
  }
  void reset() noexcept override { have_previous_ = false; }

  float noise_sigma(cudaStream_t stream) override {
    if (!sized_) return 0.0F;
    float value = 0.0F;
    check_cuda(cudaMemcpyAsync(&value, sigma_, sizeof(value), cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return value;
  }

  void process(const ImageView& input, const ImageView& output, cudaStream_t stream) override {
    const int width = input.width + (input.width & 1);
    const int height = input.height + (input.height & 1);
    if (!sized_ || width != width_ || height != height_) allocate(width, height, stream);

    // The frame into TNR's input image, the noise level, the changed pixels.
    const Nv12 frame = lock(input_.get(), VPI_LOCK_WRITE);
    check_cuda(launch_to_nv12(input, frame, stream), "conversion to NV12");
    const SamplePlane luma{
        .data = frame.y.data, .pitch = frame.y.pitch, .width = width, .height = height};
    if (params_.noise_sigma > 0.0F) {
      check_cuda(launch_set(sigma_, params_.noise_sigma, stream), "noise sigma");
    } else {
      check_cuda(launch_noise_estimate(luma, partials_, tuning::kMinSigma, sigma_, stream),
                 "noise estimate");
    }
    const bool compare = have_previous_ && locked_output_ >= 0;
    if (compare) {
      const SamplePlane previous{.data = previous_luma_.data,
                                 .pitch = previous_luma_.pitch,
                                 .width = width,
                                 .height = height};
      check_cuda(cudaMemsetAsync(changed_, 0, sizeof(unsigned int), stream), "cudaMemsetAsync");
      check_cuda(launch_count_changed(luma, previous, sigma_, kOutputVariance,
                                      params_.motion_threshold, changed_, stream),
                 "change count");
      check_cuda(cudaMemcpyAsync(changed_host_, changed_, sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync");
    }
    // VPI may read the images once our stream has written them.
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    check_vpi(vpiImageUnlock(input_.get()), "vpiImageUnlock");
    unlock_output();
    const auto pixels = static_cast<double>(width) * static_cast<double>(height);
    const bool cut =
        compare && static_cast<double>(*changed_host_) > params_.scene_cut_fraction * pixels;

    const int current = next_output_;
    VPIImage previous = have_previous_ && !cut ? outputs_[1 - current].get() : nullptr;
    check_vpi(vpiSubmitTemporalNoiseReduction(stream_, active_engine_, payload_, previous,
                                              input_.get(), outputs_[current].get(), &tnr_params_),
              "vpiSubmitTemporalNoiseReduction");
    check_vpi(vpiStreamSync(stream_), "vpiStreamSync");
    have_previous_ = true;
    next_output_ = 1 - current;

    // TNR's output into the output view; it stays locked until the next frame.
    const Nv12 result = lock(outputs_[current].get(), VPI_LOCK_READ);
    locked_output_ = current;
    previous_luma_ = result.y;
    check_cuda(launch_from_nv12(result, output, input, stream), "conversion from NV12");
  }

 private:
  Nv12 lock(VPIImage image, VPILockMode mode) const {
    VPIImageData data{};
    check_vpi(vpiImageLockData(image, mode, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &data),
              "vpiImageLockData");
    const VPIImageBufferPitchLinear& pitch = data.buffer.pitch;
    const auto plane = [&](int index) {
      const VPIImagePlanePitchLinear& p = pitch.planes[index];
      return Plane{.data = static_cast<std::uint8_t*>(p.pBase) + p.offsetBytes,
                   .pitch = static_cast<std::size_t>(p.pitchBytes)};
    };
    return Nv12{.y = plane(0), .uv = plane(1), .width = width_, .height = height_};
  }

  void unlock_output() {
    if (locked_output_ >= 0) {
      check_vpi(vpiImageUnlock(outputs_[locked_output_].get()), "vpiImageUnlock");
      locked_output_ = -1;
    }
  }

  void release_images() noexcept {
    if (locked_output_ >= 0) vpiImageUnlock(outputs_[locked_output_].get());
    locked_output_ = -1;
    input_.reset();
    for (auto& image : outputs_) image.reset();
    if (payload_ != nullptr) vpiPayloadDestroy(payload_);
    payload_ = nullptr;
  }

  void allocate(int width, int height, cudaStream_t stream) {
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    release_images();
    sized_ = false;
    // Frames too small for the VIC run on CUDA, unless the VIC was asked for.
    const bool vic_fits = width >= kVicMinWidth && height >= kVicMinHeight;
    if (engine_ == VPI_BACKEND_VIC && !vic_fits && vic_required_) {
      throw std::invalid_argument("vpi engine vic: the VIC runs TNR on frames of at least " +
                                  std::to_string(kVicMinWidth) + "x" +
                                  std::to_string(kVicMinHeight));
    }
    active_engine_ = engine_ == VPI_BACKEND_VIC && vic_fits ? VPI_BACKEND_VIC : VPI_BACKEND_CUDA;
    width_ = width;
    height_ = height;
    const std::uint64_t flags = engine_ | VPI_BACKEND_CUDA | VPI_BACKEND_CPU;
    const auto create = [&] {
      VPIImage image = nullptr;
      check_vpi(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_NV12_ER, flags, &image),
                "vpiImageCreate");
      return VpiImagePtr(image);
    };
    input_ = create();
    outputs_[0] = create();
    outputs_[1] = create();
    check_vpi(vpiCreateTemporalNoiseReduction(active_engine_, width, height,
                                              VPI_IMAGE_FORMAT_NV12_ER, VPI_TNR_DEFAULT, &payload_),
              "vpiCreateTemporalNoiseReduction");
    have_previous_ = false;
    next_output_ = 0;
    sized_ = true;
  }

  DenoiseParams params_;
  // The engine asked for (or the VIC where there is one), and the one this
  // frame size runs on.
  std::uint64_t engine_{VPI_BACKEND_CUDA};
  std::uint64_t active_engine_{VPI_BACKEND_CUDA};
  bool vic_required_{false};
  VPITNRParams tnr_params_{};
  VPIStream stream_{};
  VPIPayload payload_{};
  VpiImagePtr input_;
  std::array<VpiImagePtr, 2> outputs_;
  int next_output_{0};
  int locked_output_{-1};
  Plane previous_luma_{};
  bool have_previous_{false};
  bool sized_{false};
  int width_{};
  int height_{};
  float* sigma_{};
  std::uint64_t* partials_{};
  unsigned int* changed_{};
  unsigned int* changed_host_{};
};

}  // namespace

bool vpi_vic_tnr_available() noexcept {
  static const bool available = [] {
    VPIPayload payload = nullptr;
    if (vpiCreateTemporalNoiseReduction(VPI_BACKEND_VIC, 64, 64, VPI_IMAGE_FORMAT_NV12_ER,
                                        VPI_TNR_DEFAULT, &payload) != VPI_SUCCESS) {
      return false;
    }
    vpiPayloadDestroy(payload);
    return true;
  }();
  return available;
}

std::unique_ptr<Denoiser::Impl> make_vpi_denoiser(const DenoiseParams& params,
                                                  const VpiOptions& options) {
  return std::make_unique<VpiDenoiser>(params, options);
}

}  // namespace holoscan::denoise::detail
