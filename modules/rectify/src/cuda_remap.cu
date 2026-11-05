// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The cuda backend: one thread per output pixel reads its source position
// from a dense map and samples the input, bilinear or nearest; a source
// outside [0, size - 1] gives zero. (VPI's remap with VPI_BORDER_ZERO may
// instead blend a source less than a pixel outside with zero; the maps move
// sources within half a pixel outside onto the edge, so the two can differ
// only for sources between half a pixel and a pixel outside.)

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "remapper_factories.hpp"

namespace holoscan::rectify {
namespace {

constexpr dim3 kBlock{32, 8, 1};

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cuda remap: ") + what + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T, int C>
struct Pixel {
  float v[C];
};

template <typename T, int C>
__device__ __forceinline__ void load(const unsigned char* image, std::size_t pitch, int width,
                                     int height, int x, int y, float weight, Pixel<T, C>& acc) {
  if (x < 0 || y < 0 || x >= width || y >= height) return;  // Zero border.
  const T* p = reinterpret_cast<const T*>(image + static_cast<std::size_t>(y) * pitch) + x * C;
#pragma unroll
  for (int c = 0; c < C; ++c) acc.v[c] += weight * static_cast<float>(p[c]);
}

template <typename T, int C, bool kLinear>
__global__ void remap_kernel(const unsigned char* __restrict__ input, std::size_t input_pitch,
                             int input_width, int input_height, unsigned char* __restrict__ output,
                             std::size_t output_pitch, int output_width, int output_height,
                             const float2* __restrict__ map) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= output_width || v >= output_height) return;
  const float2 s = map[static_cast<std::size_t>(v) * output_width + u];
  Pixel<T, C> acc{};
  // A source outside [0, size - 1] is zero, not blended with the edge.
  const bool inside = s.x >= 0.0f && s.y >= 0.0f && s.x <= static_cast<float>(input_width - 1) &&
                      s.y <= static_cast<float>(input_height - 1);
  if (!inside) {
    // Zero.
  } else if constexpr (kLinear) {
    const float fx = floorf(s.x);
    const float fy = floorf(s.y);
    const int x0 = static_cast<int>(fx);
    const int y0 = static_cast<int>(fy);
    const float ax = s.x - fx;
    const float ay = s.y - fy;
    load<T, C>(input, input_pitch, input_width, input_height, x0, y0, (1 - ax) * (1 - ay), acc);
    load<T, C>(input, input_pitch, input_width, input_height, x0 + 1, y0, ax * (1 - ay), acc);
    load<T, C>(input, input_pitch, input_width, input_height, x0, y0 + 1, (1 - ax) * ay, acc);
    load<T, C>(input, input_pitch, input_width, input_height, x0 + 1, y0 + 1, ax * ay, acc);
  } else {
    load<T, C>(input, input_pitch, input_width, input_height, static_cast<int>(floorf(s.x + 0.5f)),
               static_cast<int>(floorf(s.y + 0.5f)), 1.0f, acc);
  }
  T* out = reinterpret_cast<T*>(output + static_cast<std::size_t>(v) * output_pitch) + u * C;
  constexpr float kMax = sizeof(T) == 1 ? 255.0f : 65535.0f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    out[c] = static_cast<T>(fminf(fmaxf(floorf(acc.v[c] + 0.5f), 0.0f), kMax));
  }
}

template <typename T, int C>
void launch(bool linear, const DeviceImage& in, const DeviceImage& out, const float2* map,
            cudaStream_t stream) {
  const dim3 grid((static_cast<unsigned>(out.width) + kBlock.x - 1) / kBlock.x,
                  (static_cast<unsigned>(out.height) + kBlock.y - 1) / kBlock.y, 1);
  const auto* src = static_cast<const unsigned char*>(in.data);
  auto* dst = static_cast<unsigned char*>(out.data);
  if (linear) {
    remap_kernel<T, C, true><<<grid, kBlock, 0, stream>>>(src, in.pitch, in.width, in.height, dst,
                                                          out.pitch, out.width, out.height, map);
  } else {
    remap_kernel<T, C, false><<<grid, kBlock, 0, stream>>>(src, in.pitch, in.width, in.height, dst,
                                                           out.pitch, out.width, out.height, map);
  }
}

// The dense map lives in device memory. Its upload is queued on the stream of
// the first submit, from pinned memory, ahead of the first kernel there, so
// stream order alone makes the kernels see it and no thread waits. The pinned
// copy goes once a later submit finds the upload done.
class CudaRemapper final : public Remapper {
 public:
  CudaRemapper(const RemapModel& model, int output_width, int output_height, PixelFormat format,
               Interpolation interpolation)
      : Remapper(Backend::kCuda, format, static_cast<int>(model.input().width),
                 static_cast<int>(model.input().height), output_width, output_height),
        linear_(interpolation == Interpolation::kLinear) {
    const RemapTable table = build_remap_table(model, output_width, output_height);
    map_bytes_ = table.xy.size() * sizeof(float);
    void* map = nullptr;
    check(cudaMalloc(&map, map_bytes_), "cudaMalloc");
    map_.reset(static_cast<float2*>(map));
    void* staging = nullptr;
    check(cudaMallocHost(&staging, map_bytes_), "cudaMallocHost");
    staging_.reset(staging);
    std::memcpy(staging, table.xy.data(), map_bytes_);
    cudaEvent_t uploaded = nullptr;
    check(cudaEventCreateWithFlags(&uploaded, cudaEventDisableTiming), "cudaEventCreate");
    uploaded_.reset(uploaded);
  }

  [[nodiscard]] std::string_view name() const noexcept override { return "cuda"; }

 protected:
  void do_submit(const DeviceImage& input, const DeviceImage& output, RemapQueue& queue) override {
    // VPI allows no CUDA work on the stream while its own is queued.
    finish_vpi_of(queue);
    const cudaStream_t stream = queue.cuda_stream();
    order_after_upload(stream);
    const float2* map = map_.get();
    switch (format_) {
      case PixelFormat::kMono8:
        launch<std::uint8_t, 1>(linear_, input, output, map, stream);
        break;
      case PixelFormat::kMono16:
        launch<std::uint16_t, 1>(linear_, input, output, map, stream);
        break;
      case PixelFormat::kRgb8:
      case PixelFormat::kBgr8:
        launch<std::uint8_t, 3>(linear_, input, output, map, stream);
        break;
      case PixelFormat::kRgba8:
      case PixelFormat::kBgra8:
        launch<std::uint8_t, 4>(linear_, input, output, map, stream);
        break;
    }
    check(cudaGetLastError(), "remap kernel launch");
  }

 private:
  // Queues the map's upload on the first stream; puts a later stream behind
  // it until it is known to be done, then frees the pinned copy.
  void order_after_upload(cudaStream_t stream) {
    if (!upload_queued_) {
      check(cudaMemcpyAsync(map_.get(), staging_.get(), map_bytes_, cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync");
      check(cudaEventRecord(uploaded_.get(), stream), "cudaEventRecord");
      upload_queued_ = true;
      return;
    }
    if (!staging_) return;
    const cudaError_t status = cudaEventQuery(uploaded_.get());
    if (status == cudaSuccess) {
      staging_.reset();
      uploaded_.reset();
      return;
    }
    if (status != cudaErrorNotReady) check(status, "cudaEventQuery");
    check(cudaStreamWaitEvent(stream, uploaded_.get(), 0), "cudaStreamWaitEvent");
  }

  struct Free {
    void operator()(float2* p) const noexcept { cudaFree(p); }
  };
  struct FreeHost {
    void operator()(void* p) const noexcept { cudaFreeHost(p); }
  };
  struct DestroyEvent {
    void operator()(cudaEvent_t event) const noexcept { cudaEventDestroy(event); }
  };
  bool linear_;
  std::size_t map_bytes_{};
  std::unique_ptr<void, FreeHost> staging_;
  std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, DestroyEvent> uploaded_;
  // Declared after the staging, so that it is freed first: cudaFree waits for
  // the work that may still use it, the upload from the staging included.
  std::unique_ptr<float2, Free> map_;
  bool upload_queued_{};
};

}  // namespace

std::unique_ptr<Remapper> create_cuda_remapper(const RemapModel& model, int output_width,
                                               int output_height, PixelFormat format,
                                               Interpolation interpolation) {
  return std::make_unique<CudaRemapper>(model, output_width, output_height, format, interpolation);
}

}  // namespace holoscan::rectify
