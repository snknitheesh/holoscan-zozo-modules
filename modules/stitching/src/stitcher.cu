// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "blend_setup.hpp"
#include "holoscan/stitching/stitcher.hpp"
#include "stitch_math.hpp"

namespace holoscan::stitching {
namespace {

using detail::BlendParams;
using detail::CameraParams;
using detail::GeometryParams;
using Count = unsigned long long;  // NOLINT(runtime/int): atomicAdd's 64-bit type.

constexpr dim3 kBlock{32, 8, 1};

dim3 grid_for(int width, int height) {
  return dim3((static_cast<unsigned>(width) + kBlock.x - 1) / kBlock.x,
              (static_cast<unsigned>(height) + kBlock.y - 1) / kBlock.y, 1);
}

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("Stitcher: ") + what + ": " + cudaGetErrorString(status));
  }
}

// Which output pixels a camera sees: a flag per column, the first and last
// rows, and how many. One warp is one stretch of a row (kBlock.x == 32).
__global__ void coverage_kernel(GeometryParams g, CameraParams c, std::uint8_t* columns, int* rows,
                                Count* covered) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  bool seen = false;
  if (x < g.width && y < g.height) {
    float u;
    float v;
    float margin;
    seen = detail::remap_entry(g, c, x, y, u, v, margin);
  }
  if (seen) {
    columns[x] = 1U;
  }
  const unsigned votes = __ballot_sync(0xffffffffU, seen);
  if (votes != 0U && (threadIdx.x & 31U) == static_cast<unsigned>(__ffs(votes) - 1)) {
    atomicAdd(covered, static_cast<Count>(__popc(votes)));
    atomicMin(&rows[0], y);
    atomicMax(&rows[1], y);
  }
}

// One camera's table over its region of interest.
__global__ void table_kernel(GeometryParams g, CameraParams c, detail::Roi roi, float* uv,
                             float* margin) {
  const int dx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int dy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (dx >= roi.width || dy >= roi.height) {
    return;
  }
  const int x = (roi.x0 + dx) % g.width;
  const int y = roi.y0 + dy;
  const std::size_t i = static_cast<std::size_t>(dy) * roi.width + dx;
  float u = 0.0F;
  float v = 0.0F;
  float m = 0.0F;
  if (detail::remap_entry(g, c, x, y, u, v, m)) {
    uv[2 * i] = u;
    uv[2 * i + 1] = v;
    margin[i] = m;
  } else {
    uv[2 * i] = 0.0F;
    uv[2 * i + 1] = 0.0F;
    margin[i] = 0.0F;
  }
}

// The overlap statistics of the exposure sample pixels, summed per block in
// shared memory, then into `sum` and `count` (cameras x cameras). Integers, so
// the result does not depend on the order of the additions.
__global__ void stats_kernel(BlendParams p, Count* sum, std::uint32_t* count) {
  constexpr int kPairs = detail::kMaxCameras * detail::kMaxCameras;
  __shared__ Count block_sum[kPairs];
  __shared__ unsigned block_count[kPairs];
  const int t = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  for (int i = t; i < kPairs; i += static_cast<int>(blockDim.x * blockDim.y)) {
    block_sum[i] = 0U;
    block_count[i] = 0U;
  }
  __syncthreads();
  constexpr int kHalf = kExposureStride / 2;
  const int x = kHalf + kExposureStride * static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = kHalf + kExposureStride * static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int n = p.cameras;
  if (x < p.output_width && y < p.output_height) {
    int luminance[detail::kMaxCameras];
    detail::sample_luminance(p, x, y, luminance);
    for (int i = 0; i < n; ++i) {
      if (luminance[i] < 0) continue;
      for (int j = 0; j < n; ++j) {
        if (j != i && luminance[j] >= 0) {
          atomicAdd(&block_sum[i * n + j], static_cast<Count>(luminance[i]));
          atomicAdd(&block_count[i * n + j], 1U);
        }
      }
    }
  }
  __syncthreads();
  for (int i = t; i < n * n; i += static_cast<int>(blockDim.x * blockDim.y)) {
    if (block_count[i] != 0U) {
      atomicAdd(&sum[i], block_sum[i]);
      atomicAdd(&count[i], block_count[i]);
    }
  }
}

// The gain solve, in place over the previous gains: a few cameras, one thread.
__global__ void gains_kernel(int n, const Count* sum, const std::uint32_t* count, double alpha,
                             double beta, double smoothing, float* gains) {
  detail::solve_gains(n, sum, count, alpha, beta, smoothing, gains, gains);
}

__global__ void blend_kernel(BlendParams p, const float* gains, std::uint8_t* output,
                             std::size_t pitch) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.output_width || y >= p.output_height) {
    return;
  }
  std::uint8_t pixel[4];
  detail::blend_pixel(p, gains, x, y, pixel);
  std::uint8_t* out = output + static_cast<std::size_t>(y) * pitch +
                      static_cast<std::size_t>(x) * static_cast<std::size_t>(p.output_channels);
  for (int ch = 0; ch < p.output_channels; ++ch) {
    out[ch] = pixel[ch];
  }
}

// cudaMalloc that throws.
template <typename T>
T* device_alloc(std::size_t count) {
  void* pointer = nullptr;
  check(cudaMalloc(&pointer, std::max<std::size_t>(count, 1) * sizeof(T)), "cudaMalloc");
  return static_cast<T*>(pointer);
}

}  // namespace

struct Stitcher::State {
  CameraRig rig;
  OutputGeometry geometry;
  BlendSettings settings;
  bool wraps{};
  std::vector<detail::Roi> rois;
  std::vector<float*> uv;
  std::vector<float*> margin;
  std::vector<std::uint64_t> covered;
  std::size_t table_bytes{};
  Count* sum{};
  std::uint32_t* count{};
  float* gains{};

  ~State() {
    // Frees wait for the work that may still use the buffers.
    for (float* p : uv) cudaFree(p);
    for (float* p : margin) cudaFree(p);
    cudaFree(sum);
    cudaFree(count);
    cudaFree(gains);
  }
};

Stitcher::Stitcher(CameraRig rig, OutputGeometry geometry, BlendSettings settings)
    : state_(std::make_unique<State>()) {
  validate(rig);
  validate(geometry);
  validate(settings);
  State& s = *state_;
  s.rig = std::move(rig);
  s.geometry = geometry;
  s.settings = settings;
  s.wraps = wraps(geometry);
  const int n = static_cast<int>(s.rig.cameras.size());
  const GeometryParams g = detail::geometry_params(geometry);

  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  struct StreamGuard {
    cudaStream_t stream;
    ~StreamGuard() {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  } guard{stream};

  // Coverage of every camera, then each table over its region of interest.
  const std::size_t w = static_cast<std::size_t>(g.width);
  std::uint8_t* columns = device_alloc<std::uint8_t>(w * n);
  int* rows = device_alloc<int>(2U * n);
  Count* covered = device_alloc<Count>(n);
  struct Temporaries {
    std::uint8_t* columns;
    int* rows;
    Count* covered;
    ~Temporaries() {
      cudaFree(columns);
      cudaFree(rows);
      cudaFree(covered);
    }
  } temporaries{columns, rows, covered};
  std::vector<int> initial_rows;
  for (int c = 0; c < n; ++c) {
    initial_rows.push_back(g.height);
    initial_rows.push_back(-1);
  }
  check(cudaMemsetAsync(columns, 0, w * n, stream), "cudaMemsetAsync");
  check(cudaMemsetAsync(covered, 0, sizeof(Count) * n, stream), "cudaMemsetAsync");
  check(cudaMemcpyAsync(rows, initial_rows.data(), sizeof(int) * 2U * n, cudaMemcpyHostToDevice,
                        stream),
        "cudaMemcpyAsync");
  std::vector<CameraParams> cameras;
  for (int c = 0; c < n; ++c) {
    cameras.push_back(detail::camera_params(s.rig.cameras[c]));
    coverage_kernel<<<grid_for(g.width, g.height), kBlock, 0, stream>>>(
        g, cameras.back(), columns + w * c, rows + 2 * c, covered + c);
    check(cudaGetLastError(), "coverage kernel");
  }
  std::vector<std::uint8_t> host_columns(w * n);
  std::vector<int> host_rows(2U * n);
  std::vector<Count> host_covered(n);
  check(cudaMemcpyAsync(host_columns.data(), columns, w * n, cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync");
  check(
      cudaMemcpyAsync(host_rows.data(), rows, sizeof(int) * 2U * n, cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync");
  check(cudaMemcpyAsync(host_covered.data(), covered, sizeof(Count) * n, cudaMemcpyDeviceToHost,
                        stream),
        "cudaMemcpyAsync");
  check(cudaStreamSynchronize(stream), "coverage");

  for (int c = 0; c < n; ++c) {
    const std::vector<std::uint8_t> flags(
        host_columns.begin() + static_cast<std::ptrdiff_t>(w * c),
        host_columns.begin() + static_cast<std::ptrdiff_t>(w * (c + 1)));
    const detail::Roi roi =
        detail::region_of_interest(flags, host_rows[2 * c], host_rows[2 * c + 1], s.wraps);
    const std::size_t entries = static_cast<std::size_t>(roi.width) * roi.height;
    s.rois.push_back(roi);
    s.covered.push_back(host_covered[c]);
    s.uv.push_back(device_alloc<float>(2 * entries));
    s.margin.push_back(device_alloc<float>(entries));
    s.table_bytes += entries * 3U * sizeof(float);
    if (entries != 0) {
      table_kernel<<<grid_for(roi.width, roi.height), kBlock, 0, stream>>>(g, cameras[c], roi,
                                                                           s.uv[c], s.margin[c]);
      check(cudaGetLastError(), "table kernel");
    }
  }
  s.sum = device_alloc<Count>(static_cast<std::size_t>(n) * n);
  s.count = device_alloc<std::uint32_t>(static_cast<std::size_t>(n) * n);
  s.gains = device_alloc<float>(n);
  reset_gains(stream);
  check(cudaStreamSynchronize(stream), "building the tables");
}

Stitcher::~Stitcher() = default;

cudaError_t Stitcher::stitch(std::span<const DeviceImage> images, void* output,
                             std::size_t output_pitch, cudaStream_t stream) {
  const State& s = *state_;
  const std::size_t n = s.rig.cameras.size();
  if (images.size() != n) {
    throw std::invalid_argument("Stitcher: " + std::to_string(images.size()) + " images for " +
                                std::to_string(n) + " cameras");
  }
  const int out_channels = channels(s.settings.output_format);
  if (output == nullptr ||
      output_pitch < static_cast<std::size_t>(s.geometry.width) * out_channels) {
    throw std::invalid_argument("Stitcher: the output's pitch is shorter than a row");
  }
  BlendParams params =
      detail::blend_params(s.settings, s.geometry.width, s.geometry.height, s.wraps);
  for (std::size_t c = 0; c < n; ++c) {
    const DeviceImage& image = images[c];
    const Camera& camera = s.rig.cameras[c];
    if (image.data == nullptr || image.width != camera.width || image.height != camera.height ||
        image.pitch < static_cast<std::size_t>(image.width) * channels(image.format)) {
      throw std::invalid_argument(
          "Stitcher: image " + std::to_string(c) + " is not a " + std::to_string(camera.width) +
          "x" + std::to_string(camera.height) + " image of camera " + camera.name);
    }
    detail::set_camera(params, static_cast<int>(c), static_cast<const std::uint8_t*>(image.data),
                       image.pitch, image.width, image.height, image.format, s.rois[c], s.uv[c],
                       s.margin[c]);
  }
  if (s.settings.exposure_compensation) {
    const std::size_t pairs = n * n;
    if (cudaError_t status = cudaMemsetAsync(s.sum, 0, sizeof(Count) * pairs, stream);
        status != cudaSuccess) {
      return status;
    }
    if (cudaError_t status = cudaMemsetAsync(s.count, 0, sizeof(std::uint32_t) * pairs, stream);
        status != cudaSuccess) {
      return status;
    }
    const int half = kExposureStride / 2;
    const int samples_x = (s.geometry.width - half + kExposureStride - 1) / kExposureStride;
    const int samples_y = (s.geometry.height - half + kExposureStride - 1) / kExposureStride;
    stats_kernel<<<grid_for(samples_x, samples_y), kBlock, 0, stream>>>(params, s.sum, s.count);
    if (cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
      return status;
    }
    gains_kernel<<<1, 1, 0, stream>>>(static_cast<int>(n), s.sum, s.count,
                                      detail::gain_alpha(s.settings), detail::gain_beta(s.settings),
                                      s.settings.exposure_smoothing, s.gains);
    if (cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
      return status;
    }
  }
  blend_kernel<<<grid_for(s.geometry.width, s.geometry.height), kBlock, 0, stream>>>(
      params, s.gains, static_cast<std::uint8_t*>(output), output_pitch);
  return cudaGetLastError();
}

std::vector<float> Stitcher::gains(cudaStream_t stream) const {
  std::vector<float> result(state_->rig.cameras.size());
  check(cudaMemcpyAsync(result.data(), state_->gains, sizeof(float) * result.size(),
                        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  return result;
}

void Stitcher::reset_gains(cudaStream_t stream) {
  const std::vector<float> ones(state_->rig.cameras.size(), 1.0F);
  check(cudaMemcpyAsync(state_->gains, ones.data(), sizeof(float) * ones.size(),
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync");
  // The source is on this stack.
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

RemapTables Stitcher::download_tables() const {
  const State& s = *state_;
  RemapTables tables;
  tables.output_width = s.geometry.width;
  tables.output_height = s.geometry.height;
  tables.wraps = s.wraps;
  for (std::size_t c = 0; c < s.rois.size(); ++c) {
    const detail::Roi& roi = s.rois[c];
    CameraTable table;
    table.image_width = s.rig.cameras[c].width;
    table.image_height = s.rig.cameras[c].height;
    table.x0 = roi.x0;
    table.y0 = roi.y0;
    table.width = roi.width;
    table.height = roi.height;
    const std::size_t entries = static_cast<std::size_t>(roi.width) * roi.height;
    table.uv.resize(2 * entries);
    table.margin.resize(entries);
    check(cudaMemcpy(table.uv.data(), s.uv[c], sizeof(float) * 2 * entries, cudaMemcpyDeviceToHost),
          "cudaMemcpy");
    check(cudaMemcpy(table.margin.data(), s.margin[c], sizeof(float) * entries,
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy");
    tables.cameras.push_back(std::move(table));
  }
  return tables;
}

std::size_t Stitcher::table_bytes() const noexcept { return state_->table_bytes; }

const std::vector<std::uint64_t>& Stitcher::covered_pixels() const noexcept {
  return state_->covered;
}

const CameraRig& Stitcher::rig() const noexcept { return state_->rig; }

const OutputGeometry& Stitcher::geometry() const noexcept { return state_->geometry; }

const BlendSettings& Stitcher::settings() const noexcept { return state_->settings; }

}  // namespace holoscan::stitching
