// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// GPU helpers for the tests: device buffers, the synthetic rig's frames, and
// a stitch downloaded to the host.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/stitching/stitcher.hpp>
#include <holoscan/stitching/synthetic_world.hpp>

namespace holoscan::stitching::test {

inline void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    cuda_check(cudaMalloc(&data_, bytes == 0 ? 1 : bytes), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), bytes_(other.bytes_) {}
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(DeviceBuffer&&) = delete;

  [[nodiscard]] void* data() const { return data_; }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }
  [[nodiscard]] std::vector<std::uint8_t> download() const {
    std::vector<std::uint8_t> host(bytes_);
    cuda_check(cudaMemcpy(host.data(), data_, bytes_, cudaMemcpyDeviceToHost), "cudaMemcpy");
    return host;
  }

 private:
  void* data_{};
  std::size_t bytes_{};
};

// One frame of every camera of `rig`, rendered on the device.
struct RigFrame {
  std::vector<DeviceBuffer> buffers;
  std::vector<DeviceImage> images;
  std::vector<std::vector<std::uint8_t>> host;  // The same pixels, on the host.
  std::vector<HostImage> host_images;
};

// `pad` extra bytes per row test pitched images.
inline RigFrame render_frame(const SyntheticWorld& world, const CameraRig& rig, const RigPose& pose,
                             PixelFormat format, const std::vector<float>& gains = {},
                             int samples = 3, std::size_t pad = 0) {
  RigFrame frame;
  for (std::size_t c = 0; c < rig.cameras.size(); ++c) {
    const Camera& camera = rig.cameras[c];
    const std::size_t pitch = static_cast<std::size_t>(camera.width) * channels(format) + pad;
    frame.buffers.emplace_back(pitch * camera.height);
    const float gain = gains.empty() ? 1.0F : gains[c];
    cuda_check(render_camera(world, camera, pose, gain, format, frame.buffers.back().data(), pitch,
                             samples, nullptr),
               "render_camera");
    frame.images.push_back(
        DeviceImage{frame.buffers.back().data(), camera.width, camera.height, pitch, format});
  }
  cuda_check(cudaDeviceSynchronize(), "render");
  for (std::size_t c = 0; c < rig.cameras.size(); ++c) {
    frame.host.push_back(frame.buffers[c].download());
  }
  for (std::size_t c = 0; c < rig.cameras.size(); ++c) {
    const DeviceImage& d = frame.images[c];
    frame.host_images.push_back(
        HostImage{frame.host[c].data(), d.width, d.height, d.pitch, d.format});
  }
  return frame;
}

// Stitches `frame` and returns the output on the host, packed.
inline std::vector<std::uint8_t> stitch_frame(Stitcher& stitcher, const RigFrame& frame) {
  const OutputGeometry& g = stitcher.geometry();
  const std::size_t pitch =
      static_cast<std::size_t>(g.width) * channels(stitcher.settings().output_format);
  DeviceBuffer output(pitch * g.height);
  cuda_check(stitcher.stitch(frame.images, output.data(), pitch, nullptr), "stitch");
  cuda_check(cudaDeviceSynchronize(), "stitch");
  return output.download();
}

// The truth of `geometry` at `pose`, RGBA8, packed.
inline std::vector<std::uint8_t> truth_image(const SyntheticWorld& world,
                                             const OutputGeometry& geometry, const RigPose& pose,
                                             int samples = 3) {
  const std::size_t pitch = static_cast<std::size_t>(geometry.width) * 4U;
  DeviceBuffer buffer(pitch * geometry.height);
  cuda_check(render_truth(world, geometry, pose, buffer.data(), pitch, samples, nullptr),
             "render_truth");
  cuda_check(cudaDeviceSynchronize(), "render_truth");
  return buffer.download();
}

// Whether a CUDA device is usable.
inline bool have_gpu() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace holoscan::stitching::test
