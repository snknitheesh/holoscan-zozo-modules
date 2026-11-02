// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Stitcher: the GPU engine behind StitchOp, usable without a graph. It builds
// the remap tables of a rig and an output geometry once, on the GPU, then
// stitches each frame's camera images, in CUDA device memory, on a CUDA stream:
// optional exposure gains from the overlaps, then bilinear sampling and
// blending. It computes what remap.hpp's CPU reference computes.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <holoscan/stitching/camera_rig.hpp>
#include <holoscan/stitching/projection.hpp>
#include <holoscan/stitching/remap.hpp>

namespace holoscan::stitching {

// A camera image in CUDA device memory: `pitch` bytes from one row to the next.
struct DeviceImage {
  const void* data{};
  int width{};
  int height{};
  std::size_t pitch{};
  PixelFormat format{PixelFormat::kRgb8};
};

class Stitcher {
 public:
  // Builds the tables on the current CUDA device and waits for them.
  // @throws std::invalid_argument if the rig, geometry or settings are
  // unusable; std::runtime_error if CUDA fails (out of memory included).
  Stitcher(CameraRig rig, OutputGeometry geometry, BlendSettings settings);
  ~Stitcher();
  Stitcher(const Stitcher&) = delete;
  Stitcher& operator=(const Stitcher&) = delete;

  // Stitches one image per camera, in the rig's order, each of its camera's
  // size, into `output`: the geometry's size in the settings' format,
  // `output_pitch` bytes per row, in CUDA device memory. Queued on `stream`;
  // the images must stay valid until it has run. With exposure compensation,
  // the gains of this frame are computed on the GPU first.
  // @throws std::invalid_argument if the images do not match the cameras.
  // @return the first CUDA error of queuing the work.
  cudaError_t stitch(std::span<const DeviceImage> images, void* output, std::size_t output_pitch,
                     cudaStream_t stream);

  // The gains of the last stitch, one per camera (all 1 without exposure
  // compensation, or before the first stitch). Waits for `stream`.
  [[nodiscard]] std::vector<float> gains(cudaStream_t stream) const;
  // Starts the gains over at 1, as before the first frame.
  void reset_gains(cudaStream_t stream);

  // The tables, copied to the host: for tests and tools.
  [[nodiscard]] RemapTables download_tables() const;
  // The GPU memory the tables take.
  [[nodiscard]] std::size_t table_bytes() const noexcept;
  // The output pixels each camera sees.
  [[nodiscard]] const std::vector<std::uint64_t>& covered_pixels() const noexcept;

  [[nodiscard]] const CameraRig& rig() const noexcept;
  [[nodiscard]] const OutputGeometry& geometry() const noexcept;
  [[nodiscard]] const BlendSettings& settings() const noexcept;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::stitching
