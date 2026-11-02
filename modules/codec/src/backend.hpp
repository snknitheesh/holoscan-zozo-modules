// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The interface each hardware backend implements, below VideoEncoder and VideoDecoder: NV12 frames
// in device memory in, access units out, and back. Module-private.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/codec/codec_types.hpp>
#include <holoscan/codec/video_codec.hpp>

namespace holoscan::codec::detail {

// An NV12 frame in device memory of even width and height.
struct Nv12Frame {
  const std::uint8_t* luma{};
  std::size_t luma_pitch{};
  const std::uint8_t* chroma{};
  std::size_t chroma_pitch{};
  int width{};
  int height{};
};

class EncoderBackend {
 public:
  virtual ~EncoderBackend() = default;
  EncoderBackend(const EncoderBackend&) = delete;
  EncoderBackend& operator=(const EncoderBackend&) = delete;

  // Encodes `frame`, of the size the backend was built for and ready to read, into one access
  // unit. With `keyframe`, an IDR picture with its parameter sets.
  // @throws std::runtime_error on failure.
  [[nodiscard]] virtual std::vector<std::uint8_t> encode(const Nv12Frame& frame, bool keyframe) = 0;

 protected:
  EncoderBackend() = default;
};

// A decoded frame, borrowed for the duration of a FrameSink call, and the id of the access unit it
// came from.
using FrameSink = std::function<void(const Nv12Frame& frame, std::uint64_t id)>;

class DecoderBackend {
 public:
  virtual ~DecoderBackend() = default;
  DecoderBackend(const DecoderBackend&) = delete;
  DecoderBackend& operator=(const DecoderBackend&) = delete;

  // Decodes one access unit, tagged `id`, and calls `sink` for every frame it completes. Frames
  // the sink reads on `stream` must be read before it returns or be ordered on `stream`.
  // @throws std::runtime_error on failure.
  virtual void decode(std::span<const std::uint8_t> access_unit, std::uint64_t id,
                      cudaStream_t stream, const FrameSink& sink) = 0;
  // Ends the stream: calls `sink` for every frame still held. The backend is not used after it.
  virtual void flush(cudaStream_t stream, const FrameSink& sink) = 0;

 protected:
  DecoderBackend() = default;
};

// Availability of one backend, as probe_*_backends() report it.
struct Probe {
  bool available{};
  std::string detail;
};

// NVENC / NVDEC through the Video Codec SDK (nvcodec_encoder.cpp, nvcodec_decoder.cpp).
[[nodiscard]] Probe probe_nvcodec_encoder(Codec codec);
[[nodiscard]] Probe probe_nvcodec_decoder(Codec codec);
[[nodiscard]] std::unique_ptr<EncoderBackend> create_nvcodec_encoder(
    const EncoderSettings& settings, int width, int height);
// With `low_latency`, each frame is handed out as soon as it is decoded, which needs a stream with
// one reference frame and no reordering, as VideoEncoder makes.
[[nodiscard]] std::unique_ptr<DecoderBackend> create_nvcodec_decoder(Codec codec, bool low_latency);

// The Jetson V4L2 devices (v4l2_encoder.cpp, v4l2_decoder.cpp); unavailable where the build has no
// V4L2 backend.
[[nodiscard]] Probe probe_v4l2_encoder(Codec codec);
[[nodiscard]] Probe probe_v4l2_decoder(Codec codec);
[[nodiscard]] std::unique_ptr<EncoderBackend> create_v4l2_encoder(const EncoderSettings& settings,
                                                                  int width, int height);
[[nodiscard]] std::unique_ptr<DecoderBackend> create_v4l2_decoder(Codec codec, bool low_latency);

// RGB, BGR, RGBA, BGRA or NV12 `image` to NV12 of `width` x `height` (even, at least the image's
// size), repeating the image's last column and row into the extra ones (color_convert.cu).
cudaError_t launch_to_nv12(const DeviceImage& image, std::uint8_t* luma, std::size_t luma_pitch,
                           std::uint8_t* chroma, std::size_t chroma_pitch, int width, int height,
                           cudaStream_t stream);
// NV12 `frame` to `format`, its top-left `width` x `height` pixels (color_convert.cu).
cudaError_t launch_from_nv12(const Nv12Frame& frame, PixelFormat format, std::uint8_t* out,
                             std::size_t out_pitch, int width, int height, cudaStream_t stream);

// The CUDA device the runtime uses on this thread, initialized.
// @throws std::runtime_error when there is no usable device.
int current_device();

// "<what>: <cudaGetErrorString>" as a runtime_error, unless `status` is cudaSuccess.
void check_cuda(cudaError_t status, std::string_view what);

}  // namespace holoscan::codec::detail
