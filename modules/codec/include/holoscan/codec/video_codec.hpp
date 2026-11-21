// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Hardware video encoding and decoding of GPU images, without Holoscan: VideoEncoder turns images
// in CUDA device memory into H.264 or H.265 access units, VideoDecoder turns access units back into
// NV12 images in device memory. VideoEncoderOp and VideoDecoderOp wrap them; tests and tools use
// them directly.
//
// Colors: RGB is converted to and from YUV with the BT.709 matrix in limited range (16 to 235),
// which the encoder also writes into the stream where its backend can.

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

namespace holoscan::codec {

// A borrowed image in CUDA device memory.
struct DeviceImage {
  // The pixels of a packed format, or the luma plane of kNv12.
  const std::uint8_t* data{};
  std::size_t pitch{};  // Bytes from one row to the next.
  // The interleaved U, V plane of kNv12, (height + 1) / 2 rows of 2 * ((width + 1) / 2) bytes.
  const std::uint8_t* chroma{};
  std::size_t chroma_pitch{};
  int width{};
  int height{};
  PixelFormat format{PixelFormat::kRgb8};
};

// Whether a backend works here for one codec, and why not.
struct BackendStatus {
  Backend backend{Backend::kNvcodec};
  bool available{};
  // Why it is not available, or what it runs on when it is.
  std::string detail;
};

// Each backend's status for encoding (or, for the decoder, decoding) `codec`, in the order
// Backend::kAuto tries them: kNvcodec, then kV4l2. Probing loads the libraries and opens the
// device once, and the result is kept: later calls do not probe again.
[[nodiscard]] std::vector<BackendStatus> probe_encoder_backends(Codec codec);
[[nodiscard]] std::vector<BackendStatus> probe_decoder_backends(Codec codec);

// One encoded frame: an access unit of an Annex-B byte stream.
struct EncodedFrame {
  std::vector<std::uint8_t> data;
  // An IDR picture with its parameter sets, where a decoder can start.
  bool keyframe{};
  std::int64_t pts{};
  // Frames this encoder has encoded before this one, since it started or last changed size.
  std::uint64_t index{};
  // The size of the image encoded. An odd side is encoded one pixel larger, its last row or column
  // repeated; the decoder gives the larger size and the caller crops (VideoDecoderOp does).
  int width{};
  int height{};
};

class VideoEncoder {
 public:
  // Picks the backend now: the configured one, or for kAuto the first available one.
  // @throws std::invalid_argument for unusable settings.
  // @throws std::runtime_error when the backend is not available, saying why.
  explicit VideoEncoder(EncoderSettings settings);
  ~VideoEncoder();
  VideoEncoder(const VideoEncoder&) = delete;
  VideoEncoder& operator=(const VideoEncoder&) = delete;

  [[nodiscard]] const EncoderSettings& settings() const noexcept;
  // The backend in use: never kAuto.
  [[nodiscard]] Backend backend() const noexcept;
  // What the backend runs on, for logs: "NVENC on <the GPU's name>", ...
  [[nodiscard]] const std::string& backend_detail() const noexcept;

  // Encodes one image and returns its access unit. The first image, and the first of a new size,
  // start a new stream: the hardware session is (re)built and frame numbering starts again at 0.
  // The image is read on `stream` (after whatever is queued on it), and is no longer needed when
  // this returns. The CPU waits once, for the access unit (see the README's "CPU waits and
  // copies per frame").
  // @throws std::invalid_argument for an image the encoder cannot take (format, size).
  // @throws std::runtime_error on a hardware or driver failure.
  [[nodiscard]] EncodedFrame encode(const DeviceImage& image, std::int64_t pts,
                                    cudaStream_t stream);
  // Ends the stream: the next encode() starts a new one, with an IDR keyframe and its parameter
  // sets, numbered 0, as after a change of size. Cheap: the hardware session is kept.
  void end_stream() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// One decoded frame: NV12 in device memory owned by the decoder.
struct DecodedFrame {
  DeviceImage image;
  // The pts given with the access unit this frame was decoded from, whatever the order the frames
  // come out in (a stream with B-frames is decoded in another order than it is shown). The v4l2
  // decoder without low latency does not say which access unit a frame came from: there each
  // frame takes the smallest pts still waiting, which is its own when the pts increase in the
  // order the stream shows its frames, as presentation times do.
  std::int64_t pts{};
};

// Called for each frame a decode() or flush() completes, in display order, with the frame in the
// decoder's own memory: queue the work that reads it on the `stream` given to that call, and
// return. The decoder waits for the stream before it reuses the memory, so nothing is copied.
using DecodedFrameSink = std::function<void(const DecodedFrame& frame)>;

class VideoDecoder {
 public:
  // With `low_latency`, each frame comes out of the decode() of its own access unit. That needs a
  // stream with one reference frame and no reordering, which VideoEncoder makes in low-latency
  // mode; turn it off for other streams, such as those with B-frames, whose frames then come out
  // some access units later, in display order (on Jetson's V4L2 decoder; NVDEC only runs ahead a
  // little), and the rest at flush().
  // @throws std::runtime_error when the backend is not available, saying why.
  explicit VideoDecoder(Codec codec, Backend backend = Backend::kNvcodec, bool low_latency = true);
  ~VideoDecoder();
  VideoDecoder(const VideoDecoder&) = delete;
  VideoDecoder& operator=(const VideoDecoder&) = delete;

  [[nodiscard]] Codec codec() const noexcept;
  [[nodiscard]] Backend backend() const noexcept;
  [[nodiscard]] const std::string& backend_detail() const noexcept;

  // Decodes one access unit and hands `sink` the frames it completed, in display order. Access
  // units before the first keyframe cannot be decoded and give nothing; so does an empty one. An
  // exception from `sink` stops the decoding and comes out of this call.
  // @throws std::runtime_error on a hardware or driver failure.
  void decode(std::span<const std::uint8_t> access_unit, std::int64_t pts, cudaStream_t stream,
              const DecodedFrameSink& sink);
  // Ends the stream: hands `sink` every frame the decoder still holds. The next access unit must
  // be a keyframe.
  void flush(cudaStream_t stream, const DecodedFrameSink& sink);

  // The same, returning the frames as copies in memory of the decoder's that stays valid until
  // the next decode() or flush(). They are written on `stream`: wait on it before reading them,
  // and finish reading them (on whatever stream) before the next call, which reuses the memory.
  [[nodiscard]] std::vector<DecodedFrame> decode(std::span<const std::uint8_t> access_unit,
                                                 std::int64_t pts, cudaStream_t stream);
  [[nodiscard]] std::vector<DecodedFrame> flush(cudaStream_t stream);
  // Access units given before the first keyframe, which were dropped.
  [[nodiscard]] std::uint64_t skipped() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Converts `nv12` (kNv12) to `format` into `out` (device memory, `out_pitch` bytes per row), the
// top-left `width` x `height` pixels, on `stream`. kNv12 output writes the luma rows and then,
// at `out` + `out_pitch` * `height`, the chroma rows, at the same pitch.
// @throws std::invalid_argument when `width` or `height` exceed the image.
void convert_from_nv12(const DeviceImage& nv12, PixelFormat format, std::uint8_t* out,
                       std::size_t out_pitch, int width, int height, cudaStream_t stream);

}  // namespace holoscan::codec
