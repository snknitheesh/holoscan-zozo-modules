// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/codec/video_codec.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/codec/bitstream.hpp>

#include "backend.hpp"

namespace holoscan::codec {
namespace detail {

void check_cuda(cudaError_t status, std::string_view what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

int current_device() {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  // Initializes the runtime's primary context, which the hardware sessions share.
  check_cuda(cudaFree(nullptr), "initializing CUDA");
  return device;
}

}  // namespace detail

namespace {

using detail::check_cuda;

// Device memory from cudaMallocPitch, freed on destruction.
struct PitchedBuffer {
  std::uint8_t* data{};
  std::size_t pitch{};
  std::size_t width{};
  std::size_t rows{};

  PitchedBuffer() = default;
  PitchedBuffer(const PitchedBuffer&) = delete;
  PitchedBuffer& operator=(const PitchedBuffer&) = delete;
  PitchedBuffer(PitchedBuffer&& other) noexcept { *this = std::move(other); }
  PitchedBuffer& operator=(PitchedBuffer&& other) noexcept {
    std::swap(data, other.data);
    std::swap(pitch, other.pitch);
    std::swap(width, other.width);
    std::swap(rows, other.rows);
    return *this;
  }
  ~PitchedBuffer() { cudaFree(data); }

  // At least `bytes_wide` x `row_count`, reallocated (contents lost) when smaller.
  void reserve(std::size_t bytes_wide, std::size_t row_count) {
    if (data != nullptr && width >= bytes_wide && rows >= row_count) {
      return;
    }
    cudaFree(data);
    data = nullptr;
    void* memory = nullptr;
    check_cuda(cudaMallocPitch(&memory, &pitch, bytes_wide, row_count), "cudaMallocPitch");
    data = static_cast<std::uint8_t*>(memory);
    width = bytes_wide;
    rows = row_count;
  }
};

// probe_*_backends(), once per direction and codec.
std::vector<BackendStatus> probe_cached(bool encoder, Codec codec) {
  static std::mutex mutex;
  static std::map<std::pair<bool, Codec>, std::vector<BackendStatus>> cache;
  const std::scoped_lock lock(mutex);
  const auto key = std::make_pair(encoder, codec);
  if (const auto found = cache.find(key); found != cache.end()) {
    return found->second;
  }
  const detail::Probe nvcodec =
      encoder ? detail::probe_nvcodec_encoder(codec) : detail::probe_nvcodec_decoder(codec);
  detail::Probe v4l2;
  if (nvcodec.available) {
    // Where the Video Codec SDK works (Jetson Thor), NVIDIA's V4L2 codec devices are a layer over
    // it with other buffer semantics, which this backend does not drive: use nvcodec directly.
    v4l2.detail = "the Video Codec SDK works here, so the nvcodec backend is used instead";
  } else {
    v4l2 = encoder ? detail::probe_v4l2_encoder(codec) : detail::probe_v4l2_decoder(codec);
  }
  std::vector<BackendStatus> statuses{
      {.backend = Backend::kNvcodec, .available = nvcodec.available, .detail = nvcodec.detail},
      {.backend = Backend::kV4l2, .available = v4l2.available, .detail = v4l2.detail},
  };
  cache.emplace(key, statuses);
  return statuses;
}

// The configured backend if it is available, or for kAuto the first available one, and what it
// runs on.
// @throws std::runtime_error naming why none is.
std::pair<Backend, std::string> choose_backend(bool encoder, Codec codec, Backend wanted) {
  const std::vector<BackendStatus> statuses = probe_cached(encoder, codec);
  std::string reasons;
  std::string others;
  for (const BackendStatus& status : statuses) {
    if (wanted != Backend::kAuto && status.backend != wanted) {
      if (status.available) {
        others +=
            std::string(others.empty() ? "" : ", ") + std::string(backend_name(status.backend));
      }
      continue;
    }
    if (status.available) {
      return {status.backend, status.detail};
    }
    reasons += (reasons.empty() ? "" : "; ") + std::string(backend_name(status.backend)) + ": " +
               status.detail;
  }
  if (!others.empty()) {
    reasons += "; " + others + " works here: choose it, or auto";
  }
  throw std::runtime_error(std::string("no ") + (encoder ? "encoder" : "decoder") + " for " +
                           std::string(codec_name(codec)) + " (" +
                           std::string(backend_name(wanted)) + "): " + reasons);
}

void check_image(const DeviceImage& image) {
  if (image.data == nullptr || (image.format == PixelFormat::kNv12 && image.chroma == nullptr)) {
    throw std::invalid_argument("the image has no pixels");
  }
  if (image.width < kMinFrameSide || image.height < kMinFrameSide || image.width > kMaxFrameSide ||
      image.height > kMaxFrameSide) {
    throw std::invalid_argument("images must be " + std::to_string(kMinFrameSide) + " to " +
                                std::to_string(kMaxFrameSide) + " pixels on each side, not " +
                                std::to_string(image.width) + "x" + std::to_string(image.height));
  }
  const std::size_t row = image.format == PixelFormat::kNv12
                              ? static_cast<std::size_t>(image.width)
                              : static_cast<std::size_t>(image.width) *
                                    static_cast<std::size_t>(bytes_per_pixel(image.format));
  if (pixel_format_name(image.format) == "unknown" || image.pitch < row ||
      (image.format == PixelFormat::kNv12 &&
       image.chroma_pitch < 2 * static_cast<std::size_t>((image.width + 1) / 2))) {
    throw std::invalid_argument("the image's format or row pitch is not usable");
  }
}

}  // namespace

std::vector<BackendStatus> probe_encoder_backends(Codec codec) { return probe_cached(true, codec); }

std::vector<BackendStatus> probe_decoder_backends(Codec codec) {
  return probe_cached(false, codec);
}

struct VideoEncoder::Impl {
  EncoderSettings settings;
  Backend backend{};
  std::string detail;
  std::unique_ptr<detail::EncoderBackend> session;
  int session_width{};
  int session_height{};
  // The frame the backend encodes: NV12, its chroma right after its luma at the same pitch.
  PitchedBuffer nv12;
  std::uint64_t index{};
  // The latest parameter sets, for a keyframe that comes without them.
  std::vector<std::uint8_t> parameter_sets;
};

VideoEncoder::VideoEncoder(EncoderSettings settings) : impl_(std::make_unique<Impl>()) {
  validate_settings(settings);
  impl_->settings = settings;
  std::tie(impl_->backend, impl_->detail) = choose_backend(true, settings.codec, settings.backend);
}

VideoEncoder::~VideoEncoder() = default;

const EncoderSettings& VideoEncoder::settings() const noexcept { return impl_->settings; }
Backend VideoEncoder::backend() const noexcept { return impl_->backend; }
const std::string& VideoEncoder::backend_detail() const noexcept { return impl_->detail; }

EncodedFrame VideoEncoder::encode(const DeviceImage& image, std::int64_t pts, cudaStream_t stream) {
  check_image(image);
  Impl& impl = *impl_;
  // 4:2:0 needs even sides: an odd one is encoded one larger, its last row or column repeated.
  const int width = (image.width + 1) & ~1;
  const int height = (image.height + 1) & ~1;
  if (!impl.session || width != impl.session_width || height != impl.session_height) {
    impl.session.reset();
    impl.session = impl.backend == Backend::kNvcodec
                       ? detail::create_nvcodec_encoder(impl.settings, width, height)
                       : detail::create_v4l2_encoder(impl.settings, width, height);
    impl.session_width = width;
    impl.session_height = height;
    impl.index = 0;
    impl.parameter_sets.clear();
  }
  impl.nv12.reserve(static_cast<std::size_t>(width), static_cast<std::size_t>(height) * 3 / 2);
  std::uint8_t* chroma = impl.nv12.data + impl.nv12.pitch * static_cast<std::size_t>(height);
  check_cuda(detail::launch_to_nv12(image, impl.nv12.data, impl.nv12.pitch, chroma, impl.nv12.pitch,
                                    width, height, stream),
             "converting the image to NV12");
  // The backend reads the frame after the conversion queued on `stream`, and returns once it has.

  const Codec codec = impl.settings.codec;
  const bool keyframe = impl.index % impl.settings.gop == 0;
  std::vector<std::uint8_t> access_unit =
      impl.session->encode(detail::Nv12Frame{.luma = impl.nv12.data,
                                             .luma_pitch = impl.nv12.pitch,
                                             .chroma = chroma,
                                             .chroma_pitch = impl.nv12.pitch,
                                             .width = width,
                                             .height = height},
                           keyframe, stream);
  const AccessUnitInfo info = analyze_access_unit(codec, access_unit);
  if (info.sps) {
    impl.parameter_sets = parameter_sets(codec, access_unit);
  } else if (info.idr && !impl.parameter_sets.empty()) {
    access_unit.insert(access_unit.begin(), impl.parameter_sets.begin(), impl.parameter_sets.end());
  }
  const bool is_key = is_keyframe(codec, access_unit);
  return EncodedFrame{.data = std::move(access_unit),
                      .keyframe = is_key,
                      .pts = pts,
                      .index = impl.index++,
                      .width = image.width,
                      .height = image.height};
}

void VideoEncoder::end_stream() noexcept {
  impl_->index = 0;
  if (impl_->backend == Backend::kV4l2) {
    // The V4L2 encoder places IDRs by its own frame count: a new session starts at frame 0.
    impl_->session.reset();
  }
}

// Access units whose frame has not come out yet are kept by id up to this many: more than any
// stream holds back (H.264 and H.265 keep at most 16 frames for reordering), so an id dropped
// for room belongs to an access unit that gave no frame.
constexpr std::size_t kMaxPendingIds = 64;

struct VideoDecoder::Impl {
  Codec codec{};
  Backend backend{};
  bool low_latency{};
  std::string detail;
  std::unique_ptr<detail::DecoderBackend> session;
  std::uint64_t next_id{};
  std::map<std::uint64_t, std::int64_t> pts_of;
  std::uint64_t skipped{};
  // The copies the vector-returning decode() and flush() hand out, and their memory.
  std::vector<PitchedBuffer> pool;

  // Runs `step` with a backend sink that hands each frame to `sink` with the pts of its own
  // access unit. The backends tag access units with ids in decode order; a stream with B-frames
  // shows them in another order. Where the backend gives each frame its own id, that id is looked
  // up; otherwise the frame takes the smallest pts still waiting, as a decoder shows a stream's
  // frames in increasing presentation time.
  void run(const DecodedFrameSink& sink,
           const std::function<void(const detail::FrameSink&)>& step) {
    const bool exact = session->exact_ids();
    const detail::FrameSink backend_sink = [&](const detail::Nv12Frame& frame, std::uint64_t id) {
      auto found = pts_of.find(id);
      if (!exact) {
        found = std::min_element(pts_of.begin(), pts_of.end(),
                                 [](const auto& a, const auto& b) { return a.second < b.second; });
      }
      std::int64_t pts = 0;
      if (found != pts_of.end()) {
        pts = found->second;
        pts_of.erase(found);
      }
      sink(DecodedFrame{.image = DeviceImage{.data = frame.luma,
                                             .pitch = frame.luma_pitch,
                                             .chroma = frame.chroma,
                                             .chroma_pitch = frame.chroma_pitch,
                                             .width = frame.width,
                                             .height = frame.height,
                                             .format = PixelFormat::kNv12},
                        .pts = pts});
    };
    step(backend_sink);
  }

  // A sink that copies each frame into the pool, for the vector-returning calls.
  DecodedFrameSink copy_into(std::vector<DecodedFrame>& frames, cudaStream_t stream) {
    return [this, &frames, stream](const DecodedFrame& frame) {
      if (pool.size() <= frames.size()) {
        pool.emplace_back();
      }
      PitchedBuffer& buffer = pool[frames.size()];
      const auto rows = static_cast<std::size_t>(frame.image.height);
      buffer.reserve(static_cast<std::size_t>(frame.image.width), rows * 3 / 2);
      check_cuda(
          detail::launch_from_nv12(detail::Nv12Frame{.luma = frame.image.data,
                                                     .luma_pitch = frame.image.pitch,
                                                     .chroma = frame.image.chroma,
                                                     .chroma_pitch = frame.image.chroma_pitch,
                                                     .width = frame.image.width,
                                                     .height = frame.image.height},
                                   PixelFormat::kNv12, buffer.data, buffer.pitch, frame.image.width,
                                   frame.image.height, stream),
          "copying a decoded frame");
      DecodedFrame copy = frame;
      copy.image.data = buffer.data;
      copy.image.pitch = buffer.pitch;
      copy.image.chroma = buffer.data + buffer.pitch * rows;
      copy.image.chroma_pitch = buffer.pitch;
      frames.push_back(copy);
    };
  }
};

VideoDecoder::VideoDecoder(Codec codec, Backend backend, bool low_latency)
    : impl_(std::make_unique<Impl>()) {
  if (codec_name(codec) == "unknown" || backend_name(backend) == "unknown") {
    throw std::invalid_argument("unknown codec or backend");
  }
  impl_->codec = codec;
  impl_->low_latency = low_latency;
  std::tie(impl_->backend, impl_->detail) = choose_backend(false, codec, backend);
}

VideoDecoder::~VideoDecoder() = default;

Codec VideoDecoder::codec() const noexcept { return impl_->codec; }
Backend VideoDecoder::backend() const noexcept { return impl_->backend; }
const std::string& VideoDecoder::backend_detail() const noexcept { return impl_->detail; }
std::uint64_t VideoDecoder::skipped() const noexcept { return impl_->skipped; }

void VideoDecoder::decode(std::span<const std::uint8_t> access_unit, std::int64_t pts,
                          cudaStream_t stream, const DecodedFrameSink& sink) {
  Impl& impl = *impl_;
  if (access_unit.empty()) {
    return;
  }
  if (!impl.session) {
    if (!is_keyframe(impl.codec, access_unit)) {
      ++impl.skipped;
      return;
    }
    impl.session = impl.backend == Backend::kNvcodec
                       ? detail::create_nvcodec_decoder(impl.codec, impl.low_latency)
                       : detail::create_v4l2_decoder(impl.codec, impl.low_latency);
  }
  const std::uint64_t id = impl.next_id++;
  impl.pts_of.emplace(id, pts);
  while (impl.pts_of.size() > kMaxPendingIds) {
    impl.pts_of.erase(impl.pts_of.begin());
  }
  impl.run(sink, [&](const detail::FrameSink& backend_sink) {
    impl.session->decode(access_unit, id, stream, backend_sink);
  });
}

void VideoDecoder::flush(cudaStream_t stream, const DecodedFrameSink& sink) {
  Impl& impl = *impl_;
  if (!impl.session) {
    return;
  }
  // The session ends whatever happens: the next access unit must be a keyframe.
  struct End {
    Impl& impl;
    ~End() {
      impl.session.reset();
      impl.pts_of.clear();
    }
  } end{impl};
  impl.run(sink, [&](const detail::FrameSink& backend_sink) {
    impl.session->flush(stream, backend_sink);
  });
}

std::vector<DecodedFrame> VideoDecoder::decode(std::span<const std::uint8_t> access_unit,
                                               std::int64_t pts, cudaStream_t stream) {
  std::vector<DecodedFrame> frames;
  decode(access_unit, pts, stream, impl_->copy_into(frames, stream));
  return frames;
}

std::vector<DecodedFrame> VideoDecoder::flush(cudaStream_t stream) {
  std::vector<DecodedFrame> frames;
  flush(stream, impl_->copy_into(frames, stream));
  return frames;
}

void convert_from_nv12(const DeviceImage& nv12, PixelFormat format, std::uint8_t* out,
                       std::size_t out_pitch, int width, int height, cudaStream_t stream) {
  if (nv12.format != PixelFormat::kNv12 || nv12.data == nullptr || nv12.chroma == nullptr) {
    throw std::invalid_argument("convert_from_nv12 needs an NV12 image");
  }
  if (width < 1 || height < 1 || width > nv12.width || height > nv12.height) {
    throw std::invalid_argument("convert_from_nv12: the output is larger than the image");
  }
  check_cuda(detail::launch_from_nv12(detail::Nv12Frame{.luma = nv12.data,
                                                        .luma_pitch = nv12.pitch,
                                                        .chroma = nv12.chroma,
                                                        .chroma_pitch = nv12.chroma_pitch,
                                                        .width = nv12.width,
                                                        .height = nv12.height},
                                      format, out, out_pitch, width, height, stream),
             "converting from NV12");
}

}  // namespace holoscan::codec
