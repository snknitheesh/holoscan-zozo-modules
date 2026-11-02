// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The v4l2 encoder: the Jetson's NVENC through /dev/v4l2-nvenc. Its input buffers are pitch-linear
// NV12 the driver allocates, so each frame is copied into one from device memory; the encoder's
// own engine (VIC) then lays it out for NVENC. One frame in, one access unit out.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <linux/dma-buf.h>    // NOLINT(build/include_order)
#include <linux/videodev2.h>  // NOLINT(build/include_order)
#include <sys/ioctl.h>        // NOLINT(build/include_order)
#include <sys/stat.h>         // NOLINT(build/include_order)

#include "backend.hpp"
#include "v4l2_device.hpp"
#include "v4l2_nv_extensions.hpp"

namespace holoscan::codec::detail {
namespace {

using std::chrono_literals::operator""ms;

constexpr v4l2_buf_type kRaw = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;     // Frames into the device.
constexpr v4l2_buf_type kCoded = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;  // Access units out.
constexpr std::uint32_t kRawBuffers = 2;
constexpr std::uint32_t kCodedBuffers = 2;
// Longest wait for one frame's access unit.
constexpr auto kEncodeTimeout = 2000ms;

std::uint32_t coded_format(Codec codec) {
  return codec == Codec::kH264 ? V4L2_PIX_FMT_H264 : v4l2_nv::kPixFmtH265;
}

// Brackets CPU writes to a dma-buf so the device sees them.
void sync_dmabuf(int dmabuf, std::uint64_t flags) {
  dma_buf_sync sync{};
  sync.flags = flags | DMA_BUF_SYNC_WRITE;
  ::ioctl(dmabuf, DMA_BUF_IOCTL_SYNC, &sync);
}

class V4l2Encoder final : public EncoderBackend {
 public:
  V4l2Encoder(const EncoderSettings& settings, int width, int height)
      : device_(kV4l2EncoderDevice),
        width_(width),
        height_(height),
        frame_rate_(settings.frame_rate) {
    v4l2_format coded{};
    coded.type = kCoded;
    coded.fmt.pix_mp.width = static_cast<std::uint32_t>(width);
    coded.fmt.pix_mp.height = static_cast<std::uint32_t>(height);
    coded.fmt.pix_mp.pixelformat = coded_format(settings.codec);
    coded.fmt.pix_mp.num_planes = 1;
    coded.fmt.pix_mp.plane_fmt[0].sizeimage =
        std::max<std::uint32_t>(static_cast<std::uint32_t>(width * height * 3 / 2), 1U << 20);
    device_.check_ioctl(VIDIOC_S_FMT, &coded, "v4l2 encoder: setting the stream format");

    v4l2_format raw{};
    raw.type = kRaw;
    raw.fmt.pix_mp.width = static_cast<std::uint32_t>(width);
    raw.fmt.pix_mp.height = static_cast<std::uint32_t>(height);
    raw.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12M;
    raw.fmt.pix_mp.num_planes = 2;
    device_.check_ioctl(VIDIOC_S_FMT, &raw, "v4l2 encoder: setting the frame format");
    device_.check_ioctl(VIDIOC_G_FMT, &raw, "v4l2 encoder: reading the frame format");
    // The device may round its buffers up (642x362 to 656x368); the frame fills their top-left,
    // and the stream keeps the size set on the coded side.
    if (raw.fmt.pix_mp.width < static_cast<std::uint32_t>(width) ||
        raw.fmt.pix_mp.height < static_cast<std::uint32_t>(height)) {
      throw std::invalid_argument("v4l2 encoder: the device cannot encode " +
                                  std::to_string(width) + "x" + std::to_string(height) +
                                  " (it offers " + std::to_string(raw.fmt.pix_mp.width) + "x" +
                                  std::to_string(raw.fmt.pix_mp.height) + ")");
    }
    luma_pitch_ = raw.fmt.pix_mp.plane_fmt[0].bytesperline;
    chroma_pitch_ = raw.fmt.pix_mp.plane_fmt[1].bytesperline;

    configure(settings);

    v4l2_streamparm rate{};
    rate.type = kRaw;
    rate.parm.output.timeperframe.numerator = 1000;  // codespell:ignore parm
    rate.parm.output.timeperframe.denominator =      // codespell:ignore parm
        static_cast<std::uint32_t>(settings.frame_rate * 1000.0 + 0.5);
    device_.check_ioctl(VIDIOC_S_PARM, &rate, "v4l2 encoder: setting the frame rate");

    raw_buffers_ = device_.request_buffers(kRaw, kRawBuffers, 2);
    // Pinned for CUDA, the frame copies into these buffers are DMA transfers instead of CPU
    // copies through a staging buffer. Where the memory cannot be pinned, they stay CPU copies.
    for (const V4l2Buffer& buffer : raw_buffers_) {
      for (const V4l2Buffer::Plane& plane : buffer.planes) {
        if (cudaHostRegister(plane.data, plane.length, cudaHostRegisterDefault) == cudaSuccess) {
          pinned_.push_back(plane.data);
        } else {
          (void)cudaGetLastError();
        }
      }
    }
    coded_buffers_ = device_.request_buffers(kCoded, kCodedBuffers, 1);
    device_.stream_on(kRaw);
    device_.stream_on(kCoded);
    streaming_ = true;
    for (std::uint32_t i = 0; i < coded_buffers_.size(); ++i) {
      device_.queue(kCoded, i, {0}, 0);
    }
  }

  ~V4l2Encoder() override {
    for (void* pointer : pinned_) {
      cudaHostUnregister(pointer);
    }
    if (streaming_) {
      device_.stream_off(kRaw);
      device_.stream_off(kCoded);
    }
    device_.release_buffers(kRaw, raw_buffers_);
    device_.release_buffers(kCoded, coded_buffers_);
  }

  std::vector<std::uint8_t> encode(const Nv12Frame& frame, bool /*keyframe*/) override {
    // Keyframes: the device's IDR interval is the GOP and every stream starts at frame 0, so it
    // places them where VideoEncoder asks for them.
    if (frame.width != width_ || frame.height != height_) {
      throw std::invalid_argument("v4l2 encoder: a frame of the configured size is needed");
    }
    reclaim_raw_buffers();
    const auto free_buffer = std::find_if(raw_buffers_.begin(), raw_buffers_.end(),
                                          [](const V4l2Buffer& b) { return !b.queued; });
    if (free_buffer == raw_buffers_.end()) {
      throw std::runtime_error("v4l2 encoder: no input buffer came back");
    }
    V4l2Buffer& buffer = *free_buffer;
    const auto index = static_cast<std::uint32_t>(free_buffer - raw_buffers_.begin());
    for (const V4l2Buffer::Plane& plane : buffer.planes) {
      sync_dmabuf(plane.dmabuf, DMA_BUF_SYNC_START);
    }
    const cudaError_t luma =
        cudaMemcpy2D(buffer.planes[0].data, luma_pitch_, frame.luma, frame.luma_pitch,
                     static_cast<std::size_t>(width_), static_cast<std::size_t>(height_),
                     cudaMemcpyDeviceToHost);
    const cudaError_t chroma =
        cudaMemcpy2D(buffer.planes[1].data, chroma_pitch_, frame.chroma, frame.chroma_pitch,
                     static_cast<std::size_t>(width_), static_cast<std::size_t>(height_ / 2),
                     cudaMemcpyDeviceToHost);
    for (const V4l2Buffer::Plane& plane : buffer.planes) {
      sync_dmabuf(plane.dmabuf, DMA_BUF_SYNC_END);
    }
    check_cuda(luma != cudaSuccess ? luma : chroma, "v4l2 encoder: copying the frame");
    device_.queue(
        kRaw, index,
        {static_cast<std::uint32_t>(luma_pitch_ * static_cast<std::size_t>(height_)),
         static_cast<std::uint32_t>(chroma_pitch_ * static_cast<std::size_t>(height_ / 2))},
        timestamp_us());
    ++frames_;
    buffer.queued = true;

    V4l2Device::Dequeued done;
    if (!wait_until([&] { return device_.dequeue(kCoded, 1, done); }, kEncodeTimeout)) {
      throw std::runtime_error("v4l2 encoder: no access unit within 2 s");
    }
    const std::uint8_t* bytes = coded_buffers_[done.index].planes[0].data;
    std::vector<std::uint8_t> access_unit(bytes, bytes + done.used[0]);
    device_.queue(kCoded, done.index, {0}, 0);
    return access_unit;
  }

 private:
  void configure(const EncoderSettings& settings) {
    if (settings.codec == Codec::kH264) {
      device_.set_control(V4L2_CID_MPEG_VIDEO_H264_PROFILE, V4L2_MPEG_VIDEO_H264_PROFILE_HIGH,
                          "v4l2 encoder: setting the H.264 profile");
    } else {
      device_.set_control(v4l2_nv::kCidH265Profile, v4l2_nv::kH265ProfileMain,
                          "v4l2 encoder: setting the H.265 profile");
    }
    switch (settings.rate_control) {
      case RateControl::kConstQp: {
        // With rate control off, every frame takes the initial QP.
        const auto qp = static_cast<std::uint32_t>(settings.qp);
        v4l2_nv::InitFrameQp init{qp, qp, qp, {}};
        device_.set_control(v4l2_nv::kCidInitFrameQp, &init, sizeof(init),
                            "v4l2 encoder: setting the QP");
        device_.set_control(V4L2_CID_MPEG_VIDEO_FRAME_RC_ENABLE, 0,
                            "v4l2 encoder: turning rate control off");
        break;
      }
      case RateControl::kCbr:
        device_.set_control(V4L2_CID_MPEG_VIDEO_BITRATE_MODE, V4L2_MPEG_VIDEO_BITRATE_MODE_CBR,
                            "v4l2 encoder: setting CBR");
        device_.set_control(V4L2_CID_MPEG_VIDEO_BITRATE,
                            static_cast<std::int32_t>(settings.bitrate),
                            "v4l2 encoder: setting the bitrate");
        break;
      case RateControl::kVbr:
        device_.set_control(V4L2_CID_MPEG_VIDEO_BITRATE_MODE, V4L2_MPEG_VIDEO_BITRATE_MODE_VBR,
                            "v4l2 encoder: setting VBR");
        device_.set_control(V4L2_CID_MPEG_VIDEO_BITRATE,
                            static_cast<std::int32_t>(settings.bitrate),
                            "v4l2 encoder: setting the bitrate");
        device_.set_control(
            V4L2_CID_MPEG_VIDEO_BITRATE_PEAK,
            static_cast<std::int32_t>(settings.max_bitrate != 0 ? settings.max_bitrate
                                                                : 2U * settings.bitrate),
            "v4l2 encoder: setting the peak bitrate");
        break;
    }
    const auto gop = static_cast<std::int32_t>(settings.gop);
    device_.set_control(V4L2_CID_MPEG_VIDEO_GOP_SIZE, gop, "v4l2 encoder: setting the GOP");
    device_.set_control(v4l2_nv::kCidIdrInterval, gop, "v4l2 encoder: setting the IDR interval");
    device_.set_control(v4l2_nv::kCidInsertSpsPpsAtIdr, 1,
                        "v4l2 encoder: repeating the parameter sets");
    device_.set_control(v4l2_nv::kCidNumBFrames, 0, "v4l2 encoder: turning B-frames off");
    if (settings.low_latency) {
      // One reference frame: a low-latency decoder can show each frame at once.
      std::uint32_t frames = 1;  // Passed by pointer.
      device_.set_control(v4l2_nv::kCidNumReferenceFrames, &frames, sizeof(frames),
                          "v4l2 encoder: setting one reference frame");
    }
    std::int32_t preset = v4l2_nv::kHwPresetUltraFast;
    if (settings.preset == Preset::kMedium) preset = v4l2_nv::kHwPresetMedium;
    if (settings.preset == Preset::kQuality) preset = v4l2_nv::kHwPresetSlow;
    device_.set_control(v4l2_nv::kCidHwPresetType, preset, "v4l2 encoder: setting the preset");
    if (settings.low_latency) {
      device_.set_control(v4l2_nv::kCidMaxPerformance, 1,
                          "v4l2 encoder: setting maximum performance");
    }
  }

  // The frame's time at the configured rate: the device derives the frame rate from it.
  [[nodiscard]] std::uint64_t timestamp_us() const {
    return static_cast<std::uint64_t>(static_cast<double>(frames_) * 1e6 / frame_rate_);
  }

  // Marks the input buffers the device has finished with as free.
  void reclaim_raw_buffers() {
    V4l2Device::Dequeued done;
    const auto any_free = [&] {
      while (device_.dequeue(kRaw, 2, done)) {
        raw_buffers_[done.index].queued = false;
      }
      return std::any_of(raw_buffers_.begin(), raw_buffers_.end(),
                         [](const V4l2Buffer& b) { return !b.queued; });
    };
    wait_until(any_free, kEncodeTimeout);
  }

  V4l2Device device_;
  int width_;
  int height_;
  std::size_t luma_pitch_{};
  std::size_t chroma_pitch_{};
  std::vector<V4l2Buffer> raw_buffers_;
  std::vector<V4l2Buffer> coded_buffers_;
  std::vector<void*> pinned_;
  bool streaming_{};
  double frame_rate_;
  std::uint64_t frames_{};
};

bool device_exists(const char* path) {
  struct stat info{};
  return ::stat(path, &info) == 0;
}

}  // namespace

Probe probe_v4l2_encoder(Codec codec) {
  if (!device_exists(kV4l2EncoderDevice)) {
    return {.available = false, .detail = std::string("no ") + kV4l2EncoderDevice};
  }
  if (!v4l2_library_error().empty()) {
    return {.available = false, .detail = v4l2_library_error()};
  }
  try {
    const V4l2Device device(kV4l2EncoderDevice);
    const auto formats = device.formats(kCoded);
    if (std::find(formats.begin(), formats.end(), coded_format(codec)) == formats.end()) {
      return {
          .available = false,
          .detail = std::string(kV4l2EncoderDevice) + " has no " + std::string(codec_name(codec))};
    }
    return {.available = true, .detail = std::string("NVENC through ") + kV4l2EncoderDevice};
  } catch (const std::exception& error) {
    return {.available = false, .detail = error.what()};
  }
}

std::unique_ptr<EncoderBackend> create_v4l2_encoder(const EncoderSettings& settings, int width,
                                                    int height) {
  return std::make_unique<V4l2Encoder>(settings, width, height);
}

}  // namespace holoscan::codec::detail
