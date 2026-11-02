// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The v4l2 decoder: the Jetson's NVDEC through /dev/v4l2-nvdec. Its frames come out block-linear,
// in buffers the driver allocates; VPI's VIC backend lays each one out pitch-linear into device
// memory, which is what the sink reads.
//
// VPI is loaded at run time, and privately (RTLD_LOCAL): loaded into the global namespace, its
// libraries make the V4L2 encoder in the same process fail to set up its VIC.

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <linux/videodev2.h>              // NOLINT(build/include_order)
#include <sys/stat.h>                     // NOLINT(build/include_order)
#include <vpi/Image.h>                    // NOLINT(build/include_order)
#include <vpi/Status.h>                   // NOLINT(build/include_order)
#include <vpi/Stream.h>                   // NOLINT(build/include_order)
#include <vpi/algo/ConvertImageFormat.h>  // NOLINT(build/include_order)

#include "backend.hpp"
#include "v4l2_device.hpp"
#include "v4l2_nv_extensions.hpp"

namespace holoscan::codec::detail {
namespace {

using std::chrono_literals::operator""ms;

constexpr v4l2_buf_type kCoded = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;    // Access units in.
constexpr v4l2_buf_type kFrames = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;  // Frames out.
constexpr std::uint32_t kCodedBuffers = 4;
// The largest access unit the decoder takes.
constexpr std::uint32_t kCodedBufferBytes = 8U << 20;
// Frame buffers beyond the decoder's minimum.
constexpr std::uint32_t kExtraFrameBuffers = 2;
// Longest wait for a free input buffer, for the frame of an access unit just queued (longer
// before the stream's first frame), and for the decoder to set up its frame buffers.
constexpr auto kInputTimeout = 2000ms;
constexpr auto kFrameTimeout = 200ms;
constexpr auto kWarmUpTimeout = 500ms;
constexpr auto kSetUpTimeout = 3000ms;

std::uint32_t coded_format(Codec codec) {
  return codec == Codec::kH264 ? V4L2_PIX_FMT_H264 : v4l2_nv::kPixFmtH265;
}

// The VPI 4 functions the decoder uses.
struct Vpi {
  decltype(&vpiStreamCreate) stream_create{};
  decltype(&vpiStreamDestroy) stream_destroy{};
  decltype(&vpiStreamSync) stream_sync{};
  decltype(&vpiImageCreateWrapper) image_create_wrapper{};
  decltype(&vpiImageDestroy) image_destroy{};
  decltype(&vpiImageGetFormat) image_get_format{};
  decltype(&vpiImageGetSize) image_get_size{};
  decltype(&vpiImageFormatSetMemLayout) format_set_mem_layout{};
  decltype(&vpiSubmitConvertImageFormat) submit_convert_image_format{};
  decltype(&vpiGetLastStatusMessage) last_status_message{};
  std::string error;
};

template <typename Function>
bool bind(void* handle, const char* name, Function& target) {
  target = reinterpret_cast<Function>(dlsym(handle, name));
  return target != nullptr;
}

Vpi load_vpi() {
  Vpi vpi;
  void* handle = dlopen("libnvvpi.so.4", RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* message = dlerror();
    vpi.error = std::string("VPI 4: ") + (message != nullptr ? message : "not found");
    return vpi;
  }
  const bool bound = bind(handle, "vpiStreamCreate", vpi.stream_create) &&
                     bind(handle, "vpiStreamDestroy", vpi.stream_destroy) &&
                     bind(handle, "vpiStreamSync", vpi.stream_sync) &&
                     bind(handle, "vpiImageCreateWrapper", vpi.image_create_wrapper) &&
                     bind(handle, "vpiImageDestroy", vpi.image_destroy) &&
                     bind(handle, "vpiImageGetFormat", vpi.image_get_format) &&
                     bind(handle, "vpiImageGetSize", vpi.image_get_size) &&
                     bind(handle, "vpiImageFormatSetMemLayout", vpi.format_set_mem_layout) &&
                     bind(handle, "vpiSubmitConvertImageFormat", vpi.submit_convert_image_format) &&
                     bind(handle, "vpiGetLastStatusMessage", vpi.last_status_message);
  if (!bound) {
    vpi.error = "libnvvpi.so.4 lacks the functions the v4l2 decoder needs";
  }
  return vpi;
}

const Vpi& vpi() {
  static const Vpi loaded = load_vpi();
  return loaded;
}

void check_vpi(VPIStatus status, const char* what) {
  if (status == VPI_SUCCESS) {
    return;
  }
  char message[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
  vpi().last_status_message(message, sizeof(message));
  throw std::runtime_error(std::string("v4l2 decoder: ") + what + ": " + message);
}

class V4l2Decoder final : public DecoderBackend {
 public:
  V4l2Decoder(Codec codec, bool low_latency)
      : device_(kV4l2DecoderDevice), low_latency_(low_latency) {
    if (!vpi().error.empty()) {
      throw std::runtime_error("v4l2 decoder: " + vpi().error);
    }
    v4l2_format coded{};
    coded.type = kCoded;
    coded.fmt.pix_mp.pixelformat = coded_format(codec);
    coded.fmt.pix_mp.num_planes = 1;
    coded.fmt.pix_mp.plane_fmt[0].sizeimage = kCodedBufferBytes;
    device_.check_ioctl(VIDIOC_S_FMT, &coded, "v4l2 decoder: setting the stream format");
    // Each input buffer holds one whole access unit, so the decoder need not wait for the next
    // one to find where it ends.
    device_.set_control(v4l2_nv::kCidDisableCompleteFrameInput, 0,
                        "v4l2 decoder: taking whole frames");
    if (low_latency) {
      // Otherwise the decoder holds frames back for reordering the stream does not need.
      device_.set_control(v4l2_nv::kCidDisableDpb, 1, "v4l2 decoder: turning frame reordering off");
    }
    v4l2_event_subscription event{};
    event.type = v4l2_nv::kEventResolutionChange;
    device_.check_ioctl(VIDIOC_SUBSCRIBE_EVENT, &event,
                        "v4l2 decoder: subscribing to resolution changes");
    coded_buffers_ = device_.request_buffers(kCoded, kCodedBuffers, 1);
    device_.stream_on(kCoded);
    check_vpi(vpi().stream_create(VPI_BACKEND_VIC, &vpi_stream_), "vpiStreamCreate");
  }

  ~V4l2Decoder() override {
    device_.stream_off(kCoded);
    stop_frames();
    device_.release_buffers(kCoded, coded_buffers_);
    if (vpi_stream_ != nullptr) {
      vpi().stream_sync(vpi_stream_);
    }
    if (output_image_ != nullptr) {
      vpi().image_destroy(output_image_);
    }
    if (vpi_stream_ != nullptr) {
      vpi().stream_destroy(vpi_stream_);
    }
    cudaFree(output_);
  }

  void decode(std::span<const std::uint8_t> access_unit, std::uint64_t id, cudaStream_t stream,
              const FrameSink& sink) override {
    if (access_unit.size() > coded_buffers_.front().planes[0].length) {
      throw std::runtime_error("v4l2 decoder: an access unit of " +
                               std::to_string(access_unit.size()) + " bytes exceeds the " +
                               std::to_string(coded_buffers_.front().planes[0].length) +
                               "-byte input buffers");
    }
    const std::uint32_t index = free_coded_buffer(stream, sink);
    std::memcpy(coded_buffers_[index].planes[0].data, access_unit.data(), access_unit.size());
    device_.queue(kCoded, index, {static_cast<std::uint32_t>(access_unit.size())}, id);
    coded_buffers_[index].queued = true;
    // Low latency: the frame follows within a few milliseconds, so wait for it. Otherwise the
    // decoder holds frames back, and hands them out with later access units or at flush().
    // At the start of a stream the decoder first sets up its frame buffers, and gives its first
    // frames one or two access units late; after that, each access unit's frame follows within
    // a few milliseconds. Low latency waits for this access unit's own frame.
    if (frame_buffers_.empty()) {
      wait_until([&] { return collect(stream, sink) && !frame_buffers_.empty(); }, kSetUpTimeout);
      return;
    }
    const bool caught_up = wait_until(
        [&] {
          return collect(stream, sink) &&
                 (!low_latency_ || lagging_ || (shown_ && last_shown_ >= id));
        },
        shown_ ? kFrameTimeout : kWarmUpTimeout);
    if (low_latency_ && shown_ && !caught_up) {
      // The stream holds frames back after all: stop waiting for them, they come later.
      lagging_ = true;
    }
  }

  void flush(cudaStream_t stream, const FrameSink& sink) override {
    // An empty input buffer ends the stream; the decoder then returns every frame it holds, and
    // an empty frame buffer after the last one.
    const std::uint32_t index = free_coded_buffer(stream, sink);
    device_.queue(kCoded, index, {0}, 0);
    coded_buffers_[index].queued = true;
    bool ended = false;
    wait_until([&] { return collect(stream, sink, &ended) && ended; }, kInputTimeout);
  }

 private:
  // A free input buffer, waiting (and handing out finished frames) until one comes back.
  std::uint32_t free_coded_buffer(cudaStream_t stream, const FrameSink& sink) {
    std::uint32_t index = 0;
    const bool found = wait_until(
        [&] {
          V4l2Device::Dequeued done;
          while (device_.dequeue(kCoded, 1, done)) {
            coded_buffers_[done.index].queued = false;
          }
          collect(stream, sink);
          for (index = 0; index < coded_buffers_.size(); ++index) {
            if (!coded_buffers_[index].queued) return true;
          }
          return false;
        },
        kInputTimeout);
    if (!found) {
      throw std::runtime_error("v4l2 decoder: no input buffer came back within 2 s");
    }
    return index;
  }

  // Handles a pending resolution change and hands every finished frame to `sink`. Returns true so
  // that it can serve as a wait_until() condition together with its flags.
  bool collect(cudaStream_t stream, const FrameSink& sink, bool* ended = nullptr) {
    v4l2_event event{};
    while (device_.ioctl(VIDIOC_DQEVENT, &event) == 0) {
      if (event.type == v4l2_nv::kEventResolutionChange) {
        start_frames();
      }
    }
    if (frame_buffers_.empty()) {
      return true;
    }
    V4l2Device::Dequeued done;
    while (device_.dequeue(kFrames, frame_planes_, done)) {
      if (done.used[0] == 0) {
        if (ended != nullptr) *ended = true;
        continue;
      }
      deliver(frame_buffers_[done.index], done.timestamp_us, stream, sink);
      device_.queue(kFrames, done.index, std::vector<std::uint32_t>(frame_planes_, 0), 0);
      shown_ = true;
      last_shown_ = done.timestamp_us;
    }
    return true;
  }

  // (Re)allocates the frame buffers at the stream's new size.
  void start_frames() {
    stop_frames();
    v4l2_format format{};
    format.type = kFrames;
    device_.check_ioctl(VIDIOC_G_FMT, &format, "v4l2 decoder: reading the frame format");
    frame_planes_ = format.fmt.pix_mp.num_planes;
    v4l2_control minimum{};
    minimum.id = V4L2_CID_MIN_BUFFERS_FOR_CAPTURE;
    device_.check_ioctl(VIDIOC_G_CTRL, &minimum, "v4l2 decoder: reading the frame buffer count");
    frame_buffers_ = device_.request_buffers(
        kFrames, static_cast<std::uint32_t>(minimum.value) + kExtraFrameBuffers, frame_planes_);
    device_.stream_on(kFrames);
    for (std::uint32_t i = 0; i < frame_buffers_.size(); ++i) {
      device_.queue(kFrames, i, std::vector<std::uint32_t>(frame_planes_, 0), 0);
    }
  }

  void stop_frames() {
    if (frame_buffers_.empty()) {
      return;
    }
    device_.stream_off(kFrames);
    device_.release_buffers(kFrames, frame_buffers_);
  }

  // Lays one decoded frame out pitch-linear in device memory and hands it to `sink`.
  void deliver(const V4l2Buffer& buffer, std::uint64_t id, cudaStream_t stream,
               const FrameSink& sink) {
    // A wrapper sees the buffer's contents as they were when it was made, so each frame needs a
    // new one.
    VPIImageData data{};
    data.bufferType = VPI_IMAGE_BUFFER_NVBUFFER;
    data.buffer.fd = buffer.planes[0].dmabuf;
    VPIImage input = nullptr;
    check_vpi(vpi().image_create_wrapper(&data, nullptr, VPI_BACKEND_VIC, &input),
              "wrapping a decoded frame");
    struct Destroy {
      VPIImage image;
      ~Destroy() { vpi().image_destroy(image); }
    } destroy{input};
    std::int32_t width = 0;
    std::int32_t height = 0;
    VPIImageFormat format = 0;
    check_vpi(vpi().image_get_size(input, &width, &height), "vpiImageGetSize");
    check_vpi(vpi().image_get_format(input, &format), "vpiImageGetFormat");
    ensure_output(width, height, format);
    check_vpi(vpi().submit_convert_image_format(vpi_stream_, VPI_BACKEND_VIC, input, output_image_,
                                                nullptr),
              "converting a decoded frame");
    check_vpi(vpi().stream_sync(vpi_stream_), "vpiStreamSync");
    const Nv12Frame frame{
        .luma = output_,
        .luma_pitch = output_pitch_,
        .chroma = output_ + output_pitch_ * static_cast<std::size_t>(height),
        .chroma_pitch = output_pitch_,
        .width = width,
        .height = height,
    };
    sink(frame, id);
    // The next frame is converted into the same memory.
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }

  // The pitch-linear NV12 image VIC writes into, (re)made for a new size or format.
  void ensure_output(int width, int height, VPIImageFormat block_linear) {
    if (output_image_ != nullptr && width == output_width_ && height == output_height_ &&
        block_linear == output_source_format_) {
      return;
    }
    if (output_image_ != nullptr) {
      vpi().image_destroy(output_image_);
      output_image_ = nullptr;
    }
    cudaFree(output_);
    output_ = nullptr;
    void* memory = nullptr;
    check_cuda(cudaMallocPitch(&memory, &output_pitch_, static_cast<std::size_t>(width),
                               static_cast<std::size_t>(height + height / 2)),
               "v4l2 decoder: cudaMallocPitch");
    output_ = static_cast<std::uint8_t*>(memory);
    VPIImageData data{};
    data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    VPIImageBufferPitchLinear& pitch = data.buffer.pitch;
    pitch.format = vpi().format_set_mem_layout(block_linear, VPI_MEM_LAYOUT_PITCH_LINEAR);
    pitch.numPlanes = 2;
    pitch.planes[0].pixelType = VPI_PIXEL_TYPE_U8;
    pitch.planes[0].width = width;
    pitch.planes[0].height = height;
    pitch.planes[0].pitchBytes = static_cast<std::int32_t>(output_pitch_);
    pitch.planes[0].pBase = output_;
    pitch.planes[1].pixelType = VPI_PIXEL_TYPE_2U8;
    pitch.planes[1].width = width / 2;
    pitch.planes[1].height = height / 2;
    pitch.planes[1].pitchBytes = static_cast<std::int32_t>(output_pitch_);
    pitch.planes[1].pBase = output_ + output_pitch_ * static_cast<std::size_t>(height);
    check_vpi(vpi().image_create_wrapper(&data, nullptr, VPI_BACKEND_VIC, &output_image_),
              "wrapping the output frame");
    output_width_ = width;
    output_height_ = height;
    output_source_format_ = block_linear;
  }

  V4l2Device device_;
  bool low_latency_;
  // Whether a frame has come out of this stream yet, and the id of the latest one.
  bool shown_{};
  std::uint64_t last_shown_{};
  // Whether low latency gave up waiting for frames the stream holds back.
  bool lagging_{};
  std::vector<V4l2Buffer> coded_buffers_;
  std::vector<V4l2Buffer> frame_buffers_;
  std::uint32_t frame_planes_{2};
  VPIStream vpi_stream_{};
  VPIImage output_image_{};
  std::uint8_t* output_{};
  std::size_t output_pitch_{};
  int output_width_{};
  int output_height_{};
  VPIImageFormat output_source_format_{};
};

bool device_exists(const char* path) {
  struct stat info{};
  return ::stat(path, &info) == 0;
}

}  // namespace

Probe probe_v4l2_decoder(Codec codec) {
  if (!device_exists(kV4l2DecoderDevice)) {
    return {.available = false, .detail = std::string("no ") + kV4l2DecoderDevice};
  }
  if (!v4l2_library_error().empty()) {
    return {.available = false, .detail = v4l2_library_error()};
  }
  if (!vpi().error.empty()) {
    return {.available = false, .detail = vpi().error};
  }
  try {
    const V4l2Device device(kV4l2DecoderDevice);
    const auto formats = device.formats(kCoded);
    if (std::find(formats.begin(), formats.end(), coded_format(codec)) == formats.end()) {
      return {
          .available = false,
          .detail = std::string(kV4l2DecoderDevice) + " has no " + std::string(codec_name(codec))};
    }
    return {.available = true,
            .detail = std::string("NVDEC through ") + kV4l2DecoderDevice + ", VIC through VPI"};
  } catch (const std::exception& error) {
    return {.available = false, .detail = error.what()};
  }
}

std::unique_ptr<DecoderBackend> create_v4l2_decoder(Codec codec, bool low_latency) {
  return std::make_unique<V4l2Decoder>(codec, low_latency);
}

}  // namespace holoscan::codec::detail
