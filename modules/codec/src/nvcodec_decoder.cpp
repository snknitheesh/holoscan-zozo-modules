// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The nvcodec decoder: NVDEC through the Video Codec SDK's parser and decoder. Each frame is
// mapped in device memory, handed to the sink, and unmapped once the sink's work on the stream is
// done. The parser shows each frame as soon as it is decoded (no display delay).

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "backend.hpp"
#include "nvcodec_api.hpp"

namespace holoscan::codec::detail {
namespace {

cudaVideoCodec cuvid_codec(Codec codec) {
  return codec == Codec::kH264 ? cudaVideoCodec_H264 : cudaVideoCodec_HEVC;
}

// Makes a context current on this thread for a scope.
class ContextScope {
 public:
  explicit ContextScope(CUcontext context) { check_cu(cuCtxPushCurrent(context), "cuCtxPush"); }
  ~ContextScope() {
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
  }
  ContextScope(const ContextScope&) = delete;
  ContextScope& operator=(const ContextScope&) = delete;
};

// Decode surfaces beyond the stream's minimum, so decoding is not held up by mapped frames.
constexpr unsigned kExtraDecodeSurfaces = 4;

class NvcodecDecoder final : public DecoderBackend {
 public:
  NvcodecDecoder(Codec codec, bool low_latency) : codec_(codec) {
    const auto& library = cuvid_library();
    if (!library.error.empty()) {
      throw std::runtime_error("nvcodec decoder: " + library.error);
    }
    context_ = runtime_context().context;
    CUVIDPARSERPARAMS params{};
    params.CodecType = cuvid_codec(codec);
    params.ulMaxNumDecodeSurfaces = 1;  // Raised by the sequence callback.
    // Low latency: show each frame as soon as it is decoded; otherwise let decoding run ahead.
    params.ulMaxDisplayDelay = low_latency ? 0 : 2;
    params.pUserData = this;
    params.pfnSequenceCallback = &NvcodecDecoder::on_sequence;
    params.pfnDecodePicture = &NvcodecDecoder::on_decode;
    params.pfnDisplayPicture = &NvcodecDecoder::on_display;
    const ContextScope scope(context_);
    check_cu(library.create_video_parser(&parser_, &params), "cuvidCreateVideoParser");
  }

  ~NvcodecDecoder() override {
    const auto& library = cuvid_library();
    const ContextScope scope(context_);
    if (parser_ != nullptr) {
      library.destroy_video_parser(parser_);
    }
    if (decoder_ != nullptr) {
      library.destroy_decoder(decoder_);
    }
  }

  void decode(std::span<const std::uint8_t> access_unit, std::uint64_t id, cudaStream_t stream,
              const FrameSink& sink) override {
    CUVIDSOURCEDATAPACKET packet{};
    packet.flags = CUVID_PKT_TIMESTAMP | CUVID_PKT_ENDOFPICTURE;
    packet.payload_size = static_cast<unsigned long>(access_unit.size());  // NOLINT(runtime/int)
    packet.payload = access_unit.data();
    packet.timestamp = static_cast<CUvideotimestamp>(id);
    parse(packet, stream, sink);
  }

  void flush(cudaStream_t stream, const FrameSink& sink) override {
    CUVIDSOURCEDATAPACKET packet{};
    packet.flags = CUVID_PKT_ENDOFSTREAM;
    parse(packet, stream, sink);
  }

 private:
  void parse(CUVIDSOURCEDATAPACKET& packet, cudaStream_t stream, const FrameSink& sink) {
    const ContextScope scope(context_);
    stream_ = stream;
    sink_ = &sink;
    error_.clear();
    const CUresult status = cuvid_library().parse_video_data(parser_, &packet);
    sink_ = nullptr;
    if (!error_.empty()) {
      throw std::runtime_error("nvcodec decoder: " + error_);
    }
    check_cu(status, "cuvidParseVideoData");
  }

  // Builds the decoder for a new sequence; returns the decode surfaces the parser should cycle.
  int sequence(const CUVIDEOFORMAT& format) {
    const auto& library = cuvid_library();
    if (format.chroma_format != cudaVideoChromaFormat_420 || format.bit_depth_luma_minus8 != 0) {
      error_ = "only 8-bit 4:2:0 streams are decoded";
      return 0;
    }
    const unsigned surfaces = format.min_num_decode_surfaces + kExtraDecodeSurfaces;
    const int display_width = format.display_area.right - format.display_area.left;
    const int display_height = format.display_area.bottom - format.display_area.top;
    if (decoder_ != nullptr && format.coded_width == coded_width_ &&
        format.coded_height == coded_height_ && display_width == width_ &&
        display_height == height_) {
      return static_cast<int>(surfaces);
    }
    CUVIDDECODECAPS caps{};
    caps.eCodecType = format.codec;
    caps.eChromaFormat = format.chroma_format;
    caps.nBitDepthMinus8 = format.bit_depth_luma_minus8;
    if (library.get_decoder_caps(&caps) != CUDA_SUCCESS || caps.bIsSupported == 0) {
      error_ = "this GPU's NVDEC has no " + std::string(codec_name(codec_));
      return 0;
    }
    if (format.coded_width > caps.nMaxWidth || format.coded_height > caps.nMaxHeight ||
        format.coded_width < caps.nMinWidth || format.coded_height < caps.nMinHeight) {
      error_ = std::to_string(format.coded_width) + "x" + std::to_string(format.coded_height) +
               " is outside what this NVDEC decodes, " + std::to_string(caps.nMinWidth) + "x" +
               std::to_string(caps.nMinHeight) + " to " + std::to_string(caps.nMaxWidth) + "x" +
               std::to_string(caps.nMaxHeight);
      return 0;
    }
    if (decoder_ != nullptr) {
      library.destroy_decoder(decoder_);
      decoder_ = nullptr;
    }
    CUVIDDECODECREATEINFO info{};
    info.ulWidth = format.coded_width;
    info.ulHeight = format.coded_height;
    info.ulMaxWidth = format.coded_width;
    info.ulMaxHeight = format.coded_height;
    info.ulNumDecodeSurfaces = surfaces;
    info.CodecType = format.codec;
    info.ChromaFormat = format.chroma_format;
    info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    info.bitDepthMinus8 = format.bit_depth_luma_minus8;
    info.OutputFormat = cudaVideoSurfaceFormat_NV12;
    info.DeinterlaceMode = format.progressive_sequence != 0 ? cudaVideoDeinterlaceMode_Weave
                                                            : cudaVideoDeinterlaceMode_Adaptive;
    info.display_area.left = static_cast<short>(format.display_area.left);      // NOLINT
    info.display_area.top = static_cast<short>(format.display_area.top);        // NOLINT
    info.display_area.right = static_cast<short>(format.display_area.right);    // NOLINT
    info.display_area.bottom = static_cast<short>(format.display_area.bottom);  // NOLINT
    info.ulTargetWidth = static_cast<unsigned>(display_width);
    info.ulTargetHeight = static_cast<unsigned>(display_height);
    info.ulNumOutputSurfaces = 2;
    if (const CUresult status = library.create_decoder(&decoder_, &info); status != CUDA_SUCCESS) {
      decoder_ = nullptr;
      const char* name = nullptr;
      cuGetErrorName(status, &name);
      error_ = std::string("cuvidCreateDecoder: ") + (name != nullptr ? name : "failed");
      return 0;
    }
    coded_width_ = format.coded_width;
    coded_height_ = format.coded_height;
    width_ = display_width;
    height_ = display_height;
    return static_cast<int>(surfaces);
  }

  int decode_picture(CUVIDPICPARAMS* picture) {
    if (decoder_ == nullptr) {
      error_ = "a picture came before its sequence";
      return 0;
    }
    if (const CUresult status = cuvid_library().decode_picture(decoder_, picture);
        status != CUDA_SUCCESS) {
      const char* name = nullptr;
      cuGetErrorName(status, &name);
      error_ = std::string("cuvidDecodePicture: ") + (name != nullptr ? name : "failed");
      return 0;
    }
    return 1;
  }

  int display(const CUVIDPARSERDISPINFO* shown) {
    if (shown == nullptr) {
      return 1;  // End of stream.
    }
    const auto& library = cuvid_library();
    CUVIDPROCPARAMS processing{};
    processing.progressive_frame = shown->progressive_frame;
    processing.second_field = shown->repeat_first_field + 1;
    processing.top_field_first = shown->top_field_first;
    processing.unpaired_field = shown->repeat_first_field < 0 ? 1 : 0;
    processing.output_stream = reinterpret_cast<CUstream>(stream_);
    unsigned long long pointer = 0;  // NOLINT(runtime/int)
    unsigned int pitch = 0;
    if (const CUresult status =
            library.map_video_frame(decoder_, shown->picture_index, &pointer, &pitch, &processing);
        status != CUDA_SUCCESS) {
      const char* name = nullptr;
      cuGetErrorName(status, &name);
      error_ = std::string("cuvidMapVideoFrame: ") + (name != nullptr ? name : "failed");
      return 0;
    }
    const auto* luma = reinterpret_cast<const std::uint8_t*>(pointer);
    // The chroma plane follows the luma plane's (even) rows at the same pitch.
    const Nv12Frame frame{
        .luma = luma,
        .luma_pitch = pitch,
        .chroma = luma + static_cast<std::size_t>(pitch) * ((height_ + 1) & ~1),
        .chroma_pitch = pitch,
        .width = width_,
        .height = height_,
    };
    try {
      (*sink_)(frame, static_cast<std::uint64_t>(shown->timestamp));
      // The frame is unmapped next, so whatever the sink queued on the stream must finish first.
      check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    } catch (const std::exception& error) {
      error_ = error.what();
    }
    library.unmap_video_frame(decoder_, pointer);
    return error_.empty() ? 1 : 0;
  }

  static int CUDAAPI on_sequence(void* self, CUVIDEOFORMAT* format) {
    return static_cast<NvcodecDecoder*>(self)->sequence(*format);
  }
  static int CUDAAPI on_decode(void* self, CUVIDPICPARAMS* picture) {
    return static_cast<NvcodecDecoder*>(self)->decode_picture(picture);
  }
  static int CUDAAPI on_display(void* self, CUVIDPARSERDISPINFO* shown) {
    return static_cast<NvcodecDecoder*>(self)->display(shown);
  }

  Codec codec_;
  CUcontext context_{};
  CUvideoparser parser_{};
  CUvideodecoder decoder_{};
  unsigned coded_width_{};
  unsigned coded_height_{};
  int width_{};
  int height_{};
  // Set while a packet is parsed, for the callbacks.
  cudaStream_t stream_{};
  const FrameSink* sink_{};
  std::string error_;
};

}  // namespace

Probe probe_nvcodec_decoder(Codec codec) {
  const auto& library = cuvid_library();
  if (!library.error.empty()) {
    return {.available = false, .detail = library.error};
  }
  try {
    const CudaContext context = runtime_context();
    const ContextScope scope(context.context);
    CUVIDDECODECAPS caps{};
    caps.eCodecType = cuvid_codec(codec);
    caps.eChromaFormat = cudaVideoChromaFormat_420;
    caps.nBitDepthMinus8 = 0;
    if (const CUresult status = library.get_decoder_caps(&caps); status != CUDA_SUCCESS) {
      const char* name = nullptr;
      cuGetErrorName(status, &name);
      return {.available = false,
              .detail = std::string("cuvidGetDecoderCaps: ") + (name != nullptr ? name : "failed") +
                        " on " + context.device_name};
    }
    if (caps.bIsSupported == 0) {
      return {.available = false,
              .detail =
                  "NVDEC on " + context.device_name + " has no " + std::string(codec_name(codec))};
    }
    return {.available = true, .detail = "NVDEC on " + context.device_name};
  } catch (const std::exception& error) {
    return {.available = false, .detail = error.what()};
  }
}

std::unique_ptr<DecoderBackend> create_nvcodec_decoder(Codec codec, bool low_latency) {
  return std::make_unique<NvcodecDecoder>(codec, low_latency);
}

}  // namespace holoscan::codec::detail
