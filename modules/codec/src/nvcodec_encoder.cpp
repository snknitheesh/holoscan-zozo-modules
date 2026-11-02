// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The nvcodec encoder: NVENC through the Video Codec SDK, reading NV12 straight from CUDA device
// memory. One frame in, one access unit out: synchronous, no B-frames, IDRs placed by the caller.

#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backend.hpp"
#include "nvcodec_api.hpp"

namespace holoscan::codec::detail {
namespace {

std::string nvenc_error(const char* what, NVENCSTATUS status, void* encoder) {
  std::string message = std::string("NVENC ") + what + " failed (status " +
                        std::to_string(static_cast<int>(status)) + ")";
  if (encoder != nullptr) {
    const char* detail = nvenc_library().api.nvEncGetLastErrorString(encoder);
    if (detail != nullptr && *detail != '\0') {
      message += ": ";
      message += detail;
    }
  }
  return message;
}

void check(NVENCSTATUS status, const char* what, void* encoder) {
  if (status != NV_ENC_SUCCESS) {
    throw std::runtime_error(nvenc_error(what, status, encoder));
  }
}

GUID codec_guid(Codec codec) {
  return codec == Codec::kH264 ? NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
}

GUID preset_guid(Preset preset) {
  switch (preset) {
    case Preset::kMedium:
      return NV_ENC_PRESET_P4_GUID;
    case Preset::kQuality:
      return NV_ENC_PRESET_P7_GUID;
    case Preset::kFast:
      break;
  }
  return NV_ENC_PRESET_P1_GUID;
}

// An open NVENC session, closed on destruction.
class Session {
 public:
  Session() = default;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  ~Session() {
    if (encoder_ != nullptr) {
      nvenc_library().api.nvEncDestroyEncoder(encoder_);
    }
  }

  // Opens a session on `context`; returns the status, leaving the session closed on failure.
  NVENCSTATUS open(CUcontext context) {
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS params{};
    params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    params.device = context;
    params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    params.apiVersion = NVENCAPI_VERSION;
    void* encoder = nullptr;
    const NVENCSTATUS status = nvenc_library().api.nvEncOpenEncodeSessionEx(&params, &encoder);
    if (status == NV_ENC_SUCCESS) {
      encoder_ = encoder;
    } else if (encoder != nullptr) {
      nvenc_library().api.nvEncDestroyEncoder(encoder);
    }
    return status;
  }

  [[nodiscard]] void* get() const noexcept { return encoder_; }

  [[nodiscard]] bool supports(const GUID& codec) const {
    const auto& api = nvenc_library().api;
    std::uint32_t count = 0;
    if (api.nvEncGetEncodeGUIDCount(encoder_, &count) != NV_ENC_SUCCESS || count == 0) {
      return false;
    }
    std::vector<GUID> guids(count);
    if (api.nvEncGetEncodeGUIDs(encoder_, guids.data(), count, &count) != NV_ENC_SUCCESS) {
      return false;
    }
    for (std::uint32_t i = 0; i < count; ++i) {
      if (std::memcmp(&guids[i], &codec, sizeof(GUID)) == 0) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] int caps(const GUID& codec, NV_ENC_CAPS what) const {
    NV_ENC_CAPS_PARAM param{};
    param.version = NV_ENC_CAPS_PARAM_VER;
    param.capsToQuery = what;
    int value = 0;
    if (nvenc_library().api.nvEncGetEncodeCaps(encoder_, codec, &param, &value) != NV_ENC_SUCCESS) {
      return 0;
    }
    return value;
  }

 private:
  void* encoder_{};
};

// BT.709, limited range, as the conversion to NV12 writes it.
void describe_colors(NV_ENC_CONFIG_H264_VUI_PARAMETERS& vui) {
  vui.videoSignalTypePresentFlag = 1;
  vui.videoFormat = NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
  vui.videoFullRangeFlag = 0;
  vui.colourDescriptionPresentFlag = 1;
  vui.colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES_BT709;
  vui.transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
  vui.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS_BT709;
  // Tells decoders no frame is reordered, so they output each one as soon as it is decoded.
  vui.bitstreamRestrictionFlag = 1;
}

class NvcodecEncoder final : public EncoderBackend {
 public:
  NvcodecEncoder(const EncoderSettings& settings, int width, int height)
      : settings_(settings), width_(width), height_(height) {
    const auto& library = nvenc_library();
    if (!library.error.empty()) {
      throw std::runtime_error("nvcodec encoder: " + library.error);
    }
    const CudaContext context = runtime_context();
    check(session_.open(context.context), "nvEncOpenEncodeSessionEx", nullptr);
    void* encoder = session_.get();
    const auto& api = library.api;
    const GUID codec = codec_guid(settings.codec);
    if (!session_.supports(codec)) {
      throw std::runtime_error("nvcodec encoder: this GPU's NVENC has no " +
                               std::string(codec_name(settings.codec)));
    }
    const int max_width = session_.caps(codec, NV_ENC_CAPS_WIDTH_MAX);
    const int max_height = session_.caps(codec, NV_ENC_CAPS_HEIGHT_MAX);
    const int min_width = session_.caps(codec, NV_ENC_CAPS_WIDTH_MIN);
    const int min_height = session_.caps(codec, NV_ENC_CAPS_HEIGHT_MIN);
    if (width > max_width || height > max_height || width < min_width || height < min_height) {
      throw std::invalid_argument(
          "nvcodec encoder: " + std::to_string(width) + "x" + std::to_string(height) +
          " is outside "
          "what this NVENC encodes in " +
          std::string(codec_name(settings.codec)) + ", " + std::to_string(min_width) + "x" +
          std::to_string(min_height) + " to " + std::to_string(max_width) + "x" +
          std::to_string(max_height));
    }

    const NV_ENC_TUNING_INFO tuning = settings.low_latency ? NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY
                                                           : NV_ENC_TUNING_INFO_HIGH_QUALITY;
    const GUID preset = preset_guid(settings.preset);
    NV_ENC_PRESET_CONFIG preset_config{};
    preset_config.version = NV_ENC_PRESET_CONFIG_VER;
    preset_config.presetCfg.version = NV_ENC_CONFIG_VER;
    check(api.nvEncGetEncodePresetConfigEx(encoder, codec, preset, tuning, &preset_config),
          "nvEncGetEncodePresetConfigEx", encoder);
    config_ = preset_config.presetCfg;
    config_.version = NV_ENC_CONFIG_VER;
    // IDRs come where encode() is asked for them, never on the encoder's own schedule.
    config_.gopLength = NVENC_INFINITE_GOPLENGTH;
    config_.frameIntervalP = 1;  // I and P frames only.
    NV_ENC_RC_PARAMS& rc = config_.rcParams;
    switch (settings.rate_control) {
      case RateControl::kConstQp:
        rc.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
        rc.constQP = {static_cast<std::uint32_t>(settings.qp),
                      static_cast<std::uint32_t>(settings.qp),
                      static_cast<std::uint32_t>(settings.qp)};
        break;
      case RateControl::kCbr:
        rc.rateControlMode = NV_ENC_PARAMS_RC_CBR;
        rc.averageBitRate = settings.bitrate;
        rc.maxBitRate = settings.bitrate;
        break;
      case RateControl::kVbr:
        rc.rateControlMode = NV_ENC_PARAMS_RC_VBR;
        rc.averageBitRate = settings.bitrate;
        rc.maxBitRate = settings.max_bitrate != 0 ? settings.max_bitrate : 2U * settings.bitrate;
        break;
    }
    if (settings.rate_control != RateControl::kConstQp && settings.low_latency) {
      // A one-frame VBV: every frame fits the channel in one frame time.
      rc.vbvBufferSize =
          static_cast<std::uint32_t>(static_cast<double>(settings.bitrate) / settings.frame_rate);
      rc.vbvInitialDelay = rc.vbvBufferSize;
    }
    rc.zeroReorderDelay = 1;
    if (settings.codec == Codec::kH264) {
      NV_ENC_CONFIG_H264& h264 = config_.encodeCodecConfig.h264Config;
      config_.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
      h264.idrPeriod = NVENC_INFINITE_GOPLENGTH;
      h264.repeatSPSPPS = 1;
      if (settings.low_latency) {
        h264.maxNumRefFrames = 1;  // A low-latency decoder can show each frame at once.
      }
      describe_colors(h264.h264VUIParameters);
    } else {
      NV_ENC_CONFIG_HEVC& hevc = config_.encodeCodecConfig.hevcConfig;
      config_.profileGUID = NV_ENC_HEVC_PROFILE_MAIN_GUID;
      hevc.idrPeriod = NVENC_INFINITE_GOPLENGTH;
      hevc.repeatSPSPPS = 1;
      if (settings.low_latency) {
        hevc.maxNumRefFramesInDPB = 1;
      }
      describe_colors(hevc.hevcVUIParameters);
    }

    NV_ENC_INITIALIZE_PARAMS init{};
    init.version = NV_ENC_INITIALIZE_PARAMS_VER;
    init.encodeGUID = codec;
    init.presetGUID = preset;
    init.tuningInfo = tuning;
    init.encodeWidth = static_cast<std::uint32_t>(width);
    init.encodeHeight = static_cast<std::uint32_t>(height);
    init.darWidth = init.encodeWidth;
    init.darHeight = init.encodeHeight;
    init.maxEncodeWidth = init.encodeWidth;
    init.maxEncodeHeight = init.encodeHeight;
    init.frameRateNum = static_cast<std::uint32_t>(settings.frame_rate * 1000.0 + 0.5);
    init.frameRateDen = 1000;
    init.enablePTD = 1;
    init.encodeConfig = &config_;
    check(api.nvEncInitializeEncoder(encoder, &init), "nvEncInitializeEncoder", encoder);

    NV_ENC_CREATE_BITSTREAM_BUFFER bitstream{};
    bitstream.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    check(api.nvEncCreateBitstreamBuffer(encoder, &bitstream), "nvEncCreateBitstreamBuffer",
          encoder);
    bitstream_ = bitstream.bitstreamBuffer;
  }

  ~NvcodecEncoder() override {
    const auto& api = nvenc_library().api;
    void* encoder = session_.get();
    if (registered_ != nullptr) {
      api.nvEncUnregisterResource(encoder, registered_);
    }
    if (bitstream_ != nullptr) {
      api.nvEncDestroyBitstreamBuffer(encoder, bitstream_);
    }
  }

  std::vector<std::uint8_t> encode(const Nv12Frame& frame, bool keyframe) override {
    if (frame.width != width_ || frame.height != height_ ||
        frame.luma_pitch != frame.chroma_pitch ||
        frame.chroma != frame.luma + frame.luma_pitch * static_cast<std::size_t>(frame.height)) {
      throw std::invalid_argument(
          "nvcodec encoder: a frame of the configured size, its chroma right after its luma at "
          "the same pitch, is needed");
    }
    const auto& api = nvenc_library().api;
    void* encoder = session_.get();
    register_input(frame);

    NV_ENC_MAP_INPUT_RESOURCE map{};
    map.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
    map.registeredResource = registered_;
    check(api.nvEncMapInputResource(encoder, &map), "nvEncMapInputResource", encoder);
    struct Unmap {
      void* encoder;
      NV_ENC_INPUT_PTR mapped;
      ~Unmap() { nvenc_library().api.nvEncUnmapInputResource(encoder, mapped); }
    } unmap{encoder, map.mappedResource};

    NV_ENC_PIC_PARAMS picture{};
    picture.version = NV_ENC_PIC_PARAMS_VER;
    picture.inputWidth = static_cast<std::uint32_t>(width_);
    picture.inputHeight = static_cast<std::uint32_t>(height_);
    picture.inputPitch = static_cast<std::uint32_t>(frame.luma_pitch);
    picture.inputBuffer = map.mappedResource;
    picture.bufferFmt = map.mappedBufferFmt;
    picture.outputBitstream = bitstream_;
    picture.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    picture.inputTimeStamp = frames_++;
    if (keyframe) {
      picture.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
    }
    check(api.nvEncEncodePicture(encoder, &picture), "nvEncEncodePicture", encoder);

    NV_ENC_LOCK_BITSTREAM lock{};
    lock.version = NV_ENC_LOCK_BITSTREAM_VER;
    lock.outputBitstream = bitstream_;
    check(api.nvEncLockBitstream(encoder, &lock), "nvEncLockBitstream", encoder);
    const auto* bytes = static_cast<const std::uint8_t*>(lock.bitstreamBufferPtr);
    std::vector<std::uint8_t> access_unit(bytes, bytes + lock.bitstreamSizeInBytes);
    check(api.nvEncUnlockBitstream(encoder, bitstream_), "nvEncUnlockBitstream", encoder);
    return access_unit;
  }

 private:
  // Registers the frame's memory with NVENC, once per buffer.
  void register_input(const Nv12Frame& frame) {
    if (registered_ != nullptr && registered_luma_ == frame.luma &&
        registered_pitch_ == frame.luma_pitch) {
      return;
    }
    const auto& api = nvenc_library().api;
    void* encoder = session_.get();
    if (registered_ != nullptr) {
      api.nvEncUnregisterResource(encoder, registered_);
      registered_ = nullptr;
    }
    NV_ENC_REGISTER_RESOURCE resource{};
    resource.version = NV_ENC_REGISTER_RESOURCE_VER;
    resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    resource.width = static_cast<std::uint32_t>(width_);
    resource.height = static_cast<std::uint32_t>(height_);
    resource.pitch = static_cast<std::uint32_t>(frame.luma_pitch);
    resource.resourceToRegister = const_cast<std::uint8_t*>(frame.luma);
    resource.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
    resource.bufferUsage = NV_ENC_INPUT_IMAGE;
    check(api.nvEncRegisterResource(encoder, &resource), "nvEncRegisterResource", encoder);
    registered_ = resource.registeredResource;
    registered_luma_ = frame.luma;
    registered_pitch_ = frame.luma_pitch;
  }

  EncoderSettings settings_;
  int width_;
  int height_;
  Session session_;
  NV_ENC_CONFIG config_{};
  NV_ENC_OUTPUT_PTR bitstream_{};
  NV_ENC_REGISTERED_PTR registered_{};
  const std::uint8_t* registered_luma_{};
  std::size_t registered_pitch_{};
  std::uint64_t frames_{};
};

}  // namespace

Probe probe_nvcodec_encoder(Codec codec) {
  const auto& library = nvenc_library();
  if (!library.error.empty()) {
    return {.available = false, .detail = library.error};
  }
  try {
    const CudaContext context = runtime_context();
    Session session;
    if (const NVENCSTATUS status = session.open(context.context); status != NV_ENC_SUCCESS) {
      return {.available = false,
              .detail = nvenc_error("nvEncOpenEncodeSessionEx", status, nullptr) + " on " +
                        context.device_name};
    }
    if (!session.supports(codec_guid(codec))) {
      return {.available = false,
              .detail =
                  "NVENC on " + context.device_name + " has no " + std::string(codec_name(codec))};
    }
    return {.available = true, .detail = "NVENC on " + context.device_name};
  } catch (const std::exception& error) {
    return {.available = false, .detail = error.what()};
  }
}

std::unique_ptr<EncoderBackend> create_nvcodec_encoder(const EncoderSettings& settings, int width,
                                                       int height) {
  return std::make_unique<NvcodecEncoder>(settings, width, height);
}

}  // namespace holoscan::codec::detail
