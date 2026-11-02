// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The Video Codec SDK's libraries, loaded at run time: the NVIDIA driver installs them (and the
// NVIDIA Container Toolkit mounts them when NVIDIA_DRIVER_CAPABILITIES includes `video`), so the
// module links neither and runs, without the nvcodec backend, where they are missing.

#pragma once

#include <cuda.h>

#include <cstdint>
#include <string>

// The interface headers (MIT), from FFmpeg's nv-codec-headers; see the module's NOTICE.
#include <ffnvcodec/dynlink_nvcuvid.h>  // NOLINT(build/include_order)
#include <ffnvcodec/nvEncodeAPI.h>      // NOLINT(build/include_order)

namespace holoscan::codec::detail {

// libnvidia-encode, loaded once.
struct NvencLibrary {
  NV_ENCODE_API_FUNCTION_LIST api{};
  // NvEncodeAPIGetMaxSupportedVersion: major << 4 | minor.
  std::uint32_t driver_version{};
  // Empty when loaded; otherwise why not.
  std::string error;
};
[[nodiscard]] const NvencLibrary& nvenc_library();

// libnvcuvid, loaded once.
struct CuvidLibrary {
  tcuvidGetDecoderCaps* get_decoder_caps{};
  tcuvidCreateDecoder* create_decoder{};
  tcuvidDestroyDecoder* destroy_decoder{};
  tcuvidDecodePicture* decode_picture{};
  tcuvidMapVideoFrame64* map_video_frame{};
  tcuvidUnmapVideoFrame64* unmap_video_frame{};
  tcuvidCreateVideoParser* create_video_parser{};
  tcuvidParseVideoData* parse_video_data{};
  tcuvidDestroyVideoParser* destroy_video_parser{};
  std::string error;
};
[[nodiscard]] const CuvidLibrary& cuvid_library();

// The primary context of the CUDA runtime's current device, which NVENC and NVDEC need, and the
// device's name.
struct CudaContext {
  CUcontext context{};
  std::string device_name;
};
// @throws std::runtime_error when there is no usable CUDA device.
[[nodiscard]] CudaContext runtime_context();

// "<what>: <driver error name>" as a runtime_error, unless `status` is CUDA_SUCCESS.
void check_cu(CUresult status, const char* what);

}  // namespace holoscan::codec::detail
