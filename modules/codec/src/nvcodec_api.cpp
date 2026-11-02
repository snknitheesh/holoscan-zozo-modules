// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "nvcodec_api.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include "backend.hpp"

namespace holoscan::codec::detail {
namespace {

// The libraries' sonames, as the driver installs them.
constexpr const char* kNvencLibrary = "libnvidia-encode.so.1";
constexpr const char* kCuvidLibrary = "libnvcuvid.so.1";

std::string dl_error(const char* library) {
  const char* message = dlerror();
  return std::string(library) + ": " + (message != nullptr ? message : "cannot be loaded");
}

NvencLibrary load_nvenc() {
  NvencLibrary library;
  // Never unloaded: sessions may outlive any owner of the handle.
  void* handle = dlopen(kNvencLibrary, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    library.error = dl_error(kNvencLibrary);
    return library;
  }
  using GetMaxVersion = NVENCSTATUS(NVENCAPI*)(std::uint32_t*);
  using CreateInstance = NVENCSTATUS(NVENCAPI*)(NV_ENCODE_API_FUNCTION_LIST*);
  auto get_max_version =
      reinterpret_cast<GetMaxVersion>(dlsym(handle, "NvEncodeAPIGetMaxSupportedVersion"));
  auto create_instance =
      reinterpret_cast<CreateInstance>(dlsym(handle, "NvEncodeAPICreateInstance"));
  if (get_max_version == nullptr || create_instance == nullptr) {
    library.error = std::string(kNvencLibrary) + " lacks the NVENC entry points";
    return library;
  }
  if (get_max_version(&library.driver_version) != NV_ENC_SUCCESS) {
    library.error = "NVENC did not report its version";
    return library;
  }
  constexpr std::uint32_t kHeaderVersion = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
  if (library.driver_version < kHeaderVersion) {
    library.error = "the driver's NVENC supports API " +
                    std::to_string(library.driver_version >> 4) + "." +
                    std::to_string(library.driver_version & 0xF) + ", older than the " +
                    std::to_string(NVENCAPI_MAJOR_VERSION) + "." +
                    std::to_string(NVENCAPI_MINOR_VERSION) + " this module needs";
    return library;
  }
  library.api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
  if (create_instance(&library.api) != NV_ENC_SUCCESS) {
    library.error = "NvEncodeAPICreateInstance failed";
  }
  return library;
}

template <typename Function>
bool bind(void* handle, const char* name, Function*& target) {
  target = reinterpret_cast<Function*>(dlsym(handle, name));
  return target != nullptr;
}

CuvidLibrary load_cuvid() {
  CuvidLibrary library;
  void* handle = dlopen(kCuvidLibrary, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    library.error = dl_error(kCuvidLibrary);
    return library;
  }
  const bool bound = bind(handle, "cuvidGetDecoderCaps", library.get_decoder_caps) &&
                     bind(handle, "cuvidCreateDecoder", library.create_decoder) &&
                     bind(handle, "cuvidDestroyDecoder", library.destroy_decoder) &&
                     bind(handle, "cuvidDecodePicture", library.decode_picture) &&
                     bind(handle, "cuvidMapVideoFrame64", library.map_video_frame) &&
                     bind(handle, "cuvidUnmapVideoFrame64", library.unmap_video_frame) &&
                     bind(handle, "cuvidCreateVideoParser", library.create_video_parser) &&
                     bind(handle, "cuvidParseVideoData", library.parse_video_data) &&
                     bind(handle, "cuvidDestroyVideoParser", library.destroy_video_parser);
  if (!bound) {
    library.error = std::string(kCuvidLibrary) + " lacks the NVDEC entry points";
  }
  return library;
}

}  // namespace

const NvencLibrary& nvenc_library() {
  static const NvencLibrary library = load_nvenc();
  return library;
}

const CuvidLibrary& cuvid_library() {
  static const CuvidLibrary library = load_cuvid();
  return library;
}

void check_cu(CUresult status, const char* what) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  cuGetErrorName(status, &name);
  throw std::runtime_error(std::string(what) + ": " + (name != nullptr ? name : "CUDA error"));
}

CudaContext runtime_context() {
  static std::mutex mutex;
  static std::map<int, CudaContext> contexts;
  const int device = current_device();
  const std::scoped_lock lock(mutex);
  if (const auto found = contexts.find(device); found != contexts.end()) {
    return found->second;
  }
  CUdevice cu_device = 0;
  check_cu(cuDeviceGet(&cu_device, device), "cuDeviceGet");
  CudaContext result;
  // The runtime API's own context, in which cudaMalloc's pointers are valid. Retained once per
  // device for the life of the process, as the runtime itself does.
  check_cu(cuDevicePrimaryCtxRetain(&result.context, cu_device), "cuDevicePrimaryCtxRetain");
  char name[256] = {};
  if (cuDeviceGetName(name, sizeof(name), cu_device) == CUDA_SUCCESS) {
    result.device_name = name;
  }
  contexts.emplace(device, result);
  return result;
}

}  // namespace holoscan::codec::detail
