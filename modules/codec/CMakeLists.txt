# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_codec; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_codec "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_codec)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_codec
  VERSION 1.0.0
  DESCRIPTION "Hardware H.264 and H.265 encoding and decoding of GPU images, with NVENC and NVDEC"
  LANGUAGES CXX CUDA
)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Built on its own, the usual CMake switches; inside another project (the repository root,
# FetchContent, add_subdirectory), shared libraries and no tests unless asked for, and the
# including project's own BUILD_SHARED_LIBS and BUILD_TESTING are left alone.
if(PROJECT_IS_TOP_LEVEL)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
  option(BUILD_TESTING "Build the module's tests" OFF)
  set(HOLOSCAN_CODEC_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_CODEC_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()
option(
  HOLOSCAN_CODEC_WITH_V4L2
  "Build the v4l2 backend (Jetson V4L2 codec devices) when VPI 4's headers are installed"
  ON
)
set(
  HOLOSCAN_CODEC_NV_CODEC_HEADERS_DIR
  ""
  CACHE PATH
  "A local nv-codec-headers checkout (its include/ folder) instead of the pinned download"
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_CODEC_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# The Video Codec SDK's interface headers (MIT), as FFmpeg's nv-codec-headers ship them: the
# release for SDK 12.1, which drivers from 530 on support. Only the headers: the libraries come
# with the NVIDIA driver and are loaded at run time.
if(HOLOSCAN_CODEC_NV_CODEC_HEADERS_DIR)
  set(_nv_codec_include "${HOLOSCAN_CODEC_NV_CODEC_HEADERS_DIR}")
else()
  include(FetchContent)
  FetchContent_Declare(
    holoscan_codec_nv_codec_headers
    URL https://github.com/FFmpeg/nv-codec-headers/archive/refs/tags/n12.1.14.1.tar.gz
    URL_HASH SHA256=f40c8a27417eb2688412da32985e382e811196311c4d103d1fcaafad2876cfc1
  )
  FetchContent_MakeAvailable(holoscan_codec_nv_codec_headers)
  set(_nv_codec_include "${holoscan_codec_nv_codec_headers_SOURCE_DIR}/include")
endif()
if(NOT EXISTS "${_nv_codec_include}/ffnvcodec/nvEncodeAPI.h")
  message(FATAL_ERROR "holoscan-codec: no ffnvcodec/nvEncodeAPI.h in ${_nv_codec_include}")
endif()

# The v4l2 backend reads VPI 4's headers; VPI itself is loaded at run time, on a Jetson.
set(HOLOSCAN_CODEC_V4L2 OFF)
if(HOLOSCAN_CODEC_WITH_V4L2)
  find_path(
    HOLOSCAN_CODEC_VPI_INCLUDE_DIR
    vpi/Image.h
    PATHS /opt/nvidia/vpi4/include
    DOC "VPI 4's include folder, for the v4l2 backend"
  )
  if(HOLOSCAN_CODEC_VPI_INCLUDE_DIR)
    set(HOLOSCAN_CODEC_V4L2 ON)
    message(STATUS "holoscan-codec: building the v4l2 backend")
  else()
    message(STATUS "holoscan-codec: VPI 4's headers not found; building without the v4l2 backend")
  endif()
endif()

# The encoded-frame message, generated from its FlatBuffers schema into the public include tree.
set(_generated_include_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
holoscan_add_flatbuffer_schema(
  TARGET holoscan_codec_schema
  SCHEMA schemas/encoded_video_frame.fbs
  OUTPUT_DIR "${_generated_include_dir}/holoscan/codec"
  MAX_SERIALIZED_SIZE 4096
)
get_target_property(_generated_headers holoscan_codec_schema HOLOSCAN_FLATBUFFER_GENERATED_HEADERS)

# holoscan::codec: VideoEncoder, VideoDecoder, their backends, the operators, the Annex-B reader
# and the test video.
add_library(
  holoscan_codec
  src/bitstream.cpp
  src/codec_types.cpp
  src/color_convert.cu
  src/image_input.cpp
  src/nvcodec_api.cpp
  src/nvcodec_decoder.cpp
  src/nvcodec_encoder.cpp
  src/test_video.cpp
  src/video_codec.cpp
  src/video_decoder_op.cpp
  src/video_encoder_op.cpp
)
if(HOLOSCAN_CODEC_V4L2)
  target_sources(
    holoscan_codec
    PRIVATE src/v4l2_decoder.cpp src/v4l2_device.cpp src/v4l2_encoder.cpp
  )
  target_include_directories(holoscan_codec SYSTEM PRIVATE "${HOLOSCAN_CODEC_VPI_INCLUDE_DIR}")
else()
  target_sources(holoscan_codec PRIVATE src/v4l2_unavailable.cpp)
endif()
add_library(holoscan::codec ALIAS holoscan_codec)
target_compile_features(holoscan_codec PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(
  holoscan_codec
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${_generated_include_dir}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_include_directories(holoscan_codec SYSTEM PRIVATE "${_nv_codec_include}")
target_link_libraries(
  holoscan_codec
  PUBLIC holoscan::core holoscan::sensor_io CUDA::cudart "$<BUILD_INTERFACE:holoscan_codec_schema>"
  PRIVATE CUDA::cuda_driver ${CMAKE_DL_LIBS}
)
target_compile_definitions(
  holoscan_codec
  PRIVATE HOLOSCAN_CODEC_WITH_V4L2=$<BOOL:${HOLOSCAN_CODEC_V4L2}>
)
set_target_properties(
  holoscan_codec
  PROPERTIES
    EXPORT_NAME codec
    OUTPUT_NAME holoscan_codec
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, public and generated headers, the schema, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-codec")
install(
  TARGETS holoscan_codec
  EXPORT holoscan-codec-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/codec DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(FILES ${_generated_headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/codec")
install(
  FILES schemas/encoded_video_frame.fbs
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/holoscan-codec/schemas"
)
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-codec"
)
install(
  EXPORT holoscan-codec-targets
  FILE holoscan-codec-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-codec-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-codec-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-codec-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-codec-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-codec-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_CODEC_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_codec)
  add_subdirectory(pkg/holoscan-codec)
endif()
