# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_stereo_disparity; a package is built for every major
# GPU architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_stereo_disparity "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_stereo_disparity)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_stereo_disparity
  VERSION 1.0.0
  DESCRIPTION
    "Disparity and depth from a rectified stereo pair with VPI on the GPU, the OFA or the PVA"
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
  set(HOLOSCAN_STEREO_DISPARITY_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(
    HOLOSCAN_STEREO_DISPARITY_BUILD_TESTING
    "Build the module's tests in the including project"
    OFF
  )
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_STEREO_DISPARITY_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)
# VPI 4 runs the matching on every backend.
find_package(vpi 4 REQUIRED CONFIG PATHS /opt/nvidia/vpi4)
message(STATUS "holoscan-stereo-disparity: using VPI ${vpi_VERSION}")

# The geometry message, generated from its FlatBuffers schema into the public include tree.
set(_generated_include_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
holoscan_add_flatbuffer_schema(
  TARGET holoscan_stereo_disparity_schema
  SCHEMA schemas/stereo_geometry.fbs
  OUTPUT_DIR "${_generated_include_dir}/holoscan/stereo_disparity"
  MAX_SERIALIZED_SIZE 256
)
get_target_property(
  _generated_headers
  holoscan_stereo_disparity_schema
  HOLOSCAN_FLATBUFFER_GENERATED_HEADERS
)

# holoscan::stereo_disparity: StereoDisparityOp, its VPI matcher and kernels, the geometry
# message and the synthetic scene.
add_library(
  holoscan_stereo_disparity
  src/disparity_options.cpp
  src/geometry_message.cpp
  src/kernels.cu
  src/stereo_disparity_op.cpp
  src/synthetic_scene.cu
  src/vpi_matcher.cpp
)
add_library(holoscan::stereo_disparity ALIAS holoscan_stereo_disparity)
target_compile_features(holoscan_stereo_disparity PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(
  holoscan_stereo_disparity
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${_generated_include_dir}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(
  holoscan_stereo_disparity
  PUBLIC
    holoscan::core
    holoscan::sensor_io
    CUDA::cudart
    "$<BUILD_INTERFACE:holoscan_stereo_disparity_schema>"
  PRIVATE vpi
)
set_target_properties(
  holoscan_stereo_disparity
  PROPERTIES
    EXPORT_NAME stereo_disparity
    OUTPUT_NAME holoscan_stereo_disparity
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, public and generated headers, the schema, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-stereo-disparity")
install(
  TARGETS holoscan_stereo_disparity
  EXPORT holoscan-stereo-disparity-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(
  DIRECTORY include/holoscan/stereo_disparity
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan"
)
install(
  FILES ${_generated_headers}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/stereo_disparity"
)
install(
  FILES schemas/stereo_geometry.fbs
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/holoscan-stereo-disparity/schemas"
)
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-stereo-disparity"
)
install(
  EXPORT holoscan-stereo-disparity-targets
  FILE holoscan-stereo-disparity-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-stereo-disparity-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-stereo-disparity-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-stereo-disparity-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-stereo-disparity-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-stereo-disparity-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_STEREO_DISPARITY_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_stereo_disparity)
  add_subdirectory(pkg/holoscan-stereo-disparity)
endif()
