# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_point_cloud; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_point_cloud "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_point_cloud)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_point_cloud
  VERSION 1.1.0
  DESCRIPTION "Depth images to point clouds and point clouds to depth images on the GPU"
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
  set(HOLOSCAN_POINT_CLOUD_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_POINT_CLOUD_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_POINT_CLOUD_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# holoscan::point_cloud: DepthToPointCloudOp and PointCloudToDepthOp, their kernels, the
# synthetic depth camera and cloud, and the cloud's ground truth.
add_library(
  holoscan_point_cloud
  src/deproject.cu
  src/depth_to_point_cloud_op.cpp
  src/depth_truth.cpp
  src/host_input.cpp
  src/point_cloud_to_depth_op.cpp
  src/project.cu
  src/synthetic_depth.cu
  src/synthetic_scene.cu
)
add_library(holoscan::point_cloud ALIAS holoscan_point_cloud)
# Without fused multiply-adds the kernels compute bit for bit what the host computes from
# the same expressions: the synthetic camera and cloud match their host versions, and the
# projection matches a CPU reference pixel for pixel, so checks against them can be exact.
# The projection is bound by memory and atomics, not arithmetic.
set_source_files_properties(
  src/project.cu
  src/synthetic_depth.cu
  src/synthetic_scene.cu
  PROPERTIES COMPILE_OPTIONS --fmad=false
)
# Nor does the library's host code contract, whatever the compiler's default.
target_compile_options(
  holoscan_point_cloud
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)
target_compile_features(holoscan_point_cloud PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(
  holoscan_point_cloud
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(holoscan_point_cloud PUBLIC holoscan::core holoscan::sensor_io CUDA::cudart)
set_target_properties(
  holoscan_point_cloud
  PROPERTIES
    EXPORT_NAME point_cloud
    OUTPUT_NAME holoscan_point_cloud
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, its public headers, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-point-cloud")
install(
  TARGETS holoscan_point_cloud
  EXPORT holoscan-point-cloud-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/point_cloud DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-point-cloud"
)
install(
  EXPORT holoscan-point-cloud-targets
  FILE holoscan-point-cloud-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-point-cloud-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-point-cloud-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-point-cloud-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-point-cloud-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-point-cloud-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_POINT_CLOUD_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_point_cloud)
  add_subdirectory(pkg/holoscan-point-cloud)
endif()
