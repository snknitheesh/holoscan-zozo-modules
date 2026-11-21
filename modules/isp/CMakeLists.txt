# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_isp; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_isp "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_isp)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_isp
  VERSION 1.0.0
  DESCRIPTION "An image signal processor for raw Bayer frames, on the GPU or the CPU"
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
  set(HOLOSCAN_ISP_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_ISP_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_ISP_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# holoscan::isp: IspOp, IspPipeline with its CUDA and CPU backends, and the synthetic raw
# camera.
add_library(
  holoscan_isp
  src/cpu_pipeline.cpp
  src/cuda_pipeline.cu
  src/isp_op.cpp
  src/isp_params.cpp
  src/isp_pipeline.cpp
  src/synthetic_raw.cpp
  src/synthetic_raw.cu
)
add_library(holoscan::isp ALIAS holoscan_isp)
# Without fused multiply-adds, the kernels compute bit for bit what the CPU backend computes
# from the same expressions, and the synthetic camera renders the same raw frames on both.
target_compile_options(
  holoscan_isp
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false
    -Xcompiler=-ffp-contract=off>
)
target_compile_features(holoscan_isp PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(
  holoscan_isp
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(holoscan_isp PUBLIC holoscan::core holoscan::sensor_io CUDA::cudart)
set_target_properties(
  holoscan_isp
  PROPERTIES
    EXPORT_NAME isp
    OUTPUT_NAME holoscan_isp
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, its public headers, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-isp")
install(
  TARGETS holoscan_isp
  EXPORT holoscan-isp-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/isp DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-isp"
)
install(
  EXPORT holoscan-isp-targets
  FILE holoscan-isp-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-isp-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-isp-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-isp-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-isp-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-isp-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_ISP_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_isp)
  add_subdirectory(pkg/holoscan-isp)
endif()
