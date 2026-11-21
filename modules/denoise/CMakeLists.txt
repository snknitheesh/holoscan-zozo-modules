# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_denoise; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_denoise "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_denoise)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_denoise
  VERSION 1.0.0
  DESCRIPTION
    "Spatial and motion-adaptive temporal denoising, exposure fusion and tone mapping on the GPU"
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
  set(HOLOSCAN_DENOISE_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_DENOISE_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()
option(HOLOSCAN_DENOISE_WITH_VPI "Build the vpi backend when VPI 4 is installed" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_DENOISE_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

set(HOLOSCAN_DENOISE_VPI OFF)
if(HOLOSCAN_DENOISE_WITH_VPI)
  find_package(vpi 4 QUIET CONFIG PATHS /opt/nvidia/vpi4)
  if(vpi_FOUND)
    set(HOLOSCAN_DENOISE_VPI ON)
    message(STATUS "holoscan-denoise: building the vpi backend with VPI ${vpi_VERSION}")
  else()
    message(STATUS "holoscan-denoise: VPI 4 not found; building without the vpi backend")
  endif()
endif()

# holoscan::denoise: DenoiseOp and ExposureFusionOp, the Denoiser, ExposureFusion and
# ToneMapper they run, and the synthetic scenes.
add_library(
  holoscan_denoise
  src/cuda_denoiser.cu
  src/denoise_op.cpp
  src/denoise_params.cpp
  src/denoiser.cpp
  src/exposure_fusion.cu
  src/exposure_fusion_op.cpp
  src/fusion_params.cpp
  src/image_ports.cpp
  src/noise_estimate.cu
  src/nv12_convert.cu
  src/synthetic_scene.cu
)
add_library(holoscan::denoise ALIAS holoscan_denoise)
target_compile_features(holoscan_denoise PUBLIC cxx_std_20 PRIVATE cuda_std_20)
# The kernels call the constexpr tuning functions of the public headers.
target_compile_options(
  holoscan_denoise
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_include_directories(
  holoscan_denoise
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(holoscan_denoise PUBLIC holoscan::core holoscan::sensor_io CUDA::cudart)
target_compile_definitions(
  holoscan_denoise
  PRIVATE HOLOSCAN_DENOISE_WITH_VPI=$<BOOL:${HOLOSCAN_DENOISE_VPI}>
)
if(HOLOSCAN_DENOISE_VPI)
  target_sources(holoscan_denoise PRIVATE src/vpi_denoiser.cpp)
  target_link_libraries(holoscan_denoise PRIVATE vpi)
endif()
set_target_properties(
  holoscan_denoise
  PROPERTIES
    EXPORT_NAME denoise
    OUTPUT_NAME holoscan_denoise
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, its public headers, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-denoise")
install(
  TARGETS holoscan_denoise
  EXPORT holoscan-denoise-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/denoise DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-denoise"
)
install(
  EXPORT holoscan-denoise-targets
  FILE holoscan-denoise-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-denoise-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-denoise-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-denoise-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-denoise-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-denoise-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_DENOISE_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_denoise)
  add_subdirectory(pkg/holoscan-denoise)
endif()
