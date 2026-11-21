# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_image_quality; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_image_quality "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_image_quality)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_image_quality
  VERSION 1.0.0
  DESCRIPTION
    "Per-frame image quality checks on the GPU: blur, exposure, noise, frozen frames, lens obstruction and color cast"
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
  set(HOLOSCAN_IMAGE_QUALITY_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(
    HOLOSCAN_IMAGE_QUALITY_BUILD_TESTING
    "Build the module's tests in the including project"
    OFF
  )
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_IMAGE_QUALITY_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# holoscan::image_quality: ImageQualityOp, the GPU analyzer and its kernels, the
# evaluator, the CPU reference and the synthetic scenes.
add_library(
  holoscan_image_quality
  src/analyzer.cu
  src/image_quality_op.cpp
  src/quality.cpp
  src/reference.cpp
  src/synthetic.cpp
)
add_library(holoscan::image_quality ALIAS holoscan_image_quality)
target_compile_features(holoscan_image_quality PUBLIC cxx_std_20 PRIVATE cuda_std_20)
# The kernels share the host's constexpr helpers (luma, tile bounds, pixel hash), so
# the GPU and the CPU reference compute them from the same code.
target_compile_options(
  holoscan_image_quality
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_include_directories(
  holoscan_image_quality
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(holoscan_image_quality PUBLIC holoscan::core holoscan::sensor_io CUDA::cudart)
set_target_properties(
  holoscan_image_quality
  PROPERTIES
    EXPORT_NAME image_quality
    OUTPUT_NAME holoscan_image_quality
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, its public headers, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-image-quality")
install(
  TARGETS holoscan_image_quality
  EXPORT holoscan-image-quality-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/image_quality DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-image-quality"
)
install(
  EXPORT holoscan-image-quality-targets
  FILE holoscan-image-quality-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-image-quality-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-image-quality-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-image-quality-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-image-quality-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-image-quality-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_IMAGE_QUALITY_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_image_quality)
  add_subdirectory(pkg/holoscan-image-quality)
endif()
