# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_mono_depth; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_mono_depth "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_mono_depth)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_mono_depth
  VERSION 1.0.0
  DESCRIPTION "Monocular depth estimation on the GPU with TensorRT, for Depth Anything V2"
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
  set(HOLOSCAN_MONO_DEPTH_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_MONO_DEPTH_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# HoloInfer, the SDK's TensorRT inference library, comes with its inference component.
set(_holoscan_components core sensor_io inference)
if(HOLOSCAN_MONO_DEPTH_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# holoscan::mono_depth: MonoDepthOp, its kernels, and the stub model, synthetic scene and
# host references for tests.
add_library(
  holoscan_mono_depth
  src/image_fit.cpp
  src/kernels.cu
  src/mono_depth_op.cpp
  src/reference.cpp
  src/stub_model.cpp
  src/synthetic_scene.cpp
)
add_library(holoscan::mono_depth ALIAS holoscan_mono_depth)
# Without fused multiply-adds the kernels compute bit for bit what the host references
# compute from the same expressions (pixel_math.hpp), so the tests compare them exactly.
set_source_files_properties(src/kernels.cu PROPERTIES COMPILE_OPTIONS -fmad=false)
target_compile_features(holoscan_mono_depth PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_compile_options(
  holoscan_mono_depth
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)
target_include_directories(
  holoscan_mono_depth
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(
  holoscan_mono_depth
  PUBLIC holoscan::core holoscan::sensor_io CUDA::cudart
  PRIVATE holoscan::infer
)
set_target_properties(
  holoscan_mono_depth
  PROPERTIES
    EXPORT_NAME mono_depth
    OUTPUT_NAME holoscan_mono_depth
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, public headers, the model preparation tool, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-mono-depth")
set(_share_dir "${CMAKE_INSTALL_DATAROOTDIR}/holoscan-mono-depth")
install(
  TARGETS holoscan_mono_depth
  EXPORT holoscan-mono-depth-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/mono_depth DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(PROGRAMS tools/prepare_model.py DESTINATION "${_share_dir}")
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-mono-depth"
)
install(
  EXPORT holoscan-mono-depth-targets
  FILE holoscan-mono-depth-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
set(HOLOSCAN_MONO_DEPTH_PREPARE_MODEL "${_share_dir}/prepare_model.py")
configure_package_config_file(
  cmake/holoscan-mono-depth-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-mono-depth-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  PATH_VARS HOLOSCAN_MONO_DEPTH_PREPARE_MODEL
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-mono-depth-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-mono-depth-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-mono-depth-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_MONO_DEPTH_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_mono_depth)
  add_subdirectory(pkg/holoscan-mono-depth)
endif()
