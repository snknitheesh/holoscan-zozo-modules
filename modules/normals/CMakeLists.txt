# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_normals; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_normals "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_normals)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_normals
  VERSION 1.0.0
  DESCRIPTION "Surface normals, curvature and multi-plane segmentation of point clouds on the GPU"
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
  set(HOLOSCAN_NORMALS_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_NORMALS_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_NORMALS_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# holoscan::normals: the two operators, their CUDA kernels, the views and the synthetic scene.
add_library(
  holoscan_normals
  src/cloud_op.cpp
  src/colorize.cu
  src/compact.cu
  src/host_input.cpp
  src/normal_estimation_op.cpp
  src/normals.cu
  src/plane_segmentation_op.cpp
  src/planes.cu
  src/synthetic_scene.cu
)
add_library(holoscan::normals ALIAS holoscan_normals)
# Without fused multiply-adds the kernels compute bit for bit what the host computes from the
# same expressions: the neighbor distances and plane hypotheses match the CPU references
# exactly, and the synthetic scene renders the same depths as on the host.
set_source_files_properties(
  src/normals.cu
  src/planes.cu
  src/synthetic_scene.cu
  PROPERTIES COMPILE_OPTIONS --fmad=false
)
# Nor does the library's host code contract, whatever the compiler's default.
target_compile_options(
  holoscan_normals
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)
target_compile_features(holoscan_normals PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(
  holoscan_normals
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(holoscan_normals PUBLIC holoscan::core holoscan::sensor_io CUDA::cudart)
set_target_properties(
  holoscan_normals
  PROPERTIES
    EXPORT_NAME normals
    OUTPUT_NAME holoscan_normals
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, its public headers, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-normals")
install(
  TARGETS holoscan_normals
  EXPORT holoscan-normals-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/normals DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-normals"
)
install(
  EXPORT holoscan-normals-targets
  FILE holoscan-normals-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-normals-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-normals-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-normals-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-normals-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-normals-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_NORMALS_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_normals)
  add_subdirectory(pkg/holoscan-normals)
endif()
