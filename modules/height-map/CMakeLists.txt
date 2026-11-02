# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_height_map; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_height_map "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_height_map)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_height_map
  VERSION 1.0.0
  DESCRIPTION "2.5D elevation maps of point clouds and their traversability, on the GPU"
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
  set(HOLOSCAN_HEIGHT_MAP_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_HEIGHT_MAP_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_HEIGHT_MAP_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# holoscan::height_map: the two operators, their CUDA kernels, the views and the synthetic
# terrain.
add_library(
  holoscan_height_map
  src/grid_op.cpp
  src/height_map_op.cpp
  src/height_mapper.cu
  src/host_input.cpp
  src/kernel_common.cu
  src/synthetic_terrain.cu
  src/traversability.cu
  src/traversability_op.cpp
  src/views.cu
)
add_library(holoscan::height_map ALIAS holoscan_height_map)
# Without fused multiply-adds the kernels compute bit for bit what the host computes from the
# same expressions: the map and its traversability match the CPU references
# exactly, and the synthetic terrain has the same heights as on the host.
set_source_files_properties(
  src/height_mapper.cu
  src/synthetic_terrain.cu
  src/traversability.cu
  PROPERTIES COMPILE_OPTIONS --fmad=false
)
# Nor does the library's host code contract, whatever the compiler's default.
target_compile_options(
  holoscan_height_map
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)
target_compile_features(holoscan_height_map PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(
  holoscan_height_map
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(holoscan_height_map PUBLIC holoscan::core holoscan::sensor_io CUDA::cudart)
set_target_properties(
  holoscan_height_map
  PROPERTIES
    EXPORT_NAME height_map
    OUTPUT_NAME holoscan_height_map
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, its public headers, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-height-map")
install(
  TARGETS holoscan_height_map
  EXPORT holoscan-height-map-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/height_map DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-height-map"
)
install(
  EXPORT holoscan-height-map-targets
  FILE holoscan-height-map-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-height-map-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-height-map-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-height-map-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-height-map-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-height-map-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_HEIGHT_MAP_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_height_map)
  add_subdirectory(pkg/holoscan-height-map)
endif()
