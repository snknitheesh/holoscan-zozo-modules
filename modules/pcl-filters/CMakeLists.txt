# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_pcl_filters; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_pcl_filters "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_pcl_filters)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_pcl_filters
  VERSION 1.0.0
  DESCRIPTION
    "Point-cloud filters on the GPU: voxel grid, crop box, ground removal and outlier removal"
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
  set(HOLOSCAN_PCL_FILTERS_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_PCL_FILTERS_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_PCL_FILTERS_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# holoscan::pcl_filters: the four operators, their CUDA filters and the synthetic scene.
add_library(
  holoscan_pcl_filters
  src/cloud_filter_op.cpp
  src/crop_box.cu
  src/crop_box_op.cpp
  src/ground.cu
  src/ground_removal_op.cpp
  src/host_input.cpp
  src/mask.cu
  src/outlier_removal_op.cpp
  src/outliers.cu
  src/synthetic_scene.cu
  src/voxel_downsample_op.cpp
  src/voxel_grid.cu
)
add_library(holoscan::pcl_filters ALIAS holoscan_pcl_filters)
# Without fused multiply-adds the filters compute bit for bit what the host computes from the
# same expressions, so the tests check them exactly against CPU references; the synthetic
# scene renders the same depths as on the host. They are bound by memory, not arithmetic.
# RANSAC's scoring is arithmetic and is checked against the truth with a tolerance.
set_source_files_properties(
  src/crop_box.cu
  src/outliers.cu
  src/synthetic_scene.cu
  src/voxel_grid.cu
  PROPERTIES COMPILE_OPTIONS --fmad=false
)
# Nor does the library's host code contract, whatever the compiler's default.
target_compile_options(
  holoscan_pcl_filters
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)
target_compile_features(holoscan_pcl_filters PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(
  holoscan_pcl_filters
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(holoscan_pcl_filters PUBLIC holoscan::core holoscan::sensor_io CUDA::cudart)
set_target_properties(
  holoscan_pcl_filters
  PROPERTIES
    EXPORT_NAME pcl_filters
    OUTPUT_NAME holoscan_pcl_filters
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, its public headers, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-pcl-filters")
install(
  TARGETS holoscan_pcl_filters
  EXPORT holoscan-pcl-filters-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/pcl_filters DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-pcl-filters"
)
install(
  EXPORT holoscan-pcl-filters-targets
  FILE holoscan-pcl-filters-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-pcl-filters-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-pcl-filters-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-pcl-filters-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-pcl-filters-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-pcl-filters-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_PCL_FILTERS_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_pcl_filters)
  add_subdirectory(pkg/holoscan-pcl-filters)
endif()
