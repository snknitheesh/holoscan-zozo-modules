# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_sam3; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_sam3 "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_sam3)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_sam3
  VERSION 1.0.0
  DESCRIPTION
    "Promptable concept segmentation with SAM 3 on the GPU through TensorRT, and its HoloViz overlay"
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
  set(HOLOSCAN_SAM3_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_SAM3_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()
# The repository root lists the operators asked for; without Sam3OverlayOp, the overlay,
# and so Holoscan Visualization, is left out.
set(_overlay_default ON)
if(DEFINED HOLOSCAN_ZOZO_MODULES_OPERATORS)
  if(NOT "Sam3OverlayOp" IN_LIST HOLOSCAN_ZOZO_MODULES_OPERATORS)
    set(_overlay_default OFF)
  endif()
endif()
option(
  HOLOSCAN_SAM3_BUILD_OVERLAY
  "Build Sam3OverlayOp (needs Holoscan Visualization)"
  ${_overlay_default}
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# HoloInfer, the SDK's TensorRT inference library, comes with its inference component.
set(_holoscan_components core sensor_io inference)
if(HOLOSCAN_SAM3_BUILD_OVERLAY)
  list(APPEND _holoscan_components visualization)
endif()
if(HOLOSCAN_SAM3_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# holoscan::sam3: Sam3SegmentationOp, its kernels, the prompt table, and the stub model,
# synthetic scene and host references for tests.
add_library(
  holoscan_sam3
  src/kernels.cu
  src/prompts.cpp
  src/reference.cpp
  src/sam3_segmentation_op.cpp
  src/stub_model.cpp
  src/synthetic_scene.cpp
)
add_library(holoscan::sam3 ALIAS holoscan_sam3)
# Without fused multiply-adds the kernels compute bit for bit what the host references
# compute from the same expressions (pixel_math.hpp), so the tests compare them exactly.
set_source_files_properties(src/kernels.cu PROPERTIES COMPILE_OPTIONS -fmad=false)
target_compile_features(holoscan_sam3 PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_compile_options(
  holoscan_sam3
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)
target_include_directories(
  holoscan_sam3
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(
  holoscan_sam3
  PUBLIC holoscan::core holoscan::sensor_io CUDA::cudart
  PRIVATE holoscan::infer
)
set_target_properties(
  holoscan_sam3
  PROPERTIES
    EXPORT_NAME sam3
    OUTPUT_NAME holoscan_sam3
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
set(_targets holoscan_sam3)

# holoscan::sam3_overlay: Sam3OverlayOp, which draws instances as HoloViz layers.
if(HOLOSCAN_SAM3_BUILD_OVERLAY)
  add_library(holoscan_sam3_overlay src/sam3_overlay_op.cpp)
  add_library(holoscan::sam3_overlay ALIAS holoscan_sam3_overlay)
  target_compile_features(holoscan_sam3_overlay PUBLIC cxx_std_20)
  target_link_libraries(
    holoscan_sam3_overlay
    PUBLIC holoscan_sam3
    PRIVATE holoscan::visualization::op_holoviz
  )
  set_target_properties(
    holoscan_sam3_overlay
    PROPERTIES
      EXPORT_NAME sam3_overlay
      OUTPUT_NAME holoscan_sam3_overlay
      POSITION_INDEPENDENT_CODE ON
      VERSION ${PROJECT_VERSION}
      SOVERSION ${PROJECT_VERSION_MAJOR}
  )
  list(APPEND _targets holoscan_sam3_overlay)
endif()

# Installation: libraries, public headers, the model preparation tool, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-sam3")
set(_share_dir "${CMAKE_INSTALL_DATAROOTDIR}/holoscan-sam3")
install(
  TARGETS ${_targets}
  EXPORT holoscan-sam3-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/sam3 DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(PROGRAMS tools/prepare_model.py DESTINATION "${_share_dir}")
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-sam3"
)
install(
  EXPORT holoscan-sam3-targets
  FILE holoscan-sam3-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
set(HOLOSCAN_SAM3_HOLOSCAN_COMPONENTS core sensor_io inference)
if(HOLOSCAN_SAM3_BUILD_OVERLAY)
  list(APPEND HOLOSCAN_SAM3_HOLOSCAN_COMPONENTS visualization)
endif()
set(HOLOSCAN_SAM3_PREPARE_MODEL "${_share_dir}/prepare_model.py")
configure_package_config_file(
  cmake/holoscan-sam3-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-sam3-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  PATH_VARS HOLOSCAN_SAM3_PREPARE_MODEL
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-sam3-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-sam3-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-sam3-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_SAM3_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_sam3)
  add_subdirectory(pkg/holoscan-sam3)
endif()
