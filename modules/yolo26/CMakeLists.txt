# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_yolo26; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_yolo26 "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_yolo26)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(holoscan_yolo26
  VERSION 1.0.0
  DESCRIPTION "End-to-end YOLO26 object detection on the GPU with TensorRT, and its HoloViz overlay"
  LANGUAGES CXX CUDA)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTING "Build the module's tests" OFF)
option(HOLOSCAN_YOLO26_BUILD_OVERLAY "Build Yolo26OverlayOp (needs Holoscan Visualization)" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# HoloInfer, the SDK's TensorRT inference library, comes with its inference component.
set(_holoscan_components core sensor_io inference)
if(HOLOSCAN_YOLO26_BUILD_OVERLAY)
  list(APPEND _holoscan_components visualization)
endif()
if(BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# holoscan::yolo26: Yolo26DetectionOp, its letterbox and decode kernels, and the stub
# model and synthetic scenes for tests.
add_library(holoscan_yolo26
  src/decode.cu
  src/letterbox.cpp
  src/letterbox.cu
  src/stub_model.cpp
  src/synthetic_scene.cpp
  src/yolo26_detection_op.cpp)
add_library(holoscan::yolo26 ALIAS holoscan_yolo26)
# Without fused multiply-adds the kernels compute bit for bit what the host computes
# from the same expressions, so the tests compare them exactly.
set_source_files_properties(src/decode.cu src/letterbox.cu PROPERTIES COMPILE_OPTIONS -fmad=false)
target_compile_features(holoscan_yolo26 PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_compile_options(holoscan_yolo26 PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
target_include_directories(holoscan_yolo26
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(holoscan_yolo26
  PUBLIC
    holoscan::core
    holoscan::sensor_io
    CUDA::cudart
  PRIVATE
    holoscan::infer)
set_target_properties(holoscan_yolo26 PROPERTIES
  EXPORT_NAME yolo26
  OUTPUT_NAME holoscan_yolo26
  POSITION_INDEPENDENT_CODE ON
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})
set(_targets holoscan_yolo26)

# holoscan::yolo26_overlay: Yolo26OverlayOp, which draws detections as HoloViz layers.
if(HOLOSCAN_YOLO26_BUILD_OVERLAY)
  add_library(holoscan_yolo26_overlay src/yolo26_overlay_op.cpp)
  add_library(holoscan::yolo26_overlay ALIAS holoscan_yolo26_overlay)
  target_compile_features(holoscan_yolo26_overlay PUBLIC cxx_std_20)
  target_link_libraries(holoscan_yolo26_overlay
    PUBLIC holoscan_yolo26
    PRIVATE holoscan::visualization::op_holoviz)
  set_target_properties(holoscan_yolo26_overlay PROPERTIES
    EXPORT_NAME yolo26_overlay
    OUTPUT_NAME holoscan_yolo26_overlay
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
  list(APPEND _targets holoscan_yolo26_overlay)
endif()

# Installation: libraries, public headers, the model preparation tool, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-yolo26")
set(_share_dir "${CMAKE_INSTALL_DATAROOTDIR}/holoscan-yolo26")
install(TARGETS ${_targets}
  EXPORT holoscan-yolo26-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(DIRECTORY include/holoscan/yolo26 DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(PROGRAMS tools/prepare_model.py DESTINATION "${_share_dir}")
install(FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-yolo26")
install(EXPORT holoscan-yolo26-targets
  FILE holoscan-yolo26-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}")
set(HOLOSCAN_YOLO26_HOLOSCAN_COMPONENTS core sensor_io inference)
if(HOLOSCAN_YOLO26_BUILD_OVERLAY)
  list(APPEND HOLOSCAN_YOLO26_HOLOSCAN_COMPONENTS visualization)
endif()
set(HOLOSCAN_YOLO26_PREPARE_MODEL "${_share_dir}/prepare_model.py")
configure_package_config_file(cmake/holoscan-yolo26-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-yolo26-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  PATH_VARS HOLOSCAN_YOLO26_PREPARE_MODEL
  NO_SET_AND_CHECK_MACRO)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-yolo26-config-version.cmake"
  COMPATIBILITY SameMajorVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-yolo26-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-yolo26-config-version.cmake"
  DESTINATION "${_cmake_dir}")

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_yolo26)
  add_subdirectory(pkg/holoscan-yolo26)
endif()
