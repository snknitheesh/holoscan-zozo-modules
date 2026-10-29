# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_fiducials; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_fiducials "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_fiducials)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_fiducials
  VERSION 1.0.0
  DESCRIPTION "ArUco and ChArUco detection on the CPU or the GPU, and its HoloViz overlay"
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
  set(HOLOSCAN_FIDUCIALS_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_FIDUCIALS_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()
# The repository root lists the operators asked for; without FiducialOverlayOp, the overlay,
# and so Holoscan Visualization, is left out.
set(_overlay_default ON)
if(DEFINED HOLOSCAN_ZOZO_MODULES_OPERATORS)
  if(NOT "FiducialOverlayOp" IN_LIST HOLOSCAN_ZOZO_MODULES_OPERATORS)
    set(_overlay_default OFF)
  endif()
endif()
option(
  HOLOSCAN_FIDUCIALS_BUILD_OVERLAY
  "Build FiducialOverlayOp (needs Holoscan Visualization)"
  ${_overlay_default}
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_FIDUCIALS_BUILD_OVERLAY)
  list(APPEND _holoscan_components visualization)
endif()
if(HOLOSCAN_FIDUCIALS_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# The detections message, generated from its FlatBuffers schema into the public include tree.
set(_generated_include_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
holoscan_add_flatbuffer_schema(
  TARGET holoscan_fiducials_schema
  SCHEMA schemas/fiducial_detections.fbs
  OUTPUT_DIR "${_generated_include_dir}/holoscan/fiducials"
  MAX_SERIALIZED_SIZE 65536
)
get_target_property(
  _generated_headers
  holoscan_fiducials_schema
  HOLOSCAN_FLATBUFFER_GENERATED_HEADERS
)

# holoscan::fiducials: the dictionaries, both backends, poses, ChArUco boards, the synthetic
# scenes, ArucoDetectorOp and CharucoDetectorOp.
add_library(
  holoscan_fiducials
  src/aruco_detector_op.cpp
  src/charuco.cpp
  src/charuco_detector_op.cpp
  src/cpu_detector.cpp
  src/cuda_backend.cu
  src/detector_backend.cpp
  src/dictionary.cpp
  src/dictionary_codes.cpp
  src/fiducial_messages.cpp
  src/frame_detector.cpp
  src/gray_converter.cu
  src/pose.cpp
  src/synthetic_scene.cpp
  src/types.cpp
)
add_library(holoscan::fiducials ALIAS holoscan_fiducials)
target_compile_features(holoscan_fiducials PUBLIC cxx_std_20 PRIVATE cuda_std_20)
# No FMA contraction anywhere detection_core.hpp is compiled, device or host, so the backends
# round alike: its inline functions are emitted in several files, and the linker keeps one copy.
target_compile_options(
  holoscan_fiducials
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)
target_include_directories(
  holoscan_fiducials
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${_generated_include_dir}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(
  holoscan_fiducials
  PUBLIC
    holoscan::core
    holoscan::sensor_io
    CUDA::cudart
    "$<BUILD_INTERFACE:holoscan_fiducials_schema>"
)
set_target_properties(
  holoscan_fiducials
  PROPERTIES
    EXPORT_NAME fiducials
    OUTPUT_NAME holoscan_fiducials
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
set(_targets holoscan_fiducials)

# holoscan::fiducials_overlay: FiducialOverlayOp, which draws detections as HoloViz layers.
if(HOLOSCAN_FIDUCIALS_BUILD_OVERLAY)
  add_library(holoscan_fiducials_overlay src/fiducial_overlay_op.cpp src/image_output.cpp)
  add_library(holoscan::fiducials_overlay ALIAS holoscan_fiducials_overlay)
  target_compile_features(holoscan_fiducials_overlay PUBLIC cxx_std_20)
  target_include_directories(holoscan_fiducials_overlay PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(
    holoscan_fiducials_overlay
    PUBLIC holoscan_fiducials
    PRIVATE holoscan::visualization::op_holoviz
  )
  set_target_properties(
    holoscan_fiducials_overlay
    PROPERTIES
      EXPORT_NAME fiducials_overlay
      OUTPUT_NAME holoscan_fiducials_overlay
      POSITION_INDEPENDENT_CODE ON
      VERSION ${PROJECT_VERSION}
      SOVERSION ${PROJECT_VERSION_MAJOR}
  )
  list(APPEND _targets holoscan_fiducials_overlay)
endif()

# Installation: libraries, public and generated headers, the schema, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-fiducials")
install(
  TARGETS ${_targets}
  EXPORT holoscan-fiducials-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/fiducials DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(FILES ${_generated_headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/fiducials")
install(
  FILES schemas/fiducial_detections.fbs
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/holoscan-fiducials/schemas"
)
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-fiducials"
)
install(
  EXPORT holoscan-fiducials-targets
  FILE holoscan-fiducials-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
set(HOLOSCAN_FIDUCIALS_HOLOSCAN_COMPONENTS core sensor_io)
if(HOLOSCAN_FIDUCIALS_BUILD_OVERLAY)
  list(APPEND HOLOSCAN_FIDUCIALS_HOLOSCAN_COMPONENTS visualization)
endif()
configure_package_config_file(
  cmake/holoscan-fiducials-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-fiducials-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-fiducials-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-fiducials-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-fiducials-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_FIDUCIALS_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_fiducials)
  add_subdirectory(pkg/holoscan-fiducials)
endif()
