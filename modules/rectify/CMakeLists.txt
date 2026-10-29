# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_rectify; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_rectify "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_rectify)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_rectify
  VERSION 1.0.0
  DESCRIPTION "Undistortion and stereo rectification on the GPU, the VIC or the PVA"
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
  set(HOLOSCAN_RECTIFY_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_RECTIFY_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()
option(HOLOSCAN_RECTIFY_WITH_VPI "Build the vpi-* backends when VPI 4 is installed" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_RECTIFY_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

set(HOLOSCAN_RECTIFY_VPI OFF)
if(HOLOSCAN_RECTIFY_WITH_VPI)
  find_package(vpi 4 QUIET CONFIG PATHS /opt/nvidia/vpi4)
  if(vpi_FOUND)
    set(HOLOSCAN_RECTIFY_VPI ON)
    message(STATUS "holoscan-rectify: building the vpi-* backends with VPI ${vpi_VERSION}")
  else()
    message(STATUS "holoscan-rectify: VPI 4 not found; building without the vpi-* backends")
  endif()
endif()

# The geometry message, generated from its FlatBuffers schema into the public include tree.
set(_generated_include_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
holoscan_add_flatbuffer_schema(
  TARGET holoscan_rectify_schema
  SCHEMA schemas/stereo_geometry.fbs
  OUTPUT_DIR "${_generated_include_dir}/holoscan/rectify"
  MAX_SERIALIZED_SIZE 1024
)
get_target_property(
  _generated_headers
  holoscan_rectify_schema
  HOLOSCAN_FLATBUFFER_GENERATED_HEADERS
)

# holoscan::rectify: the calibration and geometry, the remap backends, the synthetic scene,
# UndistortOp and StereoRectifyOp.
add_library(
  holoscan_rectify
  src/backend_rules.cpp
  src/calibration_io.cpp
  src/camera_model.cpp
  src/cuda_remap.cu
  src/json.cpp
  src/op_support.cpp
  src/rectification.cpp
  src/remapper.cpp
  src/stereo_rectify_op.cpp
  src/synthetic_stereo.cpp
  src/undistort_op.cpp
)
add_library(holoscan::rectify ALIAS holoscan_rectify)
target_compile_features(holoscan_rectify PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(
  holoscan_rectify
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${_generated_include_dir}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(
  holoscan_rectify
  PUBLIC
    holoscan::core
    holoscan::sensor_io
    CUDA::cudart
    "$<BUILD_INTERFACE:holoscan_rectify_schema>"
)
target_compile_definitions(
  holoscan_rectify
  PRIVATE HOLOSCAN_RECTIFY_WITH_VPI=$<BOOL:${HOLOSCAN_RECTIFY_VPI}>
)
if(HOLOSCAN_RECTIFY_VPI)
  target_sources(holoscan_rectify PRIVATE src/vpi_remap.cpp)
  target_link_libraries(holoscan_rectify PRIVATE vpi)
endif()
set_target_properties(
  holoscan_rectify
  PROPERTIES
    EXPORT_NAME rectify
    OUTPUT_NAME holoscan_rectify
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, public and generated headers, the schema, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-rectify")
install(
  TARGETS holoscan_rectify
  EXPORT holoscan-rectify-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/rectify DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(FILES ${_generated_headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/rectify")
install(
  FILES schemas/stereo_geometry.fbs
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/holoscan-rectify/schemas"
)
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-rectify"
)
install(
  EXPORT holoscan-rectify-targets
  FILE holoscan-rectify-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-rectify-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-rectify-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-rectify-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-rectify-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-rectify-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_RECTIFY_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_rectify)
  add_subdirectory(pkg/holoscan-rectify)
endif()
