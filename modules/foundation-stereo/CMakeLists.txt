# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_foundation_stereo; a package is built for every major
# GPU architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_foundation_stereo "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_foundation_stereo)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_foundation_stereo
  VERSION 1.0.0
  DESCRIPTION
    "Learned stereo disparity and depth with Fast-FoundationStereo and TensorRT on the GPU"
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
  set(HOLOSCAN_FOUNDATION_STEREO_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(
    HOLOSCAN_FOUNDATION_STEREO_BUILD_TESTING
    "Build the module's tests in the including project"
    OFF
  )
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# HoloInfer, the SDK's TensorRT inference library, comes with its inference component.
set(_holoscan_components core sensor_io inference)
if(HOLOSCAN_FOUNDATION_STEREO_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# The geometry message, generated from its FlatBuffers schema into the public include tree.
set(_generated_include_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
holoscan_add_flatbuffer_schema(
  TARGET holoscan_foundation_stereo_schema
  SCHEMA schemas/stereo_geometry.fbs
  OUTPUT_DIR "${_generated_include_dir}/holoscan/foundation_stereo"
  MAX_SERIALIZED_SIZE 256
)
get_target_property(
  _generated_headers
  holoscan_foundation_stereo_schema
  HOLOSCAN_FLATBUFFER_GENERATED_HEADERS
)

# holoscan::foundation_stereo: FoundationStereoOp, its kernels, the geometry message, and the
# stub model and synthetic scene for tests.
add_library(
  holoscan_foundation_stereo
  src/foundation_stereo_op.cpp
  src/geometry_message.cpp
  src/kernels.cu
  src/stereo_geometry.cpp
  src/stub_model.cpp
  src/synthetic_scene.cu
)
add_library(holoscan::foundation_stereo ALIAS holoscan_foundation_stereo)
# Without fused multiply-adds the kernels compute what the tests' host references compute
# from the same expressions, and the GPU renders the synthetic scene to the same bytes
# as its inline functions do on the host.
set_source_files_properties(
  src/kernels.cu
  src/synthetic_scene.cu
  PROPERTIES COMPILE_OPTIONS -fmad=false
)
target_compile_features(holoscan_foundation_stereo PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(
  holoscan_foundation_stereo
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${_generated_include_dir}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(
  holoscan_foundation_stereo
  PUBLIC
    holoscan::core
    holoscan::sensor_io
    CUDA::cudart
    "$<BUILD_INTERFACE:holoscan_foundation_stereo_schema>"
  PRIVATE holoscan::infer
)
set_target_properties(
  holoscan_foundation_stereo
  PROPERTIES
    EXPORT_NAME foundation_stereo
    OUTPUT_NAME holoscan_foundation_stereo
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, public and generated headers, the schema, the model tool, and
# the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-foundation-stereo")
set(_share_dir "${CMAKE_INSTALL_DATAROOTDIR}/holoscan-foundation-stereo")
install(
  TARGETS holoscan_foundation_stereo
  EXPORT holoscan-foundation-stereo-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(
  DIRECTORY include/holoscan/foundation_stereo
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan"
)
install(
  FILES ${_generated_headers}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/foundation_stereo"
)
install(FILES schemas/stereo_geometry.fbs DESTINATION "${_share_dir}/schemas")
install(PROGRAMS tools/prepare_model.py DESTINATION "${_share_dir}")
set(_doc_dir "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-foundation-stereo")
install(FILES ../../LICENSE NOTICE README.md DESTINATION "${_doc_dir}")
install(FILES tools/README.md DESTINATION "${_doc_dir}" RENAME prepare_model.md)
install(
  EXPORT holoscan-foundation-stereo-targets
  FILE holoscan-foundation-stereo-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
set(HOLOSCAN_FOUNDATION_STEREO_PREPARE_MODEL "${_share_dir}/prepare_model.py")
configure_package_config_file(
  cmake/holoscan-foundation-stereo-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-foundation-stereo-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  PATH_VARS HOLOSCAN_FOUNDATION_STEREO_PREPARE_MODEL
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-foundation-stereo-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-foundation-stereo-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-foundation-stereo-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_FOUNDATION_STEREO_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_foundation_stereo)
  add_subdirectory(pkg/holoscan-foundation-stereo)
endif()
