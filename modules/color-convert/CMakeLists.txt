# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_color_convert; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_color_convert "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_color_convert)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_color_convert
  VERSION 1.0.0
  DESCRIPTION "Colour conversion and resizing of camera images on the GPU or the VIC"
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
  set(HOLOSCAN_COLOR_CONVERT_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(
    HOLOSCAN_COLOR_CONVERT_BUILD_TESTING
    "Build the module's tests in the including project"
    OFF
  )
endif()
option(HOLOSCAN_COLOR_CONVERT_WITH_VPI "Build the vic backend when VPI 4 is installed" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_COLOR_CONVERT_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

set(HOLOSCAN_COLOR_CONVERT_VPI OFF)
if(HOLOSCAN_COLOR_CONVERT_WITH_VPI)
  find_package(vpi 4 QUIET CONFIG PATHS /opt/nvidia/vpi4)
  if(vpi_FOUND)
    set(HOLOSCAN_COLOR_CONVERT_VPI ON)
    message(STATUS "holoscan-color-convert: building the vic backend with VPI ${vpi_VERSION}")
  else()
    message(STATUS "holoscan-color-convert: VPI 4 not found; building without the vic backend")
  endif()
endif()

# holoscan::color_convert: the conversion engine (CUDA kernel and VIC), its CPU reference,
# ColorConvertOp and ResizeOp.
add_library(
  holoscan_color_convert
  src/color_convert_op.cpp
  src/conversion.cpp
  src/converter.cpp
  src/kernels.cu
  src/op_support.cpp
  src/reference.cpp
  src/resize_op.cpp
)
add_library(holoscan::color_convert ALIAS holoscan_color_convert)
if(HOLOSCAN_COLOR_CONVERT_VPI)
  target_sources(holoscan_color_convert PRIVATE src/vic.cpp)
  target_link_libraries(holoscan_color_convert PRIVATE vpi)
else()
  target_sources(holoscan_color_convert PRIVATE src/vic_unavailable.cpp)
endif()
# No floating-point contraction in the kernel and the reference, which share their
# arithmetic (src/kernel_math.hpp), so that both round every operation alike and agree to
# the byte.
set_source_files_properties(src/kernels.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
set_source_files_properties(src/reference.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_compile_features(holoscan_color_convert PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_compile_options(
  holoscan_color_convert
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_include_directories(
  holoscan_color_convert
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(holoscan_color_convert PUBLIC holoscan::core holoscan::sensor_io CUDA::cudart)
set_target_properties(
  holoscan_color_convert
  PROPERTIES
    EXPORT_NAME color_convert
    OUTPUT_NAME holoscan_color_convert
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, its headers and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-color-convert")
install(
  TARGETS holoscan_color_convert
  EXPORT holoscan-color-convert-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/color_convert DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-color-convert"
)
install(
  EXPORT holoscan-color-convert-targets
  FILE holoscan-color-convert-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-color-convert-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-color-convert-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-color-convert-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-color-convert-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-color-convert-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_COLOR_CONVERT_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_color_convert)
  add_subdirectory(pkg/holoscan-color-convert)
endif()
