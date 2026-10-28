# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_recorder.
option(PKG_holoscan_recorder "Generate the Debian package configuration" OFF)

project(
  holoscan_recorder
  VERSION 1.0.0
  DESCRIPTION "Record the streams of a Holoscan graph to disk and replay them"
  LANGUAGES CXX
)
set(CMAKE_CXX_EXTENSIONS OFF)

# Built on its own, the usual CMake switches; inside another project (the repository root,
# FetchContent, add_subdirectory), shared libraries and no tests unless asked for, and the
# including project's own BUILD_SHARED_LIBS and BUILD_TESTING are left alone.
if(PROJECT_IS_TOP_LEVEL)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
  option(BUILD_TESTING "Build the module's tests" OFF)
  set(HOLOSCAN_RECORDER_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_RECORDER_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_RECORDER_BUILD_TESTING)
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# holoscan::recorder: RecorderOp and ReplayerOp, and the recording format they share
# (RecordingWriter, RecordingReader), which needs no SDK.
add_library(
  holoscan_recorder
  src/csv.cpp
  src/files.cpp
  src/format.cpp
  src/json.cpp
  src/recorder_op.cpp
  src/replayer_op.cpp
)
add_library(holoscan::recorder ALIAS holoscan_recorder)
target_compile_features(holoscan_recorder PUBLIC cxx_std_20)
target_include_directories(
  holoscan_recorder
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(holoscan_recorder PUBLIC holoscan::core holoscan::sensor_io CUDA::cudart)
set_target_properties(
  holoscan_recorder
  PROPERTIES
    EXPORT_NAME recorder
    OUTPUT_NAME holoscan_recorder
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, its public headers, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-recorder")
install(
  TARGETS holoscan_recorder
  EXPORT holoscan-recorder-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/recorder DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-recorder"
)
install(
  EXPORT holoscan-recorder-targets
  FILE holoscan-recorder-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-recorder-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-recorder-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-recorder-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-recorder-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-recorder-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_RECORDER_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_recorder)
  add_subdirectory(pkg/holoscan-recorder)
endif()
