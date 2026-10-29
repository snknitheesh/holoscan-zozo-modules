# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_stereo_camera.
option(PKG_holoscan_stereo_camera "Generate the Debian package configuration" OFF)

project(
  holoscan_stereo_camera
  VERSION 1.0.0
  DESCRIPTION "Stereo cameras as Holoscan sources: RealSense, side-by-side V4L2 and recordings"
  LANGUAGES CXX
)
set(CMAKE_CXX_EXTENSIONS OFF)

# Built on its own, the usual CMake switches; inside another project (the repository root,
# FetchContent, add_subdirectory), shared libraries and no tests unless asked for, and the
# including project's own BUILD_SHARED_LIBS and BUILD_TESTING are left alone.
if(PROJECT_IS_TOP_LEVEL)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
  option(BUILD_TESTING "Build the module's tests" OFF)
  set(HOLOSCAN_STEREO_CAMERA_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(
    HOLOSCAN_STEREO_CAMERA_BUILD_TESTING
    "Build the module's tests in the including project"
    OFF
  )
endif()
option(
  HOLOSCAN_STEREO_CAMERA_WITH_REALSENSE
  "Build the realsense backend when librealsense is installed"
  ON
)
option(
  HOLOSCAN_STEREO_CAMERA_BUILD_RECORDER
  "Build and install stereo_camera_record"
  ${PROJECT_IS_TOP_LEVEL}
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_STEREO_CAMERA_BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

set(HOLOSCAN_STEREO_CAMERA_REALSENSE OFF)
if(HOLOSCAN_STEREO_CAMERA_WITH_REALSENSE)
  find_package(realsense2 2.50 QUIET CONFIG)
  if(realsense2_FOUND)
    set(HOLOSCAN_STEREO_CAMERA_REALSENSE ON)
    message(
      STATUS
      "holoscan-stereo-camera: building the realsense backend with librealsense ${realsense2_VERSION}"
    )
  else()
    message(
      STATUS
      "holoscan-stereo-camera: librealsense not found; building without the realsense backend"
    )
  endif()
endif()

# The calibration message, generated from its FlatBuffers schema into the public include tree.
set(_generated_include_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
holoscan_add_flatbuffer_schema(
  TARGET holoscan_stereo_camera_schema
  SCHEMA schemas/stereo_calibration.fbs
  OUTPUT_DIR "${_generated_include_dir}/holoscan/stereo_camera"
  MAX_SERIALIZED_SIZE 4096
)
get_target_property(
  _generated_headers
  holoscan_stereo_camera_schema
  HOLOSCAN_FLATBUFFER_GENERATED_HEADERS
)

# holoscan::stereo_camera: StereoCameraOp, its backends, the calibration and its message,
# recordings, side-by-side splitting and the synthetic camera.
add_library(
  holoscan_stereo_camera
  src/calibration.cpp
  src/calibration_message.cpp
  src/image.cpp
  src/json.cpp
  src/netpbm.cpp
  src/playback_source.cpp
  src/recording.cpp
  src/side_by_side.cpp
  src/source.cpp
  src/stereo_camera_op.cpp
  src/synthetic.cpp
  src/v4l2_source.cpp
)
add_library(holoscan::stereo_camera ALIAS holoscan_stereo_camera)
target_compile_features(holoscan_stereo_camera PUBLIC cxx_std_20)
target_include_directories(
  holoscan_stereo_camera
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${_generated_include_dir}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(
  holoscan_stereo_camera
  PUBLIC
    holoscan::core
    holoscan::sensor_io
    CUDA::cudart
    "$<BUILD_INTERFACE:holoscan_stereo_camera_schema>"
)
target_compile_definitions(
  holoscan_stereo_camera
  PRIVATE HOLOSCAN_STEREO_CAMERA_WITH_REALSENSE=$<BOOL:${HOLOSCAN_STEREO_CAMERA_REALSENSE}>
)
if(HOLOSCAN_STEREO_CAMERA_REALSENSE)
  target_sources(holoscan_stereo_camera PRIVATE src/realsense_source.cpp)
  target_link_libraries(holoscan_stereo_camera PRIVATE realsense2::realsense2)
endif()
set_target_properties(
  holoscan_stereo_camera
  PROPERTIES
    EXPORT_NAME stereo_camera
    OUTPUT_NAME holoscan_stereo_camera
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(HOLOSCAN_STEREO_CAMERA_BUILD_RECORDER)
  add_subdirectory(tools)
  install(TARGETS stereo_camera_record RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

# Installation: the library, public and generated headers, the schema, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-stereo-camera")
install(
  TARGETS holoscan_stereo_camera
  EXPORT holoscan-stereo-camera-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/stereo_camera DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(
  FILES ${_generated_headers}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/stereo_camera"
)
install(
  FILES schemas/stereo_calibration.fbs
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/holoscan-stereo-camera/schemas"
)
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-stereo-camera"
)
install(
  EXPORT holoscan-stereo-camera-targets
  FILE holoscan-stereo-camera-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-stereo-camera-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-stereo-camera-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-stereo-camera-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-stereo-camera-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-stereo-camera-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_STEREO_CAMERA_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_stereo_camera)
  add_subdirectory(pkg/holoscan-stereo-camera)
endif()
