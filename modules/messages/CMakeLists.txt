# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_messages.
option(PKG_holoscan_messages "Generate the Debian package configuration" OFF)

project(
  holoscan_messages
  VERSION 1.0.0
  DESCRIPTION "Camera info, stereo camera info and point cloud messages for every module"
  LANGUAGES CXX
)
set(CMAKE_CXX_EXTENSIONS OFF)

# Built on its own, the usual CMake switches; inside another project (the repository root,
# FetchContent, add_subdirectory), shared libraries and no tests unless asked for, and the
# including project's own BUILD_SHARED_LIBS and BUILD_TESTING are left alone.
if(PROJECT_IS_TOP_LEVEL)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
  option(BUILD_TESTING "Build the module's tests" OFF)
  set(HOLOSCAN_MESSAGES_BUILD_TESTING ${BUILD_TESTING})
else()
  set(BUILD_SHARED_LIBS ON)
  option(HOLOSCAN_MESSAGES_BUILD_TESTING "Build the module's tests in the including project" OFF)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core)
find_package(CUDAToolkit REQUIRED)

# The three messages, generated from their FlatBuffers schemas into the public include
# tree. Their largest encoded sizes hold frame ids of kMaxFrameIdBytes (camera_info.hpp);
# for the point cloud, the size bounds its metadata, not its tensors.
set(_generated_include_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
set(_generated_dir "${_generated_include_dir}/holoscan/messages")
holoscan_add_flatbuffer_schema(
  TARGET holoscan_messages_camera_info_schema
  SCHEMA schemas/camera_info.fbs
  OUTPUT_DIR "${_generated_dir}"
  MAX_SERIALIZED_SIZE 1024
)
holoscan_add_flatbuffer_schema(
  TARGET holoscan_messages_stereo_camera_info_schema
  SCHEMA schemas/stereo_camera_info.fbs
  INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/schemas"
  OUTPUT_DIR "${_generated_dir}"
  MAX_SERIALIZED_SIZE 4096
)
holoscan_add_flatbuffer_schema(
  TARGET holoscan_messages_point_cloud_schema
  SCHEMA schemas/point_cloud.fbs
  OUTPUT_DIR "${_generated_dir}"
  MAX_SERIALIZED_SIZE 2048
)
set(
  _schema_targets
  holoscan_messages_camera_info_schema
  holoscan_messages_stereo_camera_info_schema
  holoscan_messages_point_cloud_schema
)
set(_generated_headers "")
foreach(_schema IN LISTS _schema_targets)
  get_target_property(_headers ${_schema} HOLOSCAN_FLATBUFFER_GENERATED_HEADERS)
  list(APPEND _generated_headers ${_headers})
endforeach()
list(REMOVE_DUPLICATES _generated_headers)

# holoscan::messages: the messages, their plain C++ values and checks, the camera_info YAML
# files, the consumer contract, the point cloud helpers and CameraInfoSourceOp.
add_library(
  holoscan_messages
  src/camera_info.cpp
  src/camera_info_input.cpp
  src/camera_info_source_op.cpp
  src/camera_info_yaml.cpp
  src/conversion.cpp
  src/point_cloud.cpp
  src/point_cloud_layout.cpp
  src/synthetic_calibration.cpp
  src/yaml_reader.cpp
)
add_library(holoscan::messages ALIAS holoscan_messages)
target_compile_features(holoscan_messages PUBLIC cxx_std_20)
target_include_directories(
  holoscan_messages
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${_generated_include_dir}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
# holoscan::schema carries the SDK schemas the generated headers include (header, tensor).
target_link_libraries(holoscan_messages PUBLIC holoscan::core holoscan::schema CUDA::cudart)
foreach(_schema IN LISTS _schema_targets)
  target_link_libraries(holoscan_messages PUBLIC "$<BUILD_INTERFACE:${_schema}>")
endforeach()
set_target_properties(
  holoscan_messages
  PROPERTIES
    EXPORT_NAME messages
    OUTPUT_NAME holoscan_messages
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation: the library, public and generated headers, the schemas, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-messages")
install(
  TARGETS holoscan_messages
  EXPORT holoscan-messages-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/messages DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(FILES ${_generated_headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/messages")
install(
  FILES schemas/camera_info.fbs schemas/stereo_camera_info.fbs schemas/point_cloud.fbs
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/holoscan-messages/schemas"
)
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-messages"
)
install(
  EXPORT holoscan-messages-targets
  FILE holoscan-messages-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
configure_package_config_file(
  cmake/holoscan-messages-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-messages-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-messages-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-messages-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-messages-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(HOLOSCAN_MESSAGES_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_messages)
  add_subdirectory(pkg/holoscan-messages)
endif()
