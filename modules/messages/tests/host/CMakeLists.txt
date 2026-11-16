# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the plain messages and their checks, the
# camera_info YAML files and the point cloud layout, built from the module's SDK-free
# sources. Built on their own by the hosted CI job, and as part of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_messages_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_library(
  messages_host_core
  STATIC
  "${_module}/src/camera_info.cpp"
  "${_module}/src/camera_info_yaml.cpp"
  "${_module}/src/point_cloud_layout.cpp"
  "${_module}/src/synthetic_calibration.cpp"
  "${_module}/src/yaml_reader.cpp"
)
target_include_directories(
  messages_host_core
  PUBLIC "${_module}/include" "${CMAKE_CURRENT_SOURCE_DIR}/.." "${_module}/src"
)
target_compile_features(messages_host_core PUBLIC cxx_std_20)
set_target_properties(messages_host_core PROPERTIES CXX_EXTENSIONS OFF)

foreach(_test camera_info yaml point_cloud_layout)
  add_executable(messages_${_test}_test ${_test}_test.cpp)
  target_link_libraries(messages_${_test}_test PRIVATE messages_host_core)
  set_target_properties(messages_${_test}_test PROPERTIES CXX_EXTENSIONS OFF)
  add_test(NAME messages_${_test} COMMAND messages_${_test}_test)
  set_tests_properties(messages_${_test} PROPERTIES LABELS "messages;host" TIMEOUT 300)
endforeach()
