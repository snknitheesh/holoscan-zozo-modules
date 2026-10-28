# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the calibration files, the geometry and
# the backends' rules, built from the module's SDK-free sources. Built on their own by
# the hosted CI job, and as part of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_rectify_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_library(
  rectify_host_core
  STATIC
  "${_module}/src/backend_rules.cpp"
  "${_module}/src/calibration_io.cpp"
  "${_module}/src/camera_model.cpp"
  "${_module}/src/json.cpp"
  "${_module}/src/rectification.cpp"
  "${_module}/src/synthetic_stereo.cpp"
)
target_include_directories(
  rectify_host_core
  PUBLIC "${_module}/include" "${CMAKE_CURRENT_SOURCE_DIR}/.."
  PRIVATE "${_module}/src"
)
target_compile_features(rectify_host_core PUBLIC cxx_std_20)
set_target_properties(rectify_host_core PROPERTIES CXX_EXTENSIONS OFF)

foreach(_test calibration geometry backend_rules)
  add_executable(rectify_${_test}_test ${_test}_test.cpp)
  target_link_libraries(rectify_${_test}_test PRIVATE rectify_host_core)
  set_target_properties(rectify_${_test}_test PROPERTIES CXX_EXTENSIONS OFF)
  add_test(NAME rectify_${_test} COMMAND rectify_${_test}_test)
  set_tests_properties(rectify_${_test} PROPERTIES LABELS "rectify;host" TIMEOUT 300)
endforeach()
