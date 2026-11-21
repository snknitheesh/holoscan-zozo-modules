# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA, VPI or GPU: the backends' parameter and size
# rules, the auto backend's choice, the camera info rules and the depth formula, built from
# the SDK-free sources of this module and of holoscan-messages, which it depends on. Built
# on their own by the hosted CI job, and as part of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_stereo_disparity_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(_messages "${_module}/../messages")
add_executable(
  stereo_disparity_options_test
  options_test.cpp
  "${_module}/src/camera_info.cpp"
  "${_module}/src/disparity_options.cpp"
  "${_messages}/src/camera_info.cpp"
  "${_messages}/src/synthetic_calibration.cpp"
)
target_include_directories(
  stereo_disparity_options_test
  PRIVATE
    "${_module}/include"
    "${_messages}/include"
    "${_messages}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(stereo_disparity_options_test PRIVATE cxx_std_20)
set_target_properties(stereo_disparity_options_test PROPERTIES CXX_EXTENSIONS OFF)
add_test(NAME stereo_disparity_options COMMAND stereo_disparity_options_test)
set_tests_properties(stereo_disparity_options PROPERTIES LABELS "stereo_disparity;host" TIMEOUT 60)
