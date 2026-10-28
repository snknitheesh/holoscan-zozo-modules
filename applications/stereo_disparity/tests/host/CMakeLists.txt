# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The recording reader's tests, which need no Holoscan SDK, CUDA or GPU. Built on their
# own by the hosted CI job, and as part of the demo's tests.
cmake_minimum_required(VERSION 4.0)
project(stereo_disparity_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

add_executable(
  stereo_disparity_recording_test
  recording_test.cpp
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/recording.cpp"
)
target_include_directories(
  stereo_disparity_recording_test
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src" "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(stereo_disparity_recording_test PRIVATE cxx_std_20)
add_test(NAME stereo_disparity_recording COMMAND stereo_disparity_recording_test)
set_tests_properties(
  stereo_disparity_recording
  PROPERTIES LABELS "stereo_disparity;host" TIMEOUT 60
)
