# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The recording reader's tests, which need no Holoscan SDK, CUDA or GPU, built from the
# SDK-free sources of the demo and of holoscan-messages. Built on their own by the hosted CI
# job, and as part of the demo's tests.
cmake_minimum_required(VERSION 4.0)
project(stereo_disparity_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_messages "${CMAKE_CURRENT_SOURCE_DIR}/../../../../modules/messages")
add_executable(
  stereo_disparity_recording_test
  recording_test.cpp
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/recording.cpp"
  "${_messages}/src/camera_info.cpp"
  "${_messages}/src/camera_info_yaml.cpp"
  "${_messages}/src/yaml_reader.cpp"
)
target_include_directories(
  stereo_disparity_recording_test
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${_messages}/include"
    "${_messages}/src"
)
target_compile_features(stereo_disparity_recording_test PRIVATE cxx_std_20)
set_target_properties(stereo_disparity_recording_test PROPERTIES CXX_EXTENSIONS OFF)
add_test(NAME stereo_disparity_recording COMMAND stereo_disparity_recording_test)
set_tests_properties(
  stereo_disparity_recording
  PROPERTIES LABELS "stereo_disparity;host" TIMEOUT 60
)
