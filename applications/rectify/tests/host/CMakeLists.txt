# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: reading a recording's index and PGM eyes.
# Built on their own by the hosted CI job, and as part of the demo's tests.
cmake_minimum_required(VERSION 4.0)
project(rectify_demo_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

add_executable(rectify_recording_test recording_test.cpp ../../src/recording.cpp)
target_include_directories(rectify_recording_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
target_compile_features(rectify_recording_test PRIVATE cxx_std_20)
set_target_properties(rectify_recording_test PROPERTIES CXX_EXTENSIONS OFF)
add_test(
  NAME rectify_recording
  COMMAND rectify_recording_test "${CMAKE_CURRENT_BINARY_DIR}/recording_files"
)
set_tests_properties(rectify_recording PROPERTIES LABELS "rectify;host" TIMEOUT 60)
