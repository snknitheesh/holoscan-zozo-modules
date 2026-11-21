# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the camera's schedule and the flags each
# frame must get, against the module's CPU reference. Built on their own by the hosted CI
# job, and as part of the demo's tests.
cmake_minimum_required(VERSION 4.0)
project(image_quality_app_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../../../../modules/image-quality")
add_executable(
  image_quality_segments_test
  segments_test.cpp
  ../../src/segments.cpp
  "${_module}/src/quality.cpp"
  "${_module}/src/reference.cpp"
  "${_module}/src/synthetic.cpp"
)
target_include_directories(
  image_quality_segments_test
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src" "${_module}/include"
)
target_compile_features(image_quality_segments_test PRIVATE cxx_std_20)
set_target_properties(image_quality_segments_test PROPERTIES CXX_EXTENSIONS OFF)
add_test(NAME image_quality_app_segments COMMAND image_quality_segments_test)
set_tests_properties(image_quality_app_segments PROPERTIES LABELS "image_quality;host" TIMEOUT 300)
