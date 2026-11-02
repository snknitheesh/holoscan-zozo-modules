# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: every accuracy case against the CPU
# reference and the evaluator, which are plain C++. Built on their own by the hosted CI
# job, and as part of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_image_quality_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_executable(
  image_quality_host_test
  quality_host_test.cpp
  "${_module}/src/quality.cpp"
  "${_module}/src/reference.cpp"
  "${_module}/src/synthetic.cpp"
)
target_include_directories(
  image_quality_host_test
  PRIVATE "${_module}/include" "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(image_quality_host_test PRIVATE cxx_std_20)
add_test(NAME image_quality_host COMMAND image_quality_host_test)
set_tests_properties(image_quality_host PROPERTIES LABELS "image_quality;host" TIMEOUT 300)
