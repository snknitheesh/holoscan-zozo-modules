# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: reading, writing and listing PGM files.
# Built on their own by the hosted CI job, and as part of the demo's tests.
cmake_minimum_required(VERSION 4.0)
project(depth_to_point_cloud_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

add_executable(depth_image_io_test image_io_test.cpp ../../src/image_io.cpp)
target_include_directories(
  depth_image_io_test
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src" "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(depth_image_io_test PRIVATE cxx_std_20)
set_target_properties(depth_image_io_test PROPERTIES CXX_EXTENSIONS OFF)
add_test(
  NAME depth_image_io
  COMMAND depth_image_io_test "${CMAKE_CURRENT_BINARY_DIR}/image_io_files"
)
set_tests_properties(depth_image_io PROPERTIES LABELS "depth_to_point_cloud;host" TIMEOUT 60)
