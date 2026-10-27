# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the PLY reader and writer and the PGM
# and PPM writers. Built on their own by the hosted CI job, and as part of the demo's
# tests.
cmake_minimum_required(VERSION 4.0)
project(point_cloud_to_depth_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

add_executable(pcd_cloud_io_test cloud_io_test.cpp ../../src/cloud_io.cpp)
target_include_directories(
  pcd_cloud_io_test
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src" "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(pcd_cloud_io_test PRIVATE cxx_std_20)
set_target_properties(pcd_cloud_io_test PROPERTIES CXX_EXTENSIONS OFF)
add_test(NAME pcd_cloud_io COMMAND pcd_cloud_io_test "${CMAKE_CURRENT_BINARY_DIR}/cloud_io_files")
set_tests_properties(pcd_cloud_io PROPERTIES LABELS "point_cloud_to_depth;host" TIMEOUT 60)
