# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Every test needs the SDK, CUDA and a GPU. The tests check
# against the synthetic camera's host functions, so they must not contract either.
add_library(point_cloud_test_flags INTERFACE)
target_compile_options(point_cloud_test_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)

add_executable(point_cloud_deproject_test deproject_test.cu)
target_link_libraries(point_cloud_deproject_test PRIVATE
  holoscan::point_cloud point_cloud_test_flags)

add_executable(point_cloud_depth_to_point_cloud_op_test depth_to_point_cloud_op_test.cpp)
target_include_directories(point_cloud_depth_to_point_cloud_op_test PRIVATE
  "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(point_cloud_depth_to_point_cloud_op_test PRIVATE
  holoscan::point_cloud holoscan::testing point_cloud_test_flags)

add_test(NAME point_cloud_deproject COMMAND point_cloud_deproject_test)
add_test(NAME point_cloud_depth_to_point_cloud_op COMMAND point_cloud_depth_to_point_cloud_op_test)
set_tests_properties(point_cloud_deproject point_cloud_depth_to_point_cloud_op PROPERTIES
  LABELS "point_cloud;gpu" TIMEOUT 600)
