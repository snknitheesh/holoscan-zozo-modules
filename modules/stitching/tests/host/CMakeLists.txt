# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the rig calibration and its files, the
# output geometry, and the CPU reference's tables, blend and gains. Built on their own by
# the hosted CI job, and as part of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_stitching_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# The module's SDK-free sources. Without contraction, as the module builds them.
add_library(
  stitching_host_core
  STATIC
  "${_module}/src/camera_rig.cpp"
  "${_module}/src/json.cpp"
  "${_module}/src/projection.cpp"
  "${_module}/src/remap.cpp"
)
target_include_directories(stitching_host_core PUBLIC "${_module}/include" PRIVATE "${_module}/src")
target_compile_features(stitching_host_core PUBLIC cxx_std_20)
target_compile_options(stitching_host_core PUBLIC -ffp-contract=off)

add_executable(stitching_calibration_test calibration_test.cpp)
target_include_directories(stitching_calibration_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(stitching_calibration_test PRIVATE stitching_host_core)
add_test(
  NAME stitching_calibration
  COMMAND stitching_calibration_test "${CMAKE_CURRENT_BINARY_DIR}/calibration_files"
)

add_executable(stitching_reference_test reference_test.cpp)
target_include_directories(stitching_reference_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(stitching_reference_test PRIVATE stitching_host_core)
add_test(NAME stitching_reference COMMAND stitching_reference_test)

set_tests_properties(
  stitching_calibration
  stitching_reference
  PROPERTIES LABELS "stitching;host" TIMEOUT 300
)
