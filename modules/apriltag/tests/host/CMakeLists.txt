# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the overlay's axis drawing, which is
# header-only. Built on their own by the hosted CI job, and as part of the module's
# full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_apriltag_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

add_executable(apriltag_tag_axes_test tag_axes_test.cpp)
target_include_directories(
  apriltag_tag_axes_test
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(apriltag_tag_axes_test PRIVATE cxx_std_20)
add_test(NAME apriltag_tag_axes COMMAND apriltag_tag_axes_test)
set_tests_properties(apriltag_tag_axes PROPERTIES LABELS "apriltag;host" TIMEOUT 60)
