# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the webcam's YUYV to RGB8 conversion,
# which is header-only. Built on their own by the hosted CI job, and as part of the
# demo's tests.
cmake_minimum_required(VERSION 4.0)
project(sam3_segmentation_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

add_executable(sam3_segmentation_yuyv_test yuyv_test.cpp)
target_include_directories(
  sam3_segmentation_yuyv_test
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src" "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(sam3_segmentation_yuyv_test PRIVATE cxx_std_20)
add_test(NAME sam3_segmentation_yuyv COMMAND sam3_segmentation_yuyv_test)
set_tests_properties(sam3_segmentation_yuyv PROPERTIES LABELS "sam3_segmentation;host" TIMEOUT 60)
