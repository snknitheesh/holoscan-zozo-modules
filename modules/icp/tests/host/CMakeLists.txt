# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the rigid transform helpers and the 6x6
# solve of each Gauss-Newton step, which are header-only. Built on their own by the hosted
# CI job, and as part of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_icp_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

add_executable(icp_host_test host_test.cpp)
target_include_directories(
  icp_host_test
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${CMAKE_CURRENT_SOURCE_DIR}/../../src"
)
target_compile_features(icp_host_test PRIVATE cxx_std_20)
add_test(NAME icp_host COMMAND icp_host_test)
set_tests_properties(icp_host PROPERTIES LABELS "icp;host" TIMEOUT 60)
