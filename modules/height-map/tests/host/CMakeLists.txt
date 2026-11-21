# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the map's and the traversability's settings
# and their rules, the grid, and whether a transform is rigid, built from the module's
# SDK-free source. Built on their own by the hosted CI job, and as part of the module's full
# suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_height_map_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_executable(height_map_params_test params_test.cpp "${_module}/src/params.cpp")
target_include_directories(
  height_map_params_test
  PRIVATE "${_module}/include" "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(height_map_params_test PRIVATE cxx_std_20)
set_target_properties(height_map_params_test PROPERTIES CXX_EXTENSIONS OFF)
add_test(NAME height_map_params COMMAND height_map_params_test)
set_tests_properties(height_map_params PROPERTIES LABELS "height_map;host" TIMEOUT 60)
