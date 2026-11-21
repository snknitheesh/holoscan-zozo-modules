# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the settings, the auto backend's choice and
# the tone map and fusion references. Built on their own by the hosted CI job, and as part
# of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_denoise_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_denoise "${CMAKE_CURRENT_SOURCE_DIR}/../..")
# The module's SDK-free sources.
add_library(
  denoise_host_core
  STATIC
  "${_denoise}/src/denoise_params.cpp"
  "${_denoise}/src/fusion_params.cpp"
)
target_include_directories(denoise_host_core PUBLIC "${_denoise}/include" "${_denoise}/tests")
target_compile_features(denoise_host_core PUBLIC cxx_std_20)

add_executable(denoise_params_test params_test.cpp)
target_link_libraries(denoise_params_test PRIVATE denoise_host_core)
add_test(NAME denoise_params COMMAND denoise_params_test)
set_tests_properties(denoise_params PROPERTIES LABELS "denoise;host" TIMEOUT 60)
