# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the settings, and the CPU backend, which is
# plain C++, against an independent bilinear reference and the synthetic camera's truth.
# Built on their own by the hosted CI job, and as part of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_isp_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_isp "${CMAKE_CURRENT_SOURCE_DIR}/../..")
# The module's SDK-free sources, compiled as the module compiles them: without contracting.
add_library(
  isp_host_core
  STATIC
  "${_isp}/src/cpu_pipeline.cpp"
  "${_isp}/src/isp_params.cpp"
  "${_isp}/src/synthetic_raw.cpp"
)
target_include_directories(isp_host_core PUBLIC "${_isp}/include" "${_isp}/src" "${_isp}/tests")
target_compile_features(isp_host_core PUBLIC cxx_std_20)
target_compile_options(isp_host_core PUBLIC -ffp-contract=off)

add_executable(isp_params_test params_test.cpp)
target_link_libraries(isp_params_test PRIVATE isp_host_core)
add_test(NAME isp_params COMMAND isp_params_test)

add_executable(isp_cpu_reference_test cpu_reference_test.cpp)
target_link_libraries(isp_cpu_reference_test PRIVATE isp_host_core)
add_test(NAME isp_cpu_reference COMMAND isp_cpu_reference_test)

set_tests_properties(isp_cpu_reference isp_params PROPERTIES LABELS "isp;host" TIMEOUT 300)
