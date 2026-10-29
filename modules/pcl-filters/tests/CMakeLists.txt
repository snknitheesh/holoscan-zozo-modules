# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Every test needs the SDK, CUDA and a GPU. The tests check against
# CPU references and the synthetic scene's host functions, so they must not contract either.
add_library(pcl_filters_test_flags INTERFACE)
target_compile_options(
  pcl_filters_test_flags
  INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)

add_executable(pcl_filters_filters_test filters_test.cu)
target_link_libraries(pcl_filters_filters_test PRIVATE holoscan::pcl_filters pcl_filters_test_flags)

add_executable(pcl_filters_operators_test operators_test.cpp)
target_link_libraries(
  pcl_filters_operators_test
  PRIVATE holoscan::pcl_filters holoscan::testing pcl_filters_test_flags
)

# Times each filter at 640x480 and 1280x720; CI runs it briefly, to keep it working.
add_executable(pcl_filters_benchmark benchmark.cu)
target_link_libraries(pcl_filters_benchmark PRIVATE holoscan::pcl_filters)

add_test(NAME pcl_filters_filters COMMAND pcl_filters_filters_test)
add_test(NAME pcl_filters_operators COMMAND pcl_filters_operators_test)
add_test(NAME pcl_filters_benchmark COMMAND pcl_filters_benchmark --iterations 3)
set_tests_properties(
  pcl_filters_benchmark
  pcl_filters_filters
  pcl_filters_operators
  PROPERTIES LABELS "pcl_filters;gpu" TIMEOUT 900
)
