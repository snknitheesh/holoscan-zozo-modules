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

add_executable(pcl_filters_filters_test filters_test.cu launch_faults.cpp)
target_link_libraries(pcl_filters_filters_test PRIVATE holoscan::pcl_filters pcl_filters_test_flags)
# The test's kernel-launch and error functions take the place of the CUDA runtime's for the
# library, to make a launch fail.
set_target_properties(pcl_filters_filters_test PROPERTIES ENABLE_EXPORTS ON)

add_executable(pcl_filters_operators_test operators_test.cpp)
target_link_libraries(
  pcl_filters_operators_test
  PRIVATE holoscan::pcl_filters holoscan::testing pcl_filters_test_flags
)
# The test's cudaMemcpyAsync takes the place of the CUDA runtime's for the library, to make
# copies from pageable memory read it late.
set_target_properties(pcl_filters_operators_test PROPERTIES ENABLE_EXPORTS ON)

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
