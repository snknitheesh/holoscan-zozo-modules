# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Everything but the host tests needs the SDK, CUDA and a GPU,
# and makes its own data: a synthetic world rendered by virtual cameras.

# The tests compare against the CPU reference, so they must not contract either.
add_library(stitching_test_flags INTERFACE)
target_compile_options(stitching_test_flags INTERFACE -ffp-contract=off)

add_executable(stitching_stitcher_test stitcher_test.cpp)
target_link_libraries(stitching_stitcher_test PRIVATE holoscan::stitching stitching_test_flags)

add_executable(stitching_operator_test operator_test.cpp)
target_link_libraries(
  stitching_operator_test
  PRIVATE holoscan::stitching holoscan::testing stitching_test_flags
)

# Not a test: times the tables and each frame at the demo's sizes (see DEVELOPER.md).
add_executable(stitching_benchmark benchmark.cpp)
target_link_libraries(stitching_benchmark PRIVATE holoscan::stitching)

add_test(NAME stitching_stitcher COMMAND stitching_stitcher_test)
add_test(NAME stitching_operator COMMAND stitching_operator_test)
set_tests_properties(
  stitching_stitcher
  stitching_operator
  PROPERTIES LABELS "stitching;gpu" TIMEOUT 900 SKIP_RETURN_CODE 77
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
