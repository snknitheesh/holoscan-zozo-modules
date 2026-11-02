# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Every test needs the SDK, CUDA and a GPU. The tests check against
# CPU references and the synthetic scene's host functions, so they must not contract either.
add_library(normals_test_flags INTERFACE)
target_compile_options(
  normals_test_flags
  INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)

add_executable(normals_normals_test normals_test.cu)
target_link_libraries(normals_normals_test PRIVATE holoscan::normals normals_test_flags)

add_executable(normals_planes_test planes_test.cu)
target_link_libraries(normals_planes_test PRIVATE holoscan::normals normals_test_flags)

add_executable(normals_operators_test operators_test.cpp)
target_link_libraries(
  normals_operators_test
  PRIVATE holoscan::normals holoscan::testing normals_test_flags
)

# Times each step at 640x480 and 1280x720; CI runs it briefly, to keep it working.
add_executable(normals_benchmark benchmark.cu)
target_link_libraries(normals_benchmark PRIVATE holoscan::normals)

add_test(NAME normals_normals COMMAND normals_normals_test)
add_test(NAME normals_planes COMMAND normals_planes_test)
add_test(NAME normals_operators COMMAND normals_operators_test)
add_test(NAME normals_benchmark COMMAND normals_benchmark --iterations 3)
set_tests_properties(
  normals_benchmark
  normals_normals
  normals_operators
  normals_planes
  PROPERTIES LABELS "normals;gpu" TIMEOUT 900
)
