# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Every test needs the SDK, CUDA and a GPU. The tests check against
# CPU references and the synthetic terrain's host functions, so they must not contract either.
add_library(height_map_test_flags INTERFACE)
target_compile_options(
  height_map_test_flags
  INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)

add_executable(height_map_mapper_test height_map_test.cu)
target_link_libraries(height_map_mapper_test PRIVATE holoscan::height_map height_map_test_flags)

add_executable(height_map_traversability_test traversability_test.cu)
target_link_libraries(
  height_map_traversability_test
  PRIVATE holoscan::height_map height_map_test_flags
)

add_executable(height_map_operators_test operators_test.cpp)
target_link_libraries(
  height_map_operators_test
  PRIVATE holoscan::height_map holoscan::testing height_map_test_flags
)

# Times the map update at 640x480 and 1280x720 into 200x200 and 500x500 maps; CI runs it
# briefly, to keep it working.
add_executable(height_map_benchmark benchmark.cu)
target_link_libraries(height_map_benchmark PRIVATE holoscan::height_map)

add_test(NAME height_map_mapper COMMAND height_map_mapper_test)
add_test(NAME height_map_traversability COMMAND height_map_traversability_test)
add_test(NAME height_map_operators COMMAND height_map_operators_test)
add_test(NAME height_map_benchmark COMMAND height_map_benchmark --iterations 3)
set_tests_properties(
  height_map_benchmark
  height_map_mapper
  height_map_operators
  height_map_traversability
  PROPERTIES LABELS "height_map;gpu" TIMEOUT 900
)
