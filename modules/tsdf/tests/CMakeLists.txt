# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Every test needs the SDK, CUDA and a GPU. The tests check against
# CPU references and the synthetic scene's host functions, so they must not contract either.
add_library(tsdf_test_flags INTERFACE)
target_compile_options(
  tsdf_test_flags
  INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)
# The CPU reference uses the module's marching cubes table.
target_include_directories(tsdf_test_flags INTERFACE "${PROJECT_SOURCE_DIR}/src")

add_executable(tsdf_volume_test volume_test.cpp)
target_link_libraries(tsdf_volume_test PRIVATE holoscan::tsdf tsdf_test_flags)

add_executable(tsdf_operators_test operators_test.cpp)
target_link_libraries(tsdf_operators_test PRIVATE holoscan::tsdf holoscan::testing tsdf_test_flags)

# Times integration, ray casting and meshing at 640x480 and 1280x720 for 256^3 and 512^3
# volumes; CI runs it briefly on 256^3, to keep it working.
add_executable(tsdf_benchmark benchmark.cpp)
target_link_libraries(tsdf_benchmark PRIVATE holoscan::tsdf tsdf_test_flags)

add_test(NAME tsdf_volume COMMAND tsdf_volume_test)
add_test(NAME tsdf_operators COMMAND tsdf_operators_test)
add_test(
  NAME tsdf_benchmark
  COMMAND tsdf_benchmark --iterations 3 --max-dims 256
)
set_tests_properties(
  tsdf_benchmark
  tsdf_operators
  tsdf_volume
  PROPERTIES LABELS "tsdf;gpu" TIMEOUT 900
)
