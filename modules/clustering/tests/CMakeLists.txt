# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Every test needs the SDK, CUDA and a GPU. The tests check against
# CPU references and the synthetic scene's host functions, so they must not contract either.
add_library(clustering_test_flags INTERFACE)
target_compile_options(
  clustering_test_flags
  INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)

add_executable(clustering_clusterer_test clustering_test.cpp)
target_link_libraries(clustering_clusterer_test PRIVATE holoscan::clustering clustering_test_flags)

add_executable(clustering_operator_test operator_test.cpp)
target_link_libraries(
  clustering_operator_test
  PRIVATE holoscan::clustering holoscan::testing clustering_test_flags
)

# Times the clusterer at 640x480 and 1280x720; CI runs it briefly, to keep it working.
add_executable(clustering_benchmark benchmark.cpp)
target_link_libraries(clustering_benchmark PRIVATE holoscan::clustering)

add_test(NAME clustering_clusterer COMMAND clustering_clusterer_test)
add_test(NAME clustering_operator COMMAND clustering_operator_test)
add_test(NAME clustering_benchmark COMMAND clustering_benchmark --iterations 3)
set_tests_properties(
  clustering_benchmark
  clustering_clusterer
  clustering_operator
  PROPERTIES LABELS "clustering;gpu" TIMEOUT 900
)
