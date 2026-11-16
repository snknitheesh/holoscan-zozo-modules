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

# The operator on the local recordings' depth images, when HOLOSCAN_TEST_DATA names them;
# skipped otherwise, as in CI. Their depth is in units of 0.000452 m.
add_executable(clustering_real_data_test real_data_test.cpp)
target_link_libraries(
  clustering_real_data_test
  PRIVATE holoscan::clustering holoscan::testing clustering_test_flags
)

add_test(NAME clustering_clusterer COMMAND clustering_clusterer_test)
add_test(NAME clustering_operator COMMAND clustering_operator_test)
add_test(NAME clustering_benchmark COMMAND clustering_benchmark --iterations 3)
add_test(NAME clustering_real_data COMMAND clustering_real_data_test --depth-scale 0.000452)
set_tests_properties(clustering_real_data PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(
  clustering_benchmark
  clustering_clusterer
  clustering_operator
  clustering_real_data
  PROPERTIES LABELS "clustering;gpu" TIMEOUT 900
)

# The clusterer's whole test under compute-sanitizer: memcheck with a leak check, racecheck,
# initcheck and synccheck, labeled "sanitizer" (ctest -L sanitizer, or -LE sanitizer to leave
# them out). Any error the tool reports fails the test.
find_program(CLUSTERING_COMPUTE_SANITIZER compute-sanitizer HINTS "${CUDAToolkit_BIN_DIR}")
if(CLUSTERING_COMPUTE_SANITIZER)
  foreach(_tool memcheck racecheck initcheck synccheck)
    set(_options --tool ${_tool} --error-exitcode 1)
    if(_tool STREQUAL "memcheck")
      list(APPEND _options --leak-check full)
    endif()
    add_test(
      NAME clustering_clusterer_${_tool}
      COMMAND "${CLUSTERING_COMPUTE_SANITIZER}" ${_options} $<TARGET_FILE:clustering_clusterer_test>
    )
    set_tests_properties(
      clustering_clusterer_${_tool}
      PROPERTIES LABELS "clustering;gpu;sanitizer" TIMEOUT 1200
    )
  endforeach()
endif()
