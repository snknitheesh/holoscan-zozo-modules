# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Every test needs the SDK, CUDA and a GPU. The tests evaluate the
# synthetic camera on the host and compare with what the GPU computed, so they must not
# contract either.
add_library(isp_test_flags INTERFACE)
target_compile_options(
  isp_test_flags
  INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)

add_executable(isp_pipeline_test pipeline_test.cpp)
target_link_libraries(isp_pipeline_test PRIVATE holoscan::isp isp_test_flags)

add_executable(isp_op_test isp_op_test.cpp)
target_link_libraries(isp_op_test PRIVATE holoscan::isp holoscan::testing isp_test_flags)

# Times each demosaic method at 1920x1080 and 3840x2160; CI runs it briefly, to keep it working.
add_executable(isp_benchmark benchmark.cpp)
target_link_libraries(isp_benchmark PRIVATE holoscan::isp)

add_test(NAME isp_pipeline COMMAND isp_pipeline_test)
add_test(NAME isp_op COMMAND isp_op_test)
add_test(
  NAME isp_benchmark
  COMMAND isp_benchmark --iterations 3 --cpu true --cpu-iterations 1
)
set_tests_properties(isp_op isp_pipeline PROPERTIES LABELS "isp;gpu" TIMEOUT 900)
set_tests_properties(isp_benchmark PROPERTIES LABELS "isp;gpu;benchmark" TIMEOUT 900)

# The pipeline's whole test under compute-sanitizer: memcheck with a leak check, racecheck,
# initcheck and synccheck, labeled "sanitizer" (ctest -L sanitizer, or -LE sanitizer to leave
# them out). Any error the tool reports fails the test.
find_program(ISP_COMPUTE_SANITIZER compute-sanitizer HINTS "${CUDAToolkit_BIN_DIR}")
if(ISP_COMPUTE_SANITIZER)
  foreach(_tool memcheck racecheck initcheck synccheck)
    set(_options --tool ${_tool} --error-exitcode 1)
    if(_tool STREQUAL "memcheck")
      list(APPEND _options --leak-check full)
    endif()
    add_test(
      NAME isp_pipeline_${_tool}
      COMMAND "${ISP_COMPUTE_SANITIZER}" ${_options} $<TARGET_FILE:isp_pipeline_test>
    )
    set_tests_properties(isp_pipeline_${_tool} PROPERTIES LABELS "isp;gpu;sanitizer" TIMEOUT 1200)
  endforeach()
endif()

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
