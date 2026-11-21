# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Every test but the host ones needs the SDK, CUDA and a GPU.

add_executable(denoise_denoiser_test denoiser_test.cpp)
target_link_libraries(denoise_denoiser_test PRIVATE holoscan::denoise)

add_executable(denoise_fusion_test fusion_test.cpp)
target_link_libraries(denoise_fusion_test PRIVATE holoscan::denoise)

add_executable(denoise_operator_test operator_test.cpp)
target_link_libraries(denoise_operator_test PRIVATE holoscan::denoise holoscan::testing)

# A graph with the auto backend under a plain holoscan::run(plan), in its own process.
add_executable(denoise_auto_run_test auto_run_test.cpp)
target_link_libraries(denoise_auto_run_test PRIVATE holoscan::denoise)

# The local recordings' RGB frames: runs when HOLOSCAN_TEST_DATA names the folder that holds
# them, and is skipped otherwise, as in CI.
add_executable(denoise_real_data_test real_data_test.cpp)
target_link_libraries(denoise_real_data_test PRIVATE holoscan::denoise)

# Times every method at 1920x1080 and 3840x2160; CI runs it briefly, to keep it working.
add_executable(denoise_benchmark benchmark.cpp)
target_link_libraries(denoise_benchmark PRIVATE holoscan::denoise)

add_test(NAME denoise_benchmark COMMAND denoise_benchmark --iterations 3)
add_test(NAME denoise_denoiser COMMAND denoise_denoiser_test)
add_test(NAME denoise_fusion COMMAND denoise_fusion_test)
add_test(NAME denoise_operators COMMAND denoise_operator_test)
add_test(NAME denoise_auto_run COMMAND denoise_auto_run_test)
add_test(NAME denoise_real_data COMMAND denoise_real_data_test)
set_tests_properties(denoise_real_data PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(
  denoise_auto_run
  denoise_benchmark
  denoise_denoiser
  denoise_fusion
  denoise_operators
  denoise_real_data
  PROPERTIES LABELS "denoise;gpu" TIMEOUT 900
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
