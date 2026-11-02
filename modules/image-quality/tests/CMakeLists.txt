# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Everything but the host test needs the SDK, CUDA and a GPU.

add_executable(image_quality_analyzer_test analyzer_test.cpp)
target_link_libraries(image_quality_analyzer_test PRIVATE holoscan::image_quality)

add_executable(image_quality_operator_test operator_test.cpp)
target_link_libraries(image_quality_operator_test PRIVATE holoscan::image_quality holoscan::testing)

add_executable(image_quality_benchmark benchmark.cpp)
target_link_libraries(image_quality_benchmark PRIVATE holoscan::image_quality)

add_test(NAME image_quality_analyzer COMMAND image_quality_analyzer_test)
add_test(NAME image_quality_operator COMMAND image_quality_operator_test)
add_test(NAME image_quality_benchmark COMMAND image_quality_benchmark)
set_tests_properties(
  image_quality_analyzer
  image_quality_operator
  image_quality_benchmark
  PROPERTIES LABELS "image_quality;gpu" TIMEOUT 900
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
