# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Everything but the host tests needs the SDK, CUDA and a GPU; the
# VIC parity test needs a Jetson and is skipped elsewhere; the real-data test needs the local
# recordings ($HOLOSCAN_TEST_DATA) and is skipped without them.

add_executable(color_convert_converter_test converter_test.cpp)
target_link_libraries(color_convert_converter_test PRIVATE holoscan::color_convert)

add_executable(color_convert_vic_parity_test vic_parity_test.cpp)
target_link_libraries(color_convert_vic_parity_test PRIVATE holoscan::color_convert)

add_executable(color_convert_operator_test operator_test.cpp)
target_link_libraries(color_convert_operator_test PRIVATE holoscan::color_convert holoscan::testing)

add_executable(color_convert_real_data_test real_data_test.cpp)
target_link_libraries(
  color_convert_real_data_test
  PRIVATE holoscan::color_convert holoscan::messages holoscan::testing
)

# Not a test: times each conversion at 1080p and 4K on every backend (see DEVELOPER.md).
add_executable(color_convert_benchmark benchmark.cpp)
target_link_libraries(color_convert_benchmark PRIVATE holoscan::color_convert)

add_test(NAME color_convert_converter COMMAND color_convert_converter_test)
add_test(NAME color_convert_vic_parity COMMAND color_convert_vic_parity_test)
add_test(NAME color_convert_operators COMMAND color_convert_operator_test)
add_test(NAME color_convert_real_data COMMAND color_convert_real_data_test)
set_tests_properties(
  color_convert_converter
  color_convert_vic_parity
  color_convert_operators
  color_convert_real_data
  PROPERTIES LABELS "color_convert;gpu" TIMEOUT 900 SKIP_RETURN_CODE 77
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
