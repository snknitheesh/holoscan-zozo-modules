# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Everything but the host tests needs the SDK, CUDA and a GPU; the codec
# tests skip (exit 77) where no backend can encode and decode, saying why, except in CI
# (HOLOSCAN_CI_PLATFORM set), where the lane's backend must work. The real-data test reads
# $HOLOSCAN_TEST_DATA and skips without it.

# The reordered test streams (see data/make_reordered_streams.sh).
set(_data "HOLOSCAN_CODEC_TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")

add_executable(codec_test codec_test.cpp)
target_link_libraries(codec_test PRIVATE holoscan::codec)
target_compile_definitions(codec_test PRIVATE ${_data})

add_executable(codec_operator_test operator_test.cpp)
target_link_libraries(codec_operator_test PRIVATE holoscan::codec holoscan::testing)
target_compile_definitions(codec_operator_test PRIVATE ${_data})

add_executable(codec_real_data_test real_data_test.cpp)
target_link_libraries(codec_real_data_test PRIVATE holoscan::codec)

add_test(NAME codec_round_trip COMMAND codec_test)
add_test(NAME codec_operators COMMAND codec_operator_test)
add_test(NAME codec_real_data COMMAND codec_real_data_test)
set_tests_properties(
  codec_round_trip
  codec_operators
  codec_real_data
  PROPERTIES LABELS "codec;gpu" TIMEOUT 900 SKIP_RETURN_CODE 77
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
