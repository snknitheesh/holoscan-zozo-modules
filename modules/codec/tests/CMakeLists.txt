# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Everything but the host tests needs the SDK, CUDA and a GPU; the codec
# tests skip (exit 77) where no backend can encode and decode, saying why.

add_executable(codec_test codec_test.cpp)
target_link_libraries(codec_test PRIVATE holoscan::codec)

add_test(NAME codec_round_trip COMMAND codec_test)
set_tests_properties(
  codec_round_trip
  PROPERTIES LABELS "codec;gpu" TIMEOUT 900 SKIP_RETURN_CODE 77
)

add_executable(codec_operator_test operator_test.cpp)
target_link_libraries(codec_operator_test PRIVATE holoscan::codec holoscan::testing)
add_test(NAME codec_operators COMMAND codec_operator_test)
set_tests_properties(
  codec_operators
  PROPERTIES LABELS "codec;gpu" TIMEOUT 900 SKIP_RETURN_CODE 77
)

# Throughput at 1080p and 2160p, printed for the record; it fails only when a backend fails.
add_executable(codec_benchmark codec_benchmark.cpp)
target_link_libraries(codec_benchmark PRIVATE holoscan::codec)
add_test(NAME codec_benchmark COMMAND codec_benchmark --frames 120)
set_tests_properties(
  codec_benchmark
  PROPERTIES LABELS "codec;gpu;benchmark" TIMEOUT 600 SKIP_RETURN_CODE 77
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
