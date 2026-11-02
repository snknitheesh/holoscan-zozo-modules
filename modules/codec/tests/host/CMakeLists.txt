# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the Annex-B reader, the settings and their names,
# and the test video and PSNR. Built on their own by the hosted CI job, and as part of the module's
# full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_codec_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_executable(
  codec_host_test
  host_test.cpp
  "${_module}/src/bitstream.cpp"
  "${_module}/src/codec_types.cpp"
  "${_module}/src/test_video.cpp"
)
target_include_directories(
  codec_host_test
  PRIVATE "${_module}/include" "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(codec_host_test PRIVATE cxx_std_20)
add_test(NAME codec_host COMMAND codec_host_test)
set_tests_properties(codec_host PROPERTIES LABELS "codec;host" TIMEOUT 60)
