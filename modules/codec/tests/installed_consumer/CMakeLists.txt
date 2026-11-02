# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on CMAKE_PREFIX_PATH (the
# Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_codec_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-codec 1.0 REQUIRED CONFIG)
message(
  STATUS
  "Found holoscan-codec ${holoscan-codec_VERSION} in ${holoscan-codec_DIR}"
  " (v4l2 backend: ${holoscan-codec_WITH_V4L2})"
)

add_executable(codec_consumer main.cpp)
target_compile_features(codec_consumer PRIVATE cxx_std_20)
target_link_libraries(codec_consumer PRIVATE holoscan::codec)

add_test(NAME codec_installed_consumer COMMAND codec_consumer)
set_tests_properties(
  codec_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "codec_consumer: .*: PASS" SKIP_RETURN_CODE 77 TIMEOUT 300
)
