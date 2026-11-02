# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_mono_depth_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-mono-depth 1.0 REQUIRED CONFIG)
message(
  STATUS
  "Found holoscan-mono-depth ${holoscan-mono-depth_VERSION} in ${holoscan-mono-depth_DIR}"
  " (prepare_model.py: ${holoscan-mono-depth_PREPARE_MODEL})"
)
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

add_executable(mono_depth_consumer main.cpp)
target_link_libraries(mono_depth_consumer PRIVATE holoscan::mono_depth)

# A graph with the stub model, which the consumer writes to its working directory.
add_test(
  NAME mono_depth_installed_consumer
  COMMAND mono_depth_consumer
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
set_tests_properties(
  mono_depth_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "mono_depth_consumer: PASS" TIMEOUT 600
)

# The model preparation tool is installed and runs.
add_test(
  NAME mono_depth_installed_prepare_model
  COMMAND "${Python3_EXECUTABLE}" "${holoscan-mono-depth_PREPARE_MODEL}" --help
)
set_tests_properties(
  mono_depth_installed_prepare_model
  PROPERTIES
    PASS_REGULAR_EXPRESSION "usage: .*--output OUTPUT"
    ENVIRONMENT PYTHONDONTWRITEBYTECODE=1
    TIMEOUT 60
)
