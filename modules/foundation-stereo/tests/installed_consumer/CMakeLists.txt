# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_foundation_stereo_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-foundation-stereo 1.0 REQUIRED CONFIG)
message(
  STATUS
  "Found holoscan-foundation-stereo ${holoscan-foundation-stereo_VERSION} in"
  " ${holoscan-foundation-stereo_DIR} (prepare_model.py:"
  " ${holoscan-foundation-stereo_PREPARE_MODEL})"
)
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

add_executable(foundation_stereo_consumer main.cpp)
target_link_libraries(foundation_stereo_consumer PRIVATE holoscan::foundation_stereo)
# It renders the synthetic scene on the host to check the output: without contracted
# multiply-adds, to the same bytes as the module's GPU renderer.
target_compile_options(foundation_stereo_consumer PRIVATE -ffp-contract=off)

# A graph with the stub model, which the consumer writes to its working directory.
add_test(
  NAME foundation_stereo_installed_consumer
  COMMAND foundation_stereo_consumer
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
set_tests_properties(
  foundation_stereo_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "foundation_stereo_consumer: PASS" TIMEOUT 600
)

# The model preparation tool is installed and runs.
add_test(
  NAME foundation_stereo_installed_prepare_model
  COMMAND "${Python3_EXECUTABLE}" "${holoscan-foundation-stereo_PREPARE_MODEL}" --help
)
set_tests_properties(
  foundation_stereo_installed_prepare_model
  PROPERTIES
    PASS_REGULAR_EXPRESSION "usage: .*--output OUTPUT"
    ENVIRONMENT PYTHONDONTWRITEBYTECODE=1
    TIMEOUT 60
)
