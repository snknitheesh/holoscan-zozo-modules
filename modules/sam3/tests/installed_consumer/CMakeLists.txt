# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_sam3_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-sam3 1.0 REQUIRED CONFIG COMPONENTS overlay)
message(
  STATUS
  "Found holoscan-sam3 ${holoscan-sam3_VERSION} in ${holoscan-sam3_DIR}"
  " (prepare_model.py: ${holoscan-sam3_PREPARE_MODEL})"
)
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

add_executable(sam3_consumer main.cpp)
target_link_libraries(sam3_consumer PRIVATE holoscan::sam3 holoscan::sam3_overlay)

# A graph with the stub model, which the consumer writes to its working directory.
add_test(
  NAME sam3_installed_consumer
  COMMAND sam3_consumer
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
set_tests_properties(
  sam3_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "sam3_consumer: PASS" TIMEOUT 600
)

# The model preparation tool is installed and runs.
add_test(
  NAME sam3_installed_prepare_model
  COMMAND "${Python3_EXECUTABLE}" "${holoscan-sam3_PREPARE_MODEL}" --help
)
set_tests_properties(
  sam3_installed_prepare_model
  PROPERTIES
    PASS_REGULAR_EXPRESSION "usage: .*--output OUTPUT"
    ENVIRONMENT PYTHONDONTWRITEBYTECODE=1
    TIMEOUT 60
)
