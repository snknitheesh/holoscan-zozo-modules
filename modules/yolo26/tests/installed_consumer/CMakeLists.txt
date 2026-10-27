# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_yolo26_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-yolo26 1.0 REQUIRED CONFIG)
message(STATUS "Found holoscan-yolo26 ${holoscan-yolo26_VERSION} in ${holoscan-yolo26_DIR}"
  " (prepare_model.py: ${holoscan-yolo26_PREPARE_MODEL})")
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

add_executable(yolo26_consumer main.cpp)
target_link_libraries(yolo26_consumer PRIVATE holoscan::yolo26)

# A graph with the stub model, which the consumer writes to its working directory.
add_test(NAME yolo26_installed_consumer COMMAND yolo26_consumer
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_tests_properties(yolo26_installed_consumer PROPERTIES
  PASS_REGULAR_EXPRESSION "yolo26_consumer: PASS" TIMEOUT 600)

# The model preparation tool is installed and runs.
add_test(NAME yolo26_installed_prepare_model
  COMMAND "${Python3_EXECUTABLE}" "${holoscan-yolo26_PREPARE_MODEL}" --help)
set_tests_properties(yolo26_installed_prepare_model PROPERTIES
  PASS_REGULAR_EXPRESSION "usage: .*--output OUTPUT" ENVIRONMENT PYTHONDONTWRITEBYTECODE=1
  TIMEOUT 60)
