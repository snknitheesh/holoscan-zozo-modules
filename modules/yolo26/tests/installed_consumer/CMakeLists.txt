# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_yolo26_consumer LANGUAGES CXX)
enable_testing()

# The overlay component when the package has it, to exercise both overlays too.
find_package(holoscan-yolo26 1.1 REQUIRED CONFIG OPTIONAL_COMPONENTS overlay)
message(
  STATUS
  "Found holoscan-yolo26 ${holoscan-yolo26_VERSION} in ${holoscan-yolo26_DIR}"
  " (prepare_model.py: ${holoscan-yolo26_PREPARE_MODEL})"
)
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

# yolo26_consumer uses this version's API; yolo26_consumer_1_0 the same source with only
# what 1.0 offered, so code written for 1.0 keeps building and running against this package.
foreach(_consumer yolo26_consumer yolo26_consumer_1_0)
  add_executable(${_consumer} main.cpp)
  target_link_libraries(${_consumer} PRIVATE holoscan::yolo26)
  set_target_properties(${_consumer} PROPERTIES CXX_EXTENSIONS OFF)
  if(TARGET holoscan::yolo26_overlay)
    target_link_libraries(${_consumer} PRIVATE holoscan::yolo26_overlay)
    target_compile_definitions(${_consumer} PRIVATE YOLO26_CONSUMER_WITH_OVERLAY=1)
  else()
    target_compile_definitions(${_consumer} PRIVATE YOLO26_CONSUMER_WITH_OVERLAY=0)
  endif()
endforeach()
target_compile_definitions(yolo26_consumer PRIVATE YOLO26_CONSUMER_API_1_0=0)
target_compile_definitions(yolo26_consumer_1_0 PRIVATE YOLO26_CONSUMER_API_1_0=1)

# Graphs with the stub models, which the consumers write to their working directory. They
# pass on their verdict, which needs every frame right and the graph on one worker.
add_test(
  NAME yolo26_installed_consumer
  COMMAND yolo26_consumer detect
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
add_test(
  NAME yolo26_installed_segmentation_consumer
  COMMAND yolo26_consumer segment
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
add_test(
  NAME yolo26_installed_consumer_1_0_api
  COMMAND yolo26_consumer_1_0 detect
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
# The consumers share the stub model's engine cache in their working directory.
set_tests_properties(
  yolo26_installed_consumer
  yolo26_installed_segmentation_consumer
  yolo26_installed_consumer_1_0_api
  PROPERTIES
    PASS_REGULAR_EXPRESSION "yolo26_consumer: workers=1\n.*yolo26_consumer: PASS"
    TIMEOUT 600
    LABELS "yolo26;gpu"
    RESOURCE_LOCK yolo26_consumer_engines
)

# The model preparation tool is installed and runs.
add_test(
  NAME yolo26_installed_prepare_model
  COMMAND "${Python3_EXECUTABLE}" "${holoscan-yolo26_PREPARE_MODEL}" --help
)
set_tests_properties(
  yolo26_installed_prepare_model
  PROPERTIES
    PASS_REGULAR_EXPRESSION "usage: .*--output OUTPUT"
    ENVIRONMENT PYTHONDONTWRITEBYTECODE=1
    TIMEOUT 60
    LABELS yolo26
)
