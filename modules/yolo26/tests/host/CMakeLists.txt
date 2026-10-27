# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the letterbox geometry, the stub model,
# the synthetic scenes and their truth matching, built from the module's host-only
# sources; and tools/prepare_model.py, without Ultralytics or the network. Built on
# their own by the hosted CI job, and as part of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_yolo26_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_executable(yolo26_host_test host_test.cpp
  "${_module}/src/letterbox.cpp"
  "${_module}/src/stub_model.cpp"
  "${_module}/src/synthetic_scene.cpp")
target_include_directories(yolo26_host_test PRIVATE
  "${_module}/include" "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_features(yolo26_host_test PRIVATE cxx_std_20)
target_compile_options(yolo26_host_test PRIVATE -ffp-contract=off)
add_test(NAME yolo26_host
  COMMAND yolo26_host_test "${CMAKE_CURRENT_BINARY_DIR}/host_files")

add_test(NAME yolo26_prepare_model
  COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}" -p test_prepare_model.py -v)

# Keep the bytecode out of the source tree.
set_tests_properties(yolo26_prepare_model PROPERTIES ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
set_tests_properties(yolo26_host yolo26_prepare_model PROPERTIES
  LABELS "yolo26;host" TIMEOUT 60)
