# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: prompts and the prompt table, the host
# references, the stub model and the synthetic scene, built from the module's host-only
# sources; and tools/prepare_model.py, without PyTorch or the network. Built on their own
# by the hosted CI job, and as part of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_sam3_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_executable(
  sam3_host_test
  host_test.cpp
  "${_module}/src/prompts.cpp"
  "${_module}/src/reference.cpp"
  "${_module}/src/stub_model.cpp"
  "${_module}/src/synthetic_scene.cpp"
)
target_include_directories(
  sam3_host_test
  PRIVATE "${_module}/include" "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(sam3_host_test PRIVATE cxx_std_20)
target_compile_options(sam3_host_test PRIVATE -ffp-contract=off)
add_test(NAME sam3_host COMMAND sam3_host_test "${CMAKE_CURRENT_BINARY_DIR}/host_files")

add_test(
  NAME sam3_prepare_model
  COMMAND
    "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}" -p
    test_prepare_model.py -v
)

# Keep the bytecode out of the source tree.
set_tests_properties(sam3_prepare_model PROPERTIES ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
set_tests_properties(sam3_host sam3_prepare_model PROPERTIES LABELS "sam3;host" TIMEOUT 120)
