# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the stub model and its formula, the camera
# info rules and the depth formula, and the CPU references the GPU tests compare against,
# built from the SDK-free sources of this module and of holoscan-messages, which it depends
# on; and tools/prepare_model.py, without PyTorch or the network. Built on their own by the
# hosted CI job, and as part of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_foundation_stereo_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(_messages "${_module}/../messages")
add_executable(
  foundation_stereo_host_test
  host_test.cpp
  "${_module}/src/camera_info.cpp"
  "${_module}/src/stub_model.cpp"
  "${_messages}/src/camera_info.cpp"
  "${_messages}/src/synthetic_calibration.cpp"
)
target_include_directories(
  foundation_stereo_host_test
  PRIVATE
    "${_module}/include"
    "${_messages}/include"
    "${_messages}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(foundation_stereo_host_test PRIVATE cxx_std_20)
set_target_properties(foundation_stereo_host_test PROPERTIES CXX_EXTENSIONS OFF)
add_test(
  NAME foundation_stereo_host
  COMMAND foundation_stereo_host_test "${CMAKE_CURRENT_BINARY_DIR}/host_files"
)

add_test(
  NAME foundation_stereo_prepare_model
  COMMAND
    "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}" -p
    test_prepare_model.py -v
)

# Keep the bytecode out of the source tree.
set_tests_properties(
  foundation_stereo_prepare_model
  PROPERTIES ENVIRONMENT PYTHONDONTWRITEBYTECODE=1
)
set_tests_properties(
  foundation_stereo_host
  foundation_stereo_prepare_model
  PROPERTIES LABELS "foundation_stereo;host" TIMEOUT 60
)
