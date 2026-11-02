# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the CPU reference against the colour
# science, resizing, letterbox geometry, and the backend choice. Built on their own by the
# hosted CI job, and as part of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_color_convert_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# The module's SDK-free sources: the formats, the geometry, the backend choice and the
# CPU reference. Without contraction, as the module builds the reference.
add_library(
  color_convert_host_core
  STATIC
  "${_module}/src/conversion.cpp"
  "${_module}/src/reference.cpp"
)
target_include_directories(
  color_convert_host_core
  PUBLIC "${_module}/include"
  PRIVATE "${_module}/src"
)
target_compile_features(color_convert_host_core PUBLIC cxx_std_20)
target_compile_options(color_convert_host_core PRIVATE -ffp-contract=off)

add_executable(color_convert_conversion_math_test conversion_math_test.cpp)
target_include_directories(
  color_convert_conversion_math_test
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(color_convert_conversion_math_test PRIVATE color_convert_host_core)
add_test(NAME color_convert_conversion_math COMMAND color_convert_conversion_math_test)

add_executable(color_convert_backend_choice_test backend_choice_test.cpp)
target_include_directories(
  color_convert_backend_choice_test
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(color_convert_backend_choice_test PRIVATE color_convert_host_core)
add_test(NAME color_convert_backend_choice COMMAND color_convert_backend_choice_test)

set_tests_properties(
  color_convert_conversion_math
  color_convert_backend_choice
  PROPERTIES LABELS "color_convert;host" TIMEOUT 300
)
