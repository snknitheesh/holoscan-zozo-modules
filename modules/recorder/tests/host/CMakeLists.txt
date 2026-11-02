# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the recording format, its writer and reader,
# malformed recordings, and the import of stereo camera recordings, built from the module's
# SDK-free sources. Built on their own by the hosted CI job, and as part of the module's full
# suite. The real-data test runs when HOLOSCAN_TEST_DATA names a folder with recordings in
# stereo-camera/, and is skipped otherwise.
cmake_minimum_required(VERSION 4.0)
project(holoscan_recorder_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_library(
  recorder_host_format
  STATIC
  "${_module}/src/csv.cpp"
  "${_module}/src/files.cpp"
  "${_module}/src/format.cpp"
  "${_module}/src/json.cpp"
)
target_compile_features(recorder_host_format PUBLIC cxx_std_20)
target_include_directories(
  recorder_host_format
  PUBLIC "${_module}/include" "${_module}/src" "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

set(_work "${CMAKE_CURRENT_BINARY_DIR}/work")
file(MAKE_DIRECTORY "${_work}")
foreach(_name IN ITEMS format malformed host_real_data)
  add_executable(recorder_${_name}_test ${_name}_test.cpp)
  target_link_libraries(recorder_${_name}_test PRIVATE recorder_host_format)
  add_test(NAME recorder_${_name} COMMAND recorder_${_name}_test "${_work}")
  set_tests_properties(recorder_${_name} PROPERTIES LABELS "recorder;host" TIMEOUT 300)
endforeach()
# Without HOLOSCAN_TEST_DATA there is nothing to check.
set_tests_properties(
  recorder_host_real_data
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "recorder;host;data"
)
