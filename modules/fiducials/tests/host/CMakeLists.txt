# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the cpu detector on synthetic markers and
# boards, the dictionaries, poses, the overlay's drawing and the auto backend's choice. They
# build the module's CUDA-free sources themselves. Built on their own by the hosted CI job,
# and as part of the module's full suite.
cmake_minimum_required(VERSION 4.0)
project(holoscan_fiducials_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    # The detection tests render and search many 1280x720 images.
    set(CMAKE_BUILD_TYPE Release)
  endif()
endif()

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_library(
  fiducials_host_core
  STATIC
  "${_module}/src/charuco.cpp"
  "${_module}/src/cpu_detector.cpp"
  "${_module}/src/dictionary.cpp"
  "${_module}/src/dictionary_codes.cpp"
  "${_module}/src/pose.cpp"
  "${_module}/src/synthetic_scene.cpp"
  "${_module}/src/types.cpp"
)
set_source_files_properties(
  "${_module}/src/cpu_detector.cpp"
  PROPERTIES COMPILE_OPTIONS -ffp-contract=off
)
target_include_directories(
  fiducials_host_core
  PUBLIC "${_module}/include" "${_module}/src" "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(fiducials_host_core PUBLIC cxx_std_20)
set_target_properties(fiducials_host_core PROPERTIES CXX_EXTENSIONS OFF)

foreach(
  _test
  IN
  ITEMS dictionary marker_detection charuco pose overlay_drawing backend_resolution
)
  add_executable(fiducials_${_test}_test ${_test}_test.cpp)
  target_link_libraries(fiducials_${_test}_test PRIVATE fiducials_host_core)
  add_test(NAME fiducials_${_test} COMMAND fiducials_${_test}_test)
  set_tests_properties(fiducials_${_test} PROPERTIES LABELS "fiducials;host" TIMEOUT 600)
endforeach()
