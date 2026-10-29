# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Tests that need no Holoscan SDK, CUDA or GPU: the calibration file, Netpbm images, recordings,
# the synthetic camera, side-by-side splitting and the playback backend, built from the module's
# SDK-free sources without librealsense. Built on their own by the hosted CI job, and as part of
# the module's full suite. The real-data test runs when HOLOSCAN_TEST_DATA names a folder with
# recordings in stereo-camera/, and is skipped otherwise.
cmake_minimum_required(VERSION 4.0)
project(holoscan_stereo_camera_host_tests LANGUAGES CXX)
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
endif()

set(_module "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_library(
  stereo_camera_host_core
  STATIC
  "${_module}/src/calibration.cpp"
  "${_module}/src/image.cpp"
  "${_module}/src/json.cpp"
  "${_module}/src/netpbm.cpp"
  "${_module}/src/playback_source.cpp"
  "${_module}/src/recording.cpp"
  "${_module}/src/side_by_side.cpp"
  "${_module}/src/source.cpp"
  "${_module}/src/synthetic.cpp"
  "${_module}/src/v4l2_source.cpp"
)
target_compile_features(stereo_camera_host_core PUBLIC cxx_std_20)
target_include_directories(
  stereo_camera_host_core
  PUBLIC "${_module}/include" "${_module}/src" "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_definitions(stereo_camera_host_core PRIVATE HOLOSCAN_STEREO_CAMERA_WITH_REALSENSE=0)

set(_work "${CMAKE_CURRENT_BINARY_DIR}/work")
file(MAKE_DIRECTORY "${_work}")
foreach(
  _name
  IN
  ITEMS calibration recording side_by_side source real_data
)
  add_executable(stereo_camera_${_name}_test ${_name}_test.cpp)
  target_link_libraries(stereo_camera_${_name}_test PRIVATE stereo_camera_host_core)
  add_test(NAME stereo_camera_${_name} COMMAND stereo_camera_${_name}_test "${_work}")
endforeach()
set_tests_properties(
  stereo_camera_calibration
  stereo_camera_recording
  stereo_camera_side_by_side
  stereo_camera_source
  stereo_camera_real_data
  PROPERTIES LABELS "stereo_camera;host" TIMEOUT 300
)
# Without HOLOSCAN_TEST_DATA there is nothing to check.
set_tests_properties(
  stereo_camera_real_data
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "stereo_camera;host;data"
)
