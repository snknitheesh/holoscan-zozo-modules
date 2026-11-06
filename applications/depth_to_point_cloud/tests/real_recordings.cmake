# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Runs the demo on the depth images of every local recording in
# $HOLOSCAN_TEST_DATA/stereo-camera/<clip>/ that has a depth.yaml, with that camera_info
# file, and passes only if every run passes. Without HOLOSCAN_TEST_DATA it prints
# "SKIPPED: ..." and runs nothing; give the test that as its SKIP_REGULAR_EXPRESSION.
#
#   cmake -DDEMO=<depth_to_point_cloud> -DDEPTH_SCALE=<meters per unit> -P real_recordings.cmake

if(NOT DEFINED DEMO OR NOT DEFINED DEPTH_SCALE)
  message(FATAL_ERROR "real_recordings: DEMO and DEPTH_SCALE are required")
endif()
if("$ENV{HOLOSCAN_TEST_DATA}" STREQUAL "")
  message("SKIPPED: HOLOSCAN_TEST_DATA is not set")
  return()
endif()

file(GLOB _cameras "$ENV{HOLOSCAN_TEST_DATA}/stereo-camera/*/depth.yaml")
list(SORT _cameras)
if(NOT _cameras)
  message(
    FATAL_ERROR
    "real_recordings: no stereo-camera/<clip>/depth.yaml in $ENV{HOLOSCAN_TEST_DATA}"
  )
endif()

foreach(_camera IN LISTS _cameras)
  get_filename_component(_clip "${_camera}" DIRECTORY)
  execute_process(
    COMMAND
      "${DEMO}" --source images --images "${_clip}/depth" --camera-info "${_camera}" --depth-scale
      ${DEPTH_SCALE} --period-ms 5
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _output
  )
  message("${_clip}:\n${_output}")
  if(NOT _status EQUAL 0 OR NOT _output MATCHES "verdict=PASS")
    message(FATAL_ERROR "real_recordings: ${_clip} failed with exit status ${_status}")
  endif()
endforeach()
