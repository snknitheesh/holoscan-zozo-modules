# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU. They write their synthetic recordings at run time, so
# they need no data; the real-data tests play the recordings in $HOLOSCAN_TEST_DATA/stereo-camera/
# and are skipped (exit 77) without them. None needs a camera.

add_library(stereo_camera_spawn_thread_preload SHARED spawn_thread_preload.cpp)

# Runs the demo through run_and_match.cmake, which checks its exit status as
# well as its output: stereo_camera_app_test(<name> <exit status> <output
# regex> <arguments>...). It runs in this directory.
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
set(_work "${CMAKE_CURRENT_BINARY_DIR}")
function(stereo_camera_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:stereo_camera> ${ARGN}
  )
endfunction()

# Synthetic recordings, every image checked against the synthetic camera: with and
# without rgb and depth, in host and GPU memory, another size, and looped.
set(_clean "wrong_sizes=0 not_increasing=0 frame_gaps=0 calibration_changes=0")
set(
  _play
  --source
  playback
  --period-ms
  5
  --display
  false
)
stereo_camera_app_test(stereo_camera_app_synthetic 0
  "frames=20/20 source=playback camera=\"synthetic\" eyes=320x240 mono8 baseline_mm=100.000 .*rgb=320x240 rgb8 depth=320x240 mono16 .*${_clean} truth: verified=20 wrong_images=0\n.*verdict=PASS"
  ${_play} --synthetic "${_work}/synthetic" --count 20
)
stereo_camera_app_test(stereo_camera_app_synthetic_device 0
  "frames=20/20 .*${_clean} truth: verified=20 wrong_images=0\n.*verdict=PASS"
  ${_play} --synthetic "${_work}/synthetic_device" --count 20 --device-output true
)
stereo_camera_app_test(stereo_camera_app_eyes_only 0
  "frames=10/10 .*eyes=320x240 mono8 baseline_mm=100.000 fx=256.000 rectified=1 wrong_sizes=0 .*truth: verified=10 wrong_images=0\n.*verdict=PASS"
  ${_play} --synthetic "${_work}/synthetic_eyes" --count 10 --rgb false --depth false
)
stereo_camera_app_test(stereo_camera_app_synthetic_large 0
  "frames=8/8 .*eyes=1280x720 mono8 .*truth: verified=8 wrong_images=0\n.*verdict=PASS"
  ${_play} --synthetic "${_work}/synthetic_large" --synthetic-size 1280x720 --count 8
)
stereo_camera_app_test(stereo_camera_app_loop 0
  "frames=25/25 .*${_clean} truth: verified=25 wrong_images=0\n.*verdict=PASS"
  ${_play} --synthetic "${_work}/synthetic_loop" --count 25 --loop true
)
# A recording written by the module is played back as one like it.
stereo_camera_app_test(stereo_camera_app_recording 0
  "frames=5/5 source=playback .*${_clean}\n{\n  \"version\": 1,\n  \"camera\": {\n    \"name\": \"synthetic\".*verdict=PASS"
  ${_play} --recording "${_work}/synthetic" --count 5 --print-calibration true
)
set_tests_properties(stereo_camera_app_recording PROPERTIES DEPENDS stereo_camera_app_synthetic)

# A recording with a broken frame fails the run, naming the file.
add_test(
  NAME stereo_camera_app_broken_frame
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=1
    "-DEXPECT_OUTPUT=broken/right/000002.pgm: pixel data is truncated.*verdict=FAIL" -P
    "${_run_and_match}" -- sh -c
    "rm -rf \"$2/broken\" && \"$1\" --source playback --synthetic \"$2/broken\" --count 4 --compile-only true > /dev/null && printf 'P5 320 240 255 ' > \"$2/broken/right/000002.pgm\" && exec \"$1\" --source playback --recording \"$2/broken\" --period-ms 5 --display false --timeout-seconds 10"
    sh $<TARGET_FILE:stereo_camera> "${_work}"
)

# The recordings of a real camera, when HOLOSCAN_TEST_DATA has them.
function(stereo_camera_real_test name clip output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=${output}"
      "-DTEST_DATA=stereo-camera/${clip}" -P "${_run_and_match}" -- $<TARGET_FILE:stereo_camera>
      ${_play} --recording "@DATA@/stereo-camera/${clip}" ${ARGN}
  )
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()
set(_d585 "camera=\"RealSense D585[^\"]*\" serial=[0-9]+ firmware=[0-9.]+")
stereo_camera_real_test(stereo_camera_real_720p_on d585_720p_emitter_on
  "frames=60/60 source=playback ${_d585} eyes=1280x720 mono8 baseline_mm=94\\.8[0-9]+ .*rgb=1280x720 rgb8 depth=1280x720 mono16 .*${_clean}\n.*verdict=PASS"
)
stereo_camera_real_test(stereo_camera_real_720p_off d585_720p_emitter_off
  "frames=60/60 .*eyes=1280x720 mono8 .*${_clean}\n.*verdict=PASS" --device-output true
)
stereo_camera_real_test(stereo_camera_real_480p_on d585_480p_emitter_on
  "frames=60/60 .*eyes=848x480 mono8 .*rgb=848x480 rgb8 depth=848x480 mono16 .*${_clean}\n.*verdict=PASS"
)
stereo_camera_real_test(stereo_camera_real_480p_off d585_480p_emitter_off
  "frames=60/60 .*eyes=848x480 mono8 .*${_clean}\n.*verdict=PASS" --rgb false --depth false
)

# Invalid input fails fast with a usage error (exit status 2) instead of starting a run.
stereo_camera_app_test(stereo_camera_app_rejects_unknown_option 2 "^error: unknown option"
  --colour 1
)
stereo_camera_app_test(stereo_camera_app_rejects_playback_without_recording 2
  "^error: --source playback needs --recording DIR or --synthetic DIR" --source playback
)
stereo_camera_app_test(stereo_camera_app_rejects_missing_recording 2
  "^error: .*no_such_folder: not a recording directory"
  --source playback --recording "${_work}/no_such_folder"
)
add_test(
  NAME stereo_camera_app_rejects_count_beyond_recording
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=2
    "-DEXPECT_OUTPUT=error: --count exceeds the recording's 3 frames; add --loop true" -P
    "${_run_and_match}" -- sh -c
    "\"$1\" --source playback --synthetic \"$2/short\" --count 3 --compile-only true > /dev/null && exec \"$1\" --source playback --recording \"$2/short\" --count 4"
    sh $<TARGET_FILE:stereo_camera> "${_work}"
)
stereo_camera_app_test(stereo_camera_app_rejects_recording_with_live 2
  "^error: --recording, --synthetic and --loop need --source playback"
  --source v4l2-sbs --recording "${_work}/synthetic"
)
stereo_camera_app_test(stereo_camera_app_rejects_v4l2_without_calibration 2
  "^error: v4l2-sbs: needs a calibration file" --source v4l2-sbs
)
if(holoscan-stereo-camera_WITH_REALSENSE)
  # Without a camera, the realsense backend fails at start, with its reason.
  stereo_camera_app_test(stereo_camera_app_realsense_without_camera 1 "verdict=FAIL"
    --source realsense --serial 000000000000 --display false --timeout-seconds 20
  )
else()
  stereo_camera_app_test(stereo_camera_app_rejects_realsense 2
    "^error: backend realsense: this build has no librealsense" --source realsense
  )
endif()

# A V4L2 camera that is not there fails the run at start, with its reason: the
# device is opened in start(), never when the graph is built.
file(
  WRITE "${_work}/sbs_calibration.json"
  "{\"version\": 1, \"rectified\": true,\n"
  " \"left\": {\"width\": 640, \"height\": 360, \"fx\": 350, \"fy\": 350, \"cx\": 320, \"cy\": 180},\n"
  " \"right\": {\"width\": 640, \"height\": 360, \"fx\": 350, \"fy\": 350, \"cx\": 320, \"cy\": 180},\n"
  " \"right_from_left\": {\"rotation\": [1, 0, 0, 0, 1, 0, 0, 0, 1], \"translation\": [-0.12, 0, 0]}}\n"
)
stereo_camera_app_test(stereo_camera_app_v4l2_missing_device 1
  "v4l2-sbs: cannot open /dev/video99.*verdict=FAIL"
  --source v4l2-sbs --device /dev/video99 --width 1280 --height 360 --calibration
  "${_work}/sbs_calibration.json" --display false --timeout-seconds 20
)
stereo_camera_app_test(stereo_camera_app_v4l2_plan 0 "^stereo_camera: plan=OK\n*$"
  --source v4l2-sbs --device /dev/video99 --width 1280 --height 360 --calibration
  "${_work}/sbs_calibration.json" --display false --compile-only true
)

stereo_camera_app_test(stereo_camera_app_help 0 "^Usage: .*--source realsense\\|v4l2-sbs\\|playback"
  --help
)
# Only the plan's verdict: an accepted plan has no warnings.
stereo_camera_app_test(stereo_camera_app_compile_only 0 "^stereo_camera: plan=OK\n*$"
  ${_play} --synthetic "${_work}/synthetic_plan" --count 3 --compile-only true
)
if(STEREO_CAMERA_ENABLE_VISUALIZATION)
  # The display graph compiles, with or without a display to run it: with rgb and without.
  stereo_camera_app_test(stereo_camera_app_display_plan 0 "^stereo_camera: plan=OK\n*$"
    --source playback --synthetic "${_work}/synthetic_display" --count 3 --display true
    --compile-only true
  )
  stereo_camera_app_test(stereo_camera_app_display_plan_eyes 0 "^stereo_camera: plan=OK\n*$"
    --source playback --synthetic "${_work}/synthetic_display_eyes" --count 3 --rgb false
    --depth false --display true --compile-only true
  )
endif()

# Another thread running before main(), as under a profiler or a sanitizer:
# the run goes ahead without managed signal handling, and says so.
add_test(
  NAME stereo_camera_app_without_signal_domain
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS" -P "${_run_and_match}"
    -- ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:stereo_camera_spawn_thread_preload>
    $<TARGET_FILE:stereo_camera> ${_play} --synthetic "${_work}/synthetic_signals" --count 5
)

get_property(_tests DIRECTORY PROPERTY TESTS)
foreach(_test IN LISTS _tests)
  get_test_property(${_test} TIMEOUT _timeout)
  if(NOT _timeout)
    set_tests_properties(${_test} PROPERTIES TIMEOUT 300)
  endif()
endforeach()
set_tests_properties(${_tests} PROPERTIES LABELS "stereo_camera;gpu")
