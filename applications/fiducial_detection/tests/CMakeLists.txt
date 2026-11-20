# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

add_library(fiducial_spawn_thread_preload SHARED spawn_thread_preload.cpp)

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: fiducial_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(fiducial_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:fiducial_detection> ${ARGN}
  )
endfunction()

# Markers and a ChArUco board on every backend and auto, from host and from GPU memory, each
# frame checked against ground truth and, on a CI runner, auto's choice.
fiducial_app_test(fiducial_app_synthetic 0 "charuco backend=auto:.*verdict=PASS"
  --scene both --backend all,auto --count 20 --period-ms 10 --display false
)
fiducial_app_test(fiducial_app_device_input 0 "aruco backend=cuda.*charuco backend=cuda.*verdict=PASS"
  --scene both --backend cuda --count 10 --period-ms 10 --device-input true --display false
)
# Other dictionaries and boards: AprilTag markers, a legacy board of another dictionary.
fiducial_app_test(fiducial_app_apriltag_markers 0 "aruco backend=cuda .*verdict=PASS"
  --scene aruco --dictionary DICT_APRILTAG_36h11 --count 10 --period-ms 10 --display false
)
fiducial_app_test(fiducial_app_legacy_board 0 "charuco backend=cpu .*verdict=PASS"
  --scene charuco --backend cpu --board 6x8 --square-length 0.035 --board-marker-length 0.026
  --board-dictionary DICT_5X5_100 --legacy-pattern true --count 10 --period-ms 10 --display false
)

# The lossless graphs run on one worker, as the demo lays them out: the camera info is a second
# source and the frames fan out to every detector, which the automatic partitioning would split
# onto a worker each.
fiducial_app_test(fiducial_app_one_worker 0 "^fiducial_detection: plan=OK workers=1\n*$"
  --scene both --backend all,auto --display false --compile-only true
)

# Invalid input fails fast with a usage error (exit status 2) instead of starting a run.
fiducial_app_test(fiducial_app_rejects_unknown_backend 2 "^error: " --backend vpi --display false)
fiducial_app_test(fiducial_app_rejects_unknown_dictionary 2 "^error: unknown dictionary"
  --dictionary DICT_8X8_50 --display false
)
fiducial_app_test(fiducial_app_rejects_bad_board 2 "^error: marker_length"
  --board-marker-length 0.05 --display false
)
fiducial_app_test(fiducial_app_rejects_missing_camera 2 "^error: cannot open camera"
  --source camera --device /dev/no-such-camera --display false
)

fiducial_app_test(fiducial_app_rejects_camera_info_for_scenes 2
  "^error: --camera-info does not apply to synthetic scenes" --camera-info camera.yaml
  --display false
)
fiducial_app_test(fiducial_app_rejects_missing_recording 2 "^error: .*index.csv"
  --recording "${CMAKE_CURRENT_BINARY_DIR}/no_such_recording" --display false
)

# A run whose detections fail the ground-truth check: with a smallest marker size above the
# scene's markers, none is found. It exits 1.
fiducial_app_test(fiducial_app_failed_run 1 "missed=18 .*verdict=FAIL"
  --scene aruco --min-marker-px 400 --count 3 --period-ms 10 --display false
)

# Real frames: every local recording in $HOLOSCAN_TEST_DATA/stereo-camera/ with a color camera
# (rgb.yaml). They show no marker or board, so every backend must find none (no false
# positives) and agree with the others frame by frame. Skipped (77) when the variable is unset;
# failed when it is set but names no such recording.
set(_none_found "aruco backend=cpu frames=[0-9]+/[0-9]+ markers=0 ")
string(APPEND _none_found ".*aruco backend=cuda frames=[0-9]+/[0-9]+ markers=0 ")
string(APPEND _none_found ".*charuco backend=cpu frames=[0-9]+/[0-9]+ markers=0 corners=0 ")
string(APPEND _none_found ".*charuco backend=cuda frames=[0-9]+/[0-9]+ markers=0 corners=0 ")
string(APPEND _none_found ".*aruco backends_agree=yes.*charuco backends_agree=yes.*verdict=PASS")
add_test(
  NAME fiducial_app_recordings
  COMMAND
    sh -c
    "if [ -z \"$HOLOSCAN_TEST_DATA\" ]; then echo 'HOLOSCAN_TEST_DATA is not set; skipping'; exit 77; fi; n=0; for c in \"$HOLOSCAN_TEST_DATA\"/stereo-camera/*/rgb.yaml; do [ -f \"$c\" ] || continue; n=$((n+1)); \"$0\" -DEXPECT_EXIT=0 \"-DEXPECT_OUTPUT=$3\" -P \"$1\" -- \"$2\" --recording \"$(dirname \"$c\")\" --scene both --backend cpu,cuda --period-ms 10 --display false || exit 1; done; [ $n -gt 0 ] || { echo \"no recordings with rgb.yaml in $HOLOSCAN_TEST_DATA/stereo-camera\"; exit 1; }"
    ${CMAKE_COMMAND} "${_run_and_match}" $<TARGET_FILE:fiducial_detection> "${_none_found}"
)
set_tests_properties(fiducial_app_recordings PROPERTIES SKIP_RETURN_CODE 77)

# A stop request ends the run early with its own verdict, sent once the run has started (its
# "running" line), however long the start takes.
add_test(
  NAME fiducial_app_stopped
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=1 "-DEXPECT_OUTPUT=frames=[0-9]+/1000 .*verdict=STOPPED" -P
    "${_run_and_match}" -- sh "${CMAKE_CURRENT_SOURCE_DIR}/stop_when_running.sh"
    "${CMAKE_CURRENT_BINARY_DIR}/fiducial_app_stopped.log" $<TARGET_FILE:fiducial_detection> --scene
    aruco --count 1000 --period-ms 10 --display false
)

# Another thread running before main(), as under a profiler or a sanitizer: the run goes ahead
# without managed signal handling, and says so.
add_test(
  NAME fiducial_app_without_signal_domain
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS" -P "${_run_and_match}"
    -- ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:fiducial_spawn_thread_preload>
    $<TARGET_FILE:fiducial_detection> --scene aruco --count 5 --period-ms 10 --display false
)

if(FIDUCIAL_DETECTION_ENABLE_VISUALIZATION)
  # The display graphs compile, with or without a display to run them, for synthetic frames
  # and for the camera (whose device is opened only when the run starts).
  fiducial_app_test(fiducial_app_display_plan 0 "plan=OK"
    --scene both --backend all --display true --compile-only true
  )
  fiducial_app_test(fiducial_app_camera_plan 0 "plan=OK"
    --source camera --scene charuco --count 0 --display true --compile-only true
  )
endif()

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "fiducials;gpu" TIMEOUT 600)
