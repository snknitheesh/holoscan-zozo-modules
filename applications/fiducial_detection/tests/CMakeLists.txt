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

# A run whose detections fail the ground-truth check: with a smallest marker size above the
# scene's markers, none is found. It exits 1.
fiducial_app_test(fiducial_app_failed_run 1 "missed=18 .*verdict=FAIL"
  --scene aruco --min-marker-px 400 --count 3 --period-ms 10 --display false
)

# A stop request ends the run early with its own verdict. (A shell starts background jobs
# with SIGINT ignored, so the test sends SIGTERM.)
add_test(
  NAME fiducial_app_stopped
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=1 "-DEXPECT_OUTPUT=frames=[0-9]+/1000 .*verdict=STOPPED"
    -P "${_run_and_match}" -- sh -c
    "\"$1\" --scene aruco --count 1000 --period-ms 10 --display false & sleep 3 && kill -TERM $! && wait $!"
    sh $<TARGET_FILE:fiducial_detection>
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
