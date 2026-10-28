# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

add_executable(apriltag_report_op_test report_op_test.cpp)
target_link_libraries(apriltag_report_op_test PRIVATE apriltag_detection_ops holoscan::testing)
add_test(NAME apriltag_report_op COMMAND apriltag_report_op_test)

add_library(apriltag_spawn_thread_preload SHARED spawn_thread_preload.cpp)

# Runs the demo through run_and_match.cmake, which checks its exit status as well as
# its output: apriltag_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(apriltag_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:apriltag_detection> ${ARGN}
  )
endfunction()

# The demo on synthetic frames, every compiled backend and auto, from host and from GPU
# memory. Each run checks its detections against ground truth and, on a CI runner, that
# auto picked the platform's backend.
apriltag_app_test(apriltag_app_synthetic 0 "backend=auto:.*verdict=PASS"
  --backend all,auto --count 30 --period-ms 10 --display false
)
apriltag_app_test(apriltag_app_device_input 0 "backend=auto:.*verdict=PASS"
  --backend all,auto --count 20 --period-ms 10 --device-input true --display false
)

# Invalid input fails fast with a usage error (exit status 2) instead of starting a run.
apriltag_app_test(apriltag_app_rejects_unknown_backend 2 "^error: "
  --backend gpu --display false
)
apriltag_app_test(apriltag_app_rejects_zero_count 2 "^error: " --count 0 --display false)
apriltag_app_test(apriltag_app_rejects_repeated_backend 2 "^error: .*listed twice"
  --backend all,cuda --display false
)

# A run whose detections fail the ground-truth check: at 1/8 resolution the tags of a
# 320x240 scene are too small to find. It exits 1.
apriltag_app_test(apriltag_app_failed_run 1 "missed=20 .*verdict=FAIL"
  --backend cuda --width 320 --height 240 --decimate 8 --count 5 --period-ms 10 --display false
)

# A stop request ends the run early with its own verdict, and the warning it leaves is
# shown. (A shell starts background jobs with SIGINT ignored, so the test sends SIGTERM.)
add_test(
  NAME apriltag_app_stopped
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=1
    "-DEXPECT_OUTPUT=warning RUNTIME_MANAGED_SIGNAL_RECEIVED.*frames=[0-9]+/1000 .*verdict=STOPPED"
    -P "${_run_and_match}" -- sh -c
    "\"$1\" --backend cuda --count 1000 --period-ms 10 --display false & sleep 3 && kill -TERM $! && wait $!"
    sh $<TARGET_FILE:apriltag_detection>
)

# Another thread running before main(), as under a profiler or a sanitizer: the run
# goes ahead without managed signal handling, and says so.
add_test(
  NAME apriltag_app_without_signal_domain
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS" -P "${_run_and_match}"
    -- ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:apriltag_spawn_thread_preload>
    $<TARGET_FILE:apriltag_detection> --backend cuda --count 10 --period-ms 10 --display false
)

if(APRILTAG_DETECTION_ENABLE_VISUALIZATION)
  # The display graph compiles, with or without a display to run it.
  apriltag_app_test(apriltag_app_display_plan 0 "plan=OK"
    --backend all --display true --compile-only true
  )
endif()

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "apriltag;gpu" TIMEOUT 600)
