# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

add_library(normals_spawn_thread_preload SHARED spawn_thread_preload.cpp)

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: normals_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(normals_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:normals_planes> ${ARGN}
  )
endfunction()

# Every frame checked against the room's truth, with other searches, merging, and at 720p.
# The camera comes from a camera_info file written at run time; a frame before the first
# camera info gives no cloud, which the demo reports (before_camera_info), at most one.
set(_pass "before_camera_info=[01] .*wrong_planes=0 flat_sphere=0.*verdict=PASS")
normals_app_test(normals_app_window 0 "${_pass}" --count 10 --period-ms 10 --display false)
normals_app_test(normals_app_wide_merged 0 "${_pass}"
  --count 5 --period-ms 10 --window-radius 5 --merge true --display false
)
normals_app_test(normals_app_knn 0 "${_pass}" --count 5 --period-ms 10 --search knn --display false)
normals_app_test(normals_app_radius 0 "${_pass}"
  --count 5 --period-ms 10 --search radius --display false
)
normals_app_test(normals_app_720p 0 "${_pass}"
  --count 5 --period-ms 20 --width 1280 --height 720 --window-radius 6 --display false
)
# HoloViz renders both views offscreen, into CUDA memory.
normals_app_test(normals_app_offscreen 0 "showing=[1-9].*verdict=PASS"
  --count 10 --period-ms 30 --display offscreen
)
# Only the plan's verdict: an accepted plan has no warnings, and the whole graph runs on one
# worker, with or without the display (whose graph compiles without a display server).
normals_app_test(normals_app_compile_only 0 "^normals_planes: plan=OK workers=1\n*$"
  --display false --compile-only true
)
normals_app_test(normals_app_display_graph 0 "^normals_planes: plan=OK workers=1\n*$"
  --display true --compile-only true
)
normals_app_test(normals_app_offscreen_graph 0 "^normals_planes: plan=OK workers=1\n*$"
  --display offscreen --compile-only true
)

# A failed run reports its cause and exits 1: operators sized for fewer points than the
# image has refuse every cloud.
normals_app_test(normals_app_failed_run 1 "exceed the configured maximum of 1000.*verdict=FAIL"
  --count 5 --period-ms 10 --display false --max-points 1000
)

# A run that outlives its deadline is stopped and fails: 100 frames of 2048x2048, each
# checked point by point on the host, take far longer than the second allowed beyond the
# 0.2 s the frames need.
normals_app_test(normals_app_timed_out 1 "timed out.*frames=[0-9]+/100 .*verdict=FAIL"
  --width 2048 --height 2048 --window-radius 6 --count 100 --period-ms 1 --display false
  --timeout-seconds 1
)

# A stop request ends the run early with its own verdict, and the warning it leaves is
# shown. The signal goes out once the reporter has checked the first frame, so the run has
# started. The test sends SIGTERM, as a shell starts background jobs with SIGINT ignored,
# and its script has no semicolons, where CMake would split it.
set(_stopped "${CMAKE_CURRENT_BINARY_DIR}/stopped")
add_test(
  NAME normals_app_stopped
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=1
    "-DEXPECT_OUTPUT=warning RUNTIME_MANAGED_SIGNAL_RECEIVED.*frames=[0-9]+/100000 .*verdict=STOPPED"
    -P "${_run_and_match}" -- sh -c
    "rm -rf \"$2\" && mkdir -p \"$2\" || exit 1
     \"$1\" --count 100000 --period-ms 10 --width 64 --height 48 --display false > \"$2/out\" 2>&1 &
     for i in $(seq 600)
     do
       grep -q 'the first frame is checked' \"$2/out\" && break
       sleep 0.05
     done
     kill -TERM $!
     wait $!
     status=$?
     cat \"$2/out\"
     exit $status"
    sh $<TARGET_FILE:normals_planes> "${_stopped}"
)

# Another thread running before main(), as under a profiler or a sanitizer: the run goes
# ahead without managed signal handling, and says so.
add_test(
  NAME normals_app_without_signal_domain
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS" -P "${_run_and_match}"
    -- ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:normals_spawn_thread_preload>
    $<TARGET_FILE:normals_planes> --count 5 --period-ms 10 --display false
)

# Unusable arguments are usage errors.
normals_app_test(normals_app_rejects_size 2 "^error: scene images" --width 10)
normals_app_test(normals_app_rejects_window 2 "^error: window_radius" --window-radius 0)
normals_app_test(normals_app_rejects_search 2 "^error: --search" --search median)
normals_app_test(normals_app_rejects_option 2 "^error: unknown option" --frames 3)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "normals_app;gpu" TIMEOUT 300)
