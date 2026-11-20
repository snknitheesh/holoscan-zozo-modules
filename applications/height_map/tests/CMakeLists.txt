# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: height_map_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(height_map_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:height_map> ${ARGN}
  )
endfunction()

# Every map checked against the course's truth: each elevation layer, a finer and a larger
# map, and 720p.
set(_pass "window_wrong=0 mask_wrong=0.*verdict=PASS")
height_map_app_test(height_map_app_fused 0 "${_pass}" --count 20 --period-ms 10 --display false)
height_map_app_test(height_map_app_mean 0 "${_pass}"
  --count 10 --period-ms 10 --elevation mean --display false
)
height_map_app_test(height_map_app_max 0 "${_pass}"
  --count 10 --period-ms 10 --elevation max --display false
)
height_map_app_test(height_map_app_large_map 0 "${_pass}"
  --count 10 --period-ms 20 --map-cells 500 --resolution 0.02 --display false
)
height_map_app_test(height_map_app_720p 0 "${_pass}"
  --count 10 --period-ms 20 --width 1280 --height 720 --display false
)
# HoloViz renders both maps offscreen, into CUDA memory.
height_map_app_test(height_map_app_offscreen 0 "showing=[1-9].*verdict=PASS"
  --count 20 --period-ms 30 --display offscreen
)
# One worker for the whole lossless graph, with and without the display; the automatic
# partitioning would split it onto several. Only the plan's verdict: an accepted plan has no
# warnings. The display graph compiles without a display server.
height_map_app_test(height_map_app_one_worker 0 "^height_map: plan=OK workers=1\n*$"
  --compile-only true --display false
)
height_map_app_test(height_map_app_display_graph 0 "^height_map: plan=OK workers=1\n*$"
  --display true --compile-only true
)

# A stop request ends the run early with its own verdict, and the camera_info file's
# temporary folder is gone afterwards. The signal goes out two seconds after the folder
# appears, so the run has started. The test sends SIGTERM, as a shell starts background jobs
# with SIGINT ignored, and its script has no semicolons, where CMake would split it.
set(_stopped "${CMAKE_CURRENT_BINARY_DIR}/stopped_tmp")
add_test(
  NAME height_map_app_stopped
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=1
    "-DEXPECT_OUTPUT=frames=[0-9]+/100000 .*verdict=STOPPED.*camera_info_removed"
    -P "${_run_and_match}" -- sh -c
    "rm -rf \"$2\" && mkdir -p \"$2\" || exit 1
     TMPDIR=\"$2\" \"$1\" --count 100000 --period-ms 10 --width 64 --height 48 --display false &
     for i in $(seq 600)
     do
       ls \"$2\" | grep -q height_map_ && break
       sleep 0.05
     done
     sleep 2
     kill -TERM $! && wait $!
     status=$?
     ls -A \"$2\" | grep -q . || echo camera_info_removed
     exit $status"
    sh $<TARGET_FILE:height_map> "${_stopped}"
)

# Unusable arguments are usage errors.
height_map_app_test(height_map_app_rejects_size 2 "^error: camera images" --width 10)
height_map_app_test(height_map_app_rejects_map 2 "^error: --map-cells" --map-cells 8)
height_map_app_test(height_map_app_rejects_elevation 2 "^error: --elevation" --elevation median)
height_map_app_test(height_map_app_rejects_option 2 "^error: unknown option" --frames 3)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "height_map_app;gpu" TIMEOUT 300)
