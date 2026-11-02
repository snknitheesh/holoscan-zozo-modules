# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

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
set(_pass "wrong_planes=0 flat_sphere=0.*verdict=PASS")
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
# The display graph compiles, without a display server.
normals_app_test(normals_app_display_graph 0 "plan=OK" --display true --compile-only true)

# Unusable arguments are usage errors.
normals_app_test(normals_app_rejects_size 2 "^error: scene images" --width 10)
normals_app_test(normals_app_rejects_window 2 "^error: window_radius" --window-radius 0)
normals_app_test(normals_app_rejects_search 2 "^error: --search" --search median)
normals_app_test(normals_app_rejects_option 2 "^error: unknown option" --frames 3)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "normals_app;gpu" TIMEOUT 300)
