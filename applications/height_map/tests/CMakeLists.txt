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
# The display graph compiles, without a display server.
height_map_app_test(height_map_app_display_graph 0 "plan=OK" --display true --compile-only true)

# Unusable arguments are usage errors.
height_map_app_test(height_map_app_rejects_size 2 "^error: camera images" --width 10)
height_map_app_test(height_map_app_rejects_map 2 "^error: --map-cells" --map-cells 8)
height_map_app_test(height_map_app_rejects_elevation 2 "^error: --elevation" --elevation median)
height_map_app_test(height_map_app_rejects_option 2 "^error: unknown option" --frames 3)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "height_map_app;gpu" TIMEOUT 300)
