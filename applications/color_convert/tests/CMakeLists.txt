# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

# Runs the demo through run_and_match.cmake, which checks its exit status as well as
# its output: color_convert_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(color_convert_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:color_convert> ${ARGN}
  )
endfunction()

# Headless, every output checked against the reference byte for byte (cuda) or within the
# VIC's tolerance: both camera formats, each resize mode, odd sizes.
color_convert_app_test(color_convert_app_nv12 0 "differing=0 .*differing=0.*verdict=PASS"
  --camera nv12 --backend cuda --count 20 --period-ms 5 --display none
)
color_convert_app_test(color_convert_app_yuyv_area 0 "differing=0 .*differing=0.*verdict=PASS"
  --camera yuyv --backend cuda --mode area --resize-width 300 --resize-height 170 --count 10
  --period-ms 5 --display none
)
color_convert_app_test(color_convert_app_nearest_stretch 0 "differing=0 .*differing=0.*verdict=PASS"
  --camera nv12 --width 1280 --height 720 --backend cuda --mode nearest --letterbox false
  --resize-width 333 --resize-height 201 --count 10 --period-ms 5 --display none
)
color_convert_app_test(color_convert_app_auto 0 "verdict=PASS"
  --camera yuyv --backend auto --count 10 --period-ms 5 --display none
)

# Invalid options fail fast with a usage error (exit status 2).
color_convert_app_test(color_convert_app_rejects_camera 2 "^error: " --camera rgb8)
color_convert_app_test(color_convert_app_rejects_odd_width 2 "^error: " --width 1921)
color_convert_app_test(color_convert_app_rejects_zero_count 2 "^error: "
  --count 0 --display none
)

if(COLOR_CONVERT_APP_ENABLE_VISUALIZATION)
  # HoloViz renders offscreen into CUDA memory; every render is checked against the frame.
  color_convert_app_test(color_convert_app_offscreen 0 "renders=[1-9][0-9]* insets=[1-9].*verdict=PASS"
    --camera nv12 --backend cuda --count 15 --period-ms 5 --display offscreen
  )
  # The window graph compiles, with or without a display to run it.
  color_convert_app_test(color_convert_app_window_plan 0 "plan=OK"
    --display window --compile-only true
  )
endif()

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "color_convert;gpu" TIMEOUT 600)
