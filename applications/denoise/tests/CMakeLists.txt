# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: denoise_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(denoise_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:denoise> ${ARGN}
  )
endfunction()

set(
  _fast
  --count
  24
  --period-ms
  10
  --width
  640
  --height
  360
  --display
  false
)
# Every frame checked against the truth: each method, a scene cut, NV12-free colour at 1080p.
denoise_app_test(denoise_app_temporal_bilateral 0 "spatial=bilateral temporal=on.*verdict=PASS"
  ${_fast}
)
denoise_app_test(denoise_app_temporal_nlm 0 "spatial=nlm.*verdict=PASS" ${_fast} --spatial nlm)
denoise_app_test(denoise_app_temporal_only 0 "spatial=none temporal=on.*verdict=PASS"
  ${_fast} --spatial none
)
denoise_app_test(denoise_app_bilateral_only 0 "temporal=off.*verdict=PASS"
  ${_fast} --temporal false --speed 0
)
denoise_app_test(denoise_app_scene_cut 0 "verdict=PASS" ${_fast} --cut-frame 12)
denoise_app_test(denoise_app_auto 0 "backend=auto:.*verdict=PASS" ${_fast} --backend auto)
denoise_app_test(denoise_app_1080p 0 "verdict=PASS"
  --count 8 --period-ms 20 --width 1920 --height 1080 --display false
)
if(holoscan-denoise_WITH_VPI)
  denoise_app_test(denoise_app_vpi 0 "backend=vpi.*verdict=PASS" ${_fast} --backend vpi)
endif()
# HoloViz renders the view offscreen, into CUDA memory.
denoise_app_test(denoise_app_offscreen 0 "showing=[1-9].*verdict=PASS"
  --count 12 --period-ms 30 --width 640 --height 360 --display offscreen
)
# The display graph compiles, without a display server.
denoise_app_test(denoise_app_display_graph 0 "plan=OK" --display true --compile-only true)

# Unusable arguments are usage errors.
denoise_app_test(denoise_app_rejects_size 2 "^error: frames must be even" --width 641)
denoise_app_test(denoise_app_rejects_strength 2 "^error: strength" --strength 2)
denoise_app_test(denoise_app_rejects_vpi_spatial 2 "^error: (the vpi backend|backend vpi)"
  --backend vpi --spatial bilateral
)
denoise_app_test(denoise_app_rejects_option 2 "^error: unknown option" --frames 3)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "denoise_app;gpu" TIMEOUT 300)
