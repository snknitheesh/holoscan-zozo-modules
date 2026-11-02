# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

# Runs the demo through run_and_match.cmake, which checks its exit status as well as
# its output: stitching_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(stitching_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:stitching> ${ARGN}
  )
endfunction()

# Headless, every output checked against the ground truth.
stitching_app_test(stitching_app_feather 0 "frames=10/10 .*verdict=PASS"
  --count 10 --period-ms 5 --display none
)
stitching_app_test(stitching_app_exposure 0 "frames=10/10 .*verdict=PASS"
  --count 10 --period-ms 5 --display none --exposure true --camera-gains 1,0.85,1.15,0.95
)
stitching_app_test(stitching_app_hard_seam 0 "frames=10/10 .*verdict=PASS"
  --count 10 --period-ms 5 --display none --blend hard_seam --panorama-width 2048
  --panorama-height 512 --birds-eye-size 512
)
# Different exposures without compensation fail the brightness check: the check
# can fail.
stitching_app_test(stitching_app_detects_exposure 1 "first failure: .*verdict=FAIL"
  --count 3 --period-ms 5 --display none --camera-gains 1,0.6,1.4,0.8
)

# Invalid options fail fast with a usage error (exit status 2).
stitching_app_test(stitching_app_rejects_blend 2 "^error: --blend" --blend multiband)
stitching_app_test(stitching_app_rejects_gains 2 "^error: give one camera gain per camera"
  --camera-gains 1,1 --display none
)
stitching_app_test(stitching_app_rejects_missing_rig 2 "^error: .*cannot read"
  --rig /nonexistent/rig.json --display none
)
stitching_app_test(stitching_app_rejects_zero_count 2 "^error: " --count 0 --display none)

if(STITCHING_APP_ENABLE_VISUALIZATION)
  # HoloViz renders offscreen into CUDA memory; every render is checked.
  stitching_app_test(stitching_app_offscreen 0 "renders=[1-9][0-9]*.*verdict=PASS"
    --count 10 --period-ms 5 --display offscreen
  )
  # The window graph compiles, with or without a display to run it.
  stitching_app_test(stitching_app_window_plan 0 "plan=OK"
    --display window --compile-only true
  )
endif()

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "stitching;gpu" TIMEOUT 600)
