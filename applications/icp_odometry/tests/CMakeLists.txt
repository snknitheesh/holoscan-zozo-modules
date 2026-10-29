# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU. Each runs the demo through run_and_match.cmake,
# which checks its exit status as well as its output: icp_app_test(<name> <exit status>
# <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(icp_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:icp_odometry> ${ARGN}
  )
endfunction()

# Odometry along the room's path, checked against the truth: point-to-plane at two sizes,
# point-to-point, and without noise or subsampling.
set(_pass "unconverged=0 .*verdict=PASS")
icp_app_test(icp_app_point_to_plane 0 "frames=150/150 320x240 .*${_pass}"
  --width 320 --height 240 --count 150 --period-ms 2 --display false
)
icp_app_test(icp_app_640x480 0 "frames=60/60 640x480 .*${_pass}"
  --count 60 --path-frames 150 --period-ms 2 --display false
)
icp_app_test(icp_app_point_to_point 0 "frames=150/150 .*${_pass}"
  --width 320 --height 240 --count 150 --period-ms 2 --display false --method point-to-point
  --max-drift-percent 6 --max-rotation-error-deg 4
)
icp_app_test(icp_app_noiseless_full_resolution 0 "frames=40/40 .*${_pass}"
  --width 160 --height 120 --count 40 --path-frames 150 --period-ms 2 --display false
  --noise false --voxel-size 0 --normal-radius 0.15
)
# The display's graph, without a display server: compiled, not run.
if(ICP_ODOMETRY_ENABLE_VISUALIZATION)
  icp_app_test(icp_app_display_plan 0 "plan=OK" --display true --compile-only true)
endif()

# Limits it cannot meet fail the run.
icp_app_test(icp_app_fails_beyond_limits 1 "verdict=FAIL"
  --width 160 --height 120 --count 20 --period-ms 2 --display false --max-drift-percent 0
)
# Too few iterations to converge: frames are counted as failed, and the run fails.
icp_app_test(icp_app_fails_unconverged 1 "unconverged=[1-9][0-9]* .*verdict=FAIL"
  --width 160 --height 120 --count 20 --period-ms 2 --display false --max-iterations 1
)

# Invalid input fails fast with a usage error (exit status 2) instead of starting a run.
icp_app_test(icp_app_rejects_unknown_option 2 "^error: unknown option" --colour true)
icp_app_test(icp_app_rejects_zero_count 2 "^error: " --count 0)
icp_app_test(icp_app_rejects_small_image 2 "^error: depth images" --width 32)
icp_app_test(icp_app_rejects_method 2 "^error: --method" --method plane)
icp_app_test(icp_app_rejects_distances 2 "^error: min_correspondence_distance"
  --max-distance 0.05 --min-distance 0.1
)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "icp_odometry;gpu" TIMEOUT 300)
