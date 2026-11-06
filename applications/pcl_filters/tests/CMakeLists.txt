# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

add_executable(pcl_report_op_test report_op_test.cpp)
target_link_libraries(pcl_report_op_test PRIVATE pcl_filters_app_ops holoscan::testing)
add_test(NAME pcl_report_op COMMAND pcl_report_op_test)

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: pcl_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(pcl_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:pcl_filters> ${ARGN}
  )
endfunction()

# Every frame checked against the scene's truth, with each outlier mode, at another size.
set(_pass "wrong_voxel_counts=0 .*wrong_planes=0.*verdict=PASS")
pcl_app_test(pcl_app_radius 0 "${_pass}" --count 10 --period-ms 10 --display false)
pcl_app_test(pcl_app_statistical 0 "${_pass}"
  --count 10 --period-ms 10 --display false --outliers statistical
)
pcl_app_test(pcl_app_720p 0 "${_pass}"
  --count 5 --period-ms 20 --width 1280 --height 720 --voxel-size 0.05 --display false
)
# Large frames: the time allowed grows with the points, beyond --timeout-seconds.
pcl_app_test(pcl_app_large_frames 0 "${_pass}"
  --count 10 --period-ms 1 --width 2048 --height 2048 --timeout-seconds 1 --display false
)
# The voxels exactly as many as there are, waiting for their count.
pcl_app_test(pcl_app_dense_voxels 0 "${_pass}"
  --count 10 --period-ms 10 --display false --voxel-layout dense
)
# The display graph compiles, without a display server. Only the plan's verdict: an
# accepted plan has no warnings, and the graph runs on one worker, as the demo lays it out;
# the automatic partitioning would split it onto three.
pcl_app_test(pcl_app_display_graph 0 "^pcl_filters: plan=OK workers=1\n*$"
  --display true --compile-only true
)
pcl_app_test(pcl_app_one_worker 0 "^pcl_filters: plan=OK workers=1\n*$"
  --display false --compile-only true
)

# Unusable arguments are usage errors.
pcl_app_test(pcl_app_rejects_size 2 "^error: scene images" --width 10)
pcl_app_test(pcl_app_rejects_voxel_size 2 "^error: voxel_size" --voxel-size 0)
pcl_app_test(pcl_app_rejects_outliers 2 "^error: --outliers" --outliers median)
pcl_app_test(pcl_app_rejects_voxel_layout 2 "^error: --voxel-layout" --voxel-layout keep)
pcl_app_test(pcl_app_rejects_option 2 "^error: unknown option" --frames 3)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "pcl_filters_app;gpu" TIMEOUT 300)
