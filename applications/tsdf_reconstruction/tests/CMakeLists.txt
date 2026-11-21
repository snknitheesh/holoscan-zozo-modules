# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: tsdf_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(tsdf_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:tsdf_reconstruction> ${ARGN}
  )
endfunction()

# Every frame and every mesh checked against the scene's truth: float32 depth, a finer volume,
# 720p, and the mesh every frame. The ray casts are judged after the first 10 frames, and a run
# with none judged fails: each runs more.
set(_pass "judged=[1-9].*bad_triangles=0 bad_normals=0.*verdict=PASS")
tsdf_app_test(tsdf_app_float_depth 0 "${_pass}"
  --count 20 --orbit 20 --mesh-every 10 --period-ms 10 --depth m --dims 128 --display false
)
tsdf_app_test(tsdf_app_720p 0 "${_pass}"
  --count 14 --orbit 20 --mesh-every 7 --period-ms 20 --width 1280 --height 720 --dims 192
  --display false
)
tsdf_app_test(tsdf_app_mesh_every_frame 0 "meshes=12/12.*verdict=PASS"
  --count 12 --orbit 40 --mesh-every 1 --period-ms 10 --dims 96 --display false
)
# The final mesh saved as a PLY file.
tsdf_app_test(tsdf_app_ply 0 "saved the mesh to .*verdict=PASS"
  --count 12 --orbit 12 --mesh-every 12 --period-ms 10 --dims 96 --display false
  --ply "${CMAKE_CURRENT_BINARY_DIR}/tsdf_app_test.ply"
)
# The lossless graph on one worker.
tsdf_app_test(tsdf_app_one_worker 0 "plan=OK workers=1[^0-9]" --display false --compile-only true)
if(TSDF_RECONSTRUCTION_ENABLE_VISUALIZATION)
  # HoloViz renders both views offscreen, into CUDA memory.
  tsdf_app_test(tsdf_app_offscreen 0 "showing=[1-9].*verdict=PASS"
    --count 20 --orbit 20 --mesh-every 10 --period-ms 30 --dims 128 --display offscreen
  )
  # The display graph compiles, without a display server, on one worker.
  tsdf_app_test(tsdf_app_display_graph 0 "plan=OK workers=1[^0-9]" --display true --compile-only true)
endif()

# Unusable arguments are usage errors.
tsdf_app_test(tsdf_app_rejects_size 2 "^error: camera images" --width 10)
tsdf_app_test(tsdf_app_rejects_dims 2 "^error: --dims" --dims 1024)
tsdf_app_test(tsdf_app_rejects_mesh_every 2 "^error: --mesh-every" --mesh-every 0)
tsdf_app_test(tsdf_app_rejects_count 2 "^error: --count must be positive" --count 0)
tsdf_app_test(tsdf_app_rejects_depth 2 "^error: --depth" --depth cm)
tsdf_app_test(tsdf_app_rejects_option 2 "^error: unknown option" --frames 3)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "tsdf_app;gpu" TIMEOUT 300)
