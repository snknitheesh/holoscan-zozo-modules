# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA, VPI and a GPU, except the SDK-free ones in host/. The
# real-data tests also need the recordings: they run when HOLOSCAN_TEST_DATA names the
# folder that holds stereo-camera/, and are skipped otherwise.

add_executable(stereo_disparity_write_test_recording write_test_recording.cpp)
target_link_libraries(
  stereo_disparity_write_test_recording
  PRIVATE holoscan::stereo_disparity CUDA::cudart
)

add_library(stereo_disparity_spawn_thread_preload SHARED spawn_thread_preload.cpp)

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: stereo_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(stereo_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:stereo_disparity> ${ARGN}
  )
endfunction()

# The synthetic scene through every backend this platform has, and auto, each checked
# against the truth; RGB8 in device memory and MONO8 in host memory; with confidence.
set(_headless --display false --period-ms 10)
stereo_app_test(stereo_app_synthetic 0 "truth: .*verdict=PASS"
  --backend all,auto --count 10 ${_headless}
)
stereo_app_test(stereo_app_synthetic_mono_host 0 "truth: .*verdict=PASS"
  --backend all --count 10 --color false --device-input false --width 640 --height 480
  ${_headless}
)
stereo_app_test(stereo_app_synthetic_confidence 0 "mean_confidence=.*verdict=PASS"
  --backend cuda --count 5 --confidence true --width 640 --height 480 ${_headless}
)

# A recording of the synthetic scene, whose depth is the truth: the reader, the
# recording source, configured and per-frame geometry, and the camera-depth check.
set(_recording "${CMAKE_CURRENT_BINARY_DIR}/test_recording")
add_test(
  NAME stereo_app_recording_setup
  COMMAND $<TARGET_FILE:stereo_disparity_write_test_recording> "${_recording}" 6 640 480
)
set_tests_properties(stereo_app_recording_setup PROPERTIES FIXTURES_SETUP stereo_recording)
set(_checked "camera_depth: .*\\(checked\\).*verdict=PASS")
# The plumbing, on cuda: on Jetson Thor, ofa and pva give whole-pixel disparities, too coarse
# for this scene's background (6.4 px) to meet the 3 % median; the real clips and the
# synthetic scene check every backend.
stereo_app_test(stereo_app_recording 0 "frames=6/6 .*${_checked}"
  --source recording --recording "${_recording}" --backend cuda ${_headless}
)
stereo_app_test(stereo_app_recording_geometry_input 0 "frames=6/6 .*${_checked}"
  --source recording --recording "${_recording}" --geometry-input true ${_headless}
)
stereo_app_test(stereo_app_recording_loop 0 "frames=9/9 .*verdict=PASS"
  --source recording --recording "${_recording}" --loop true --count 9 ${_headless}
)
stereo_app_test(stereo_app_rejects_count_beyond_recording 2 "\nerror: .*--loop true"
  --source recording --recording "${_recording}" --count 7
)
set_tests_properties(
  stereo_app_recording
  stereo_app_recording_geometry_input
  stereo_app_recording_loop
  stereo_app_rejects_count_beyond_recording
  PROPERTIES FIXTURES_REQUIRED stereo_recording
)

# The four recorded clips, through every backend this platform has, against the
# camera's own depth: checked with the projector on; with it off, the infrared images
# are nearly black and the camera finds depth on about 2 % of the pixels, so the run
# must complete and report its valid share, and its accuracy is only reported.
foreach(_clip d585_720p_emitter_on d585_480p_emitter_on d585_720p_emitter_off d585_480p_emitter_off)
  if(_clip MATCHES "emitter_on")
    set(_expect "frames=60/60 .*${_checked}")
  else()
    set(_expect "frames=60/60 .* valid=.*reported only.*verdict=PASS")
  endif()
  add_test(
    NAME stereo_app_real_${_clip}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=${_expect}" -DREQUIRE_ENV=HOLOSCAN_TEST_DATA
      -P "${_run_and_match}" -- $<TARGET_FILE:stereo_disparity> --source recording --clip ${_clip}
      --backend all ${_headless}
  )
  set_tests_properties(stereo_app_real_${_clip} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED: ")
endforeach()
add_test(
  NAME stereo_app_real_geometry_input
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=${_checked}" -DREQUIRE_ENV=HOLOSCAN_TEST_DATA
    -P "${_run_and_match}" -- $<TARGET_FILE:stereo_disparity> --source recording --clip
    d585_720p_emitter_on --backend auto --geometry-input true --count 20 ${_headless}
)
set_tests_properties(stereo_app_real_geometry_input PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED: ")

# Invalid input fails fast with a usage error (exit status 2) instead of starting a run.
stereo_app_test(stereo_app_rejects_unknown_option 2 "^error: unknown option" --colour true)
stereo_app_test(stereo_app_rejects_unknown_backend 2 "^error: unknown backend 'cpu'"
  --backend cpu
)
stereo_app_test(stereo_app_rejects_ofa_max_disparity 2 "^error: max_disparity must be 128 or 256"
  --backend ofa --max-disparity 64
)
stereo_app_test(stereo_app_rejects_small_scene 2 "^error: synthetic images must be"
  --width 100 --height 100
)
stereo_app_test(stereo_app_rejects_missing_recording 2 "^error: .*cannot open"
  --source recording --recording "${CMAKE_CURRENT_BINARY_DIR}/no_such_recording"
)
stereo_app_test(stereo_app_rejects_recording_without_folder 2 "^error: .*--recording DIR"
  --source recording
)
add_test(
  NAME stereo_app_rejects_clip_without_data
  COMMAND
    ${CMAKE_COMMAND} -E env --unset=HOLOSCAN_TEST_DATA ${CMAKE_COMMAND} -DEXPECT_EXIT=2
    "-DEXPECT_OUTPUT=^error: --clip needs HOLOSCAN_TEST_DATA" -P "${_run_and_match}" --
    $<TARGET_FILE:stereo_disparity> --source recording --clip d585_720p_emitter_on
)
# Where the platform has no OFA, asking for it says so.
if(NOT "$ENV{HOLOSCAN_CI_PLATFORM}" STREQUAL "jetson")
  stereo_app_test(stereo_app_rejects_ofa_without_hardware 2 "VPI reports none on this platform"
    --backend ofa --count 1 --display false
  )
endif()

# The graph compiles with the display, without a display server.
stereo_app_test(stereo_app_compiles_with_display 0 "plan=OK"
  --display true --compile-only true
)

# Another thread running before main(), as under a profiler or a sanitizer: the run goes
# ahead without managed signal handling, and says so.
add_test(
  NAME stereo_app_without_signal_domain
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS" -P "${_run_and_match}"
    -- ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:stereo_disparity_spawn_thread_preload>
    $<TARGET_FILE:stereo_disparity> --count 5 --width 320 --height 240 ${_headless}
)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "stereo_disparity;gpu" TIMEOUT 600)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
