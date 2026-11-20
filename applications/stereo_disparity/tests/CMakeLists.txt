# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA, VPI and a GPU, except the SDK-free ones in host/. The
# real-data tests also need the recordings: they run when HOLOSCAN_TEST_DATA names the
# folder that holds stereo-camera/, and are skipped otherwise.

add_executable(stereo_disparity_write_test_recording write_test_recording.cpp)
target_link_libraries(
  stereo_disparity_write_test_recording
  PRIVATE holoscan::stereo_disparity holoscan::messages CUDA::cudart
)

add_library(stereo_disparity_spawn_thread_preload SHARED spawn_thread_preload.cpp)

add_executable(stereo_disparity_print_backends print_backends.cpp)
target_link_libraries(
  stereo_disparity_print_backends
  PRIVATE holoscan::stereo_disparity holoscan::messages
)
# The module's library loads holoscan-messages' library through this program's run path, so
# keep it linked, although the program calls nothing in it.
target_link_options(stereo_disparity_print_backends BEFORE PRIVATE "LINKER:--no-as-needed")

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

# A recording of the synthetic scene, whose depth is the truth: the reader, its
# camera_info files, the recording source, the camera-depth check and the depth's camera
# info. Its depth is in millimeters.
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
stereo_app_test(stereo_app_recording 0 "camera_depth: .*\\(checked\\).*verdict=PASS"
  --source recording --recording "${_recording}" --backend cuda --camera-depth-scale 0.001
  ${_headless}
)
stereo_app_test(stereo_app_recording_without_depth 0 "verdict=PASS"
  --source recording --recording "${_recording}" --backend cuda ${_headless}
)
stereo_app_test(stereo_app_recording_loop 0 "verdict=PASS"
  --source recording --recording "${_recording}" --loop true --count 9 ${_headless}
)
stereo_app_test(stereo_app_rejects_count_beyond_recording 2 "\nerror: .*--loop true"
  --source recording --recording "${_recording}" --count 7
)
# The whole graph on one worker: CameraInfoSourceOp is a second source, and the camera's
# depth goes to every reporter, which the automatic partitioning would split onto four.
stereo_app_test(stereo_app_recording_one_worker 0 "\nstereo_disparity: plan=OK workers=1\n*$"
  --source recording --recording "${_recording}" --backend all,auto --camera-depth-scale 0.001
  --display false --compile-only true
)
set_tests_properties(
  stereo_app_recording
  stereo_app_recording_without_depth
  stereo_app_recording_loop
  stereo_app_rejects_count_beyond_recording
  stereo_app_recording_one_worker
  PROPERTIES FIXTURES_REQUIRED stereo_recording
)

# The local recordings, through every backend this platform has, against the camera's own
# depth (0.000452 m per unit in these recordings): checked with the projector on; with it
# off, the infrared images are nearly black and the camera finds depth on about 2 % of the
# pixels, so the run must complete and report its valid share, and its accuracy is only
# reported. The clips' folder names are the recordings' own.
set(_local_scale --camera-depth-scale 0.000452)
foreach(_clip 720p_emitter_on 480p_emitter_on 720p_emitter_off 480p_emitter_off)
  if(_clip MATCHES "emitter_on")
    set(_expect "${_checked}")
  else()
    set(_expect " valid=.*reported only.*verdict=PASS")
  endif()
  add_test(
    NAME stereo_app_real_${_clip}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=${_expect}" -DREQUIRE_ENV=HOLOSCAN_TEST_DATA
      -DCLIP_SUFFIX=${_clip} -P "${_run_and_match}" -- $<TARGET_FILE:stereo_disparity> --source
      recording --backend all ${_local_scale} ${_headless}
  )
  set_tests_properties(stereo_app_real_${_clip} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
add_test(
  NAME stereo_app_real_auto
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=${_checked}" -DREQUIRE_ENV=HOLOSCAN_TEST_DATA
    -DCLIP_SUFFIX=720p_emitter_on -P "${_run_and_match}" -- $<TARGET_FILE:stereo_disparity> --source
    recording --backend auto --count 20 ${_local_scale} ${_headless}
)
set_tests_properties(stereo_app_real_auto PROPERTIES SKIP_RETURN_CODE 77)

# Invalid input fails fast with a usage error (exit status 2) instead of starting a run.
stereo_app_test(stereo_app_rejects_unknown_option 2 "^error: unknown option" --colour true)
stereo_app_test(stereo_app_rejects_unknown_backend 2 "^error: unknown backend 'cpu'"
  --backend cpu
)
stereo_app_test(stereo_app_rejects_ofa_max_disparity 2 "^error: max_disparity must be 128 or 256"
  --backend ofa --max-disparity 64
)
stereo_app_test(stereo_app_rejects_cuda_downscale 2
  "^error: downscale needs the ofa or pva backend" --backend cuda --downscale 2
)
stereo_app_test(stereo_app_rejects_small_scene 2 "^error: synthetic images must be"
  --width 100 --height 100
)
stereo_app_test(stereo_app_rejects_missing_recording 2 "^error: .*left.yaml"
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
    $<TARGET_FILE:stereo_disparity> --source recording --clip any_clip
)
# Where the platform has an OFA: ofa and pva downscale their outputs, checked against the
# truth at the downscaled size. Where it has none, asking for it says so. Each is skipped
# on the other kind of platform, which the probe finds when the test runs.
set(_probe -DBACKENDS_PROBE=$<TARGET_FILE:stereo_disparity_print_backends>)
add_test(
  NAME stereo_app_synthetic_downscale
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=truth: .*truth: .*verdict=PASS" ${_probe}
    -DREQUIRE_BACKEND=ofa -P "${_run_and_match}" -- $<TARGET_FILE:stereo_disparity> --backend
    ofa,pva --downscale 4 --count 10 ${_headless}
)
add_test(
  NAME stereo_app_rejects_ofa_without_hardware
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=2 "-DEXPECT_OUTPUT=VPI reports none on this platform" ${_probe}
    -DWITHOUT_BACKEND=ofa -P "${_run_and_match}" -- $<TARGET_FILE:stereo_disparity> --backend ofa
    --count 1 --display false
)
set_tests_properties(
  stereo_app_synthetic_downscale
  stereo_app_rejects_ofa_without_hardware
  PROPERTIES SKIP_RETURN_CODE 77
)

# The graph compiles with the display, without a display server, and runs on one worker.
stereo_app_test(stereo_app_compiles_with_display 0 "^stereo_disparity: plan=OK workers=1\n*$"
  --display true --compile-only true
)
# Only the plan's verdict: an accepted plan has no warnings. Every backend, auto and
# confidence still run on one worker, as the demo lays the graph out; the automatic
# partitioning would give each matcher and each reporter a worker of its own.
stereo_app_test(stereo_app_one_worker 0 "^stereo_disparity: plan=OK workers=1\n*$"
  --backend all,auto --confidence true --display false --compile-only true
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
