# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU, except the SDK-free ones in host/. None
# needs a download: the module's stub model stands in for Fast-FoundationStereo, and the
# demo checks it against the CPU reference of the operator's chain. The tests of the
# real model are skipped (exit 77) unless prepare_model.py has written it to
# FOUNDATION_STEREO_DATA_DIR; those on the local recordings also need HOLOSCAN_TEST_DATA,
# the folder that holds stereo-camera/, which CI never sets.
set(
  FOUNDATION_STEREO_DATA_DIR
  "${PROJECT_SOURCE_DIR}/../../data/foundation_stereo"
  CACHE PATH
  "Where prepare_model.py wrote the Fast-FoundationStereo model, for the real-model tests"
)
set(_model "${FOUNDATION_STEREO_DATA_DIR}/c_fast_foundationstereo_480x736.onnx")

add_executable(foundation_stereo_write_test_recording write_test_recording.cpp)
target_link_libraries(
  foundation_stereo_write_test_recording
  PRIVATE holoscan::foundation_stereo holoscan::messages CUDA::cudart
)

add_library(foundation_stereo_spawn_thread_preload SHARED spawn_thread_preload.cpp)

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: stereo_app_test(<name> <exit status> <output regex> <arguments>...). `--model stub`
# writes the stub model, and its engine is cached, under XDG_CACHE_HOME, which every test
# sets to this directory's cache/.
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(stereo_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:foundation_stereo> ${ARGN}
  )
endfunction()
# The same for the real model, skipped (exit 77) unless it is prepared and, with
# REQUIRE_ENV, the variable is set; CLIP_SUFFIX plays the local recording whose folder name
# ends in it.
function(stereo_real_test name output)
  cmake_parse_arguments(REAL "" "REQUIRE_ENV;CLIP_SUFFIX" "" ${ARGN})
  set(_options)
  if(REAL_REQUIRE_ENV)
    list(APPEND _options -DREQUIRE_ENV=${REAL_REQUIRE_ENV})
  endif()
  if(REAL_CLIP_SUFFIX)
    list(APPEND _options -DCLIP_SUFFIX=${REAL_CLIP_SUFFIX})
  endif()
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=${output}" ${_options}
      "-DSKIP_UNLESS_EXISTS=${_model}" -P "${_run_and_match}" -- $<TARGET_FILE:foundation_stereo>
      --model "${_model}" ${REAL_UNPARSED_ARGUMENTS}
  )
  # The first builds the engine: about 2 minutes on an RTX 5090 and 45 on a Jetson Orin.
  set_tests_properties(
    ${name}
    PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 3600
  )
endfunction()

# One run first builds and caches the stub's FP16 engine, so that the tests that share it
# never build it at once; the FP32 test has a cache of its own, locked for it.
add_test(
  NAME foundation_stereo_app_stub_setup
  COMMAND foundation_stereo --model stub --count 1 --display false --period-ms 1
)
set_tests_properties(foundation_stereo_app_stub_setup PROPERTIES FIXTURES_SETUP stereo_stub)

# The stub model on the synthetic scene, each frame against the CPU reference: RGB8 in
# device memory with an FP16 engine, MONO8 in host memory with an FP32 one, and in real
# time with drop-oldest edges.
set(_headless --display false --period-ms 10)
set(_stub_reference "stub_reference: pixels=[1-9][0-9]* wrong=0 ")
set(_stub_ok "${_stub_reference}.*verdict=PASS")
stereo_app_test(foundation_stereo_app_stub 0 "frames=10/10 .*${_stub_ok}"
  --model stub --count 10 ${_headless}
)
stereo_app_test(foundation_stereo_app_stub_mono_host_fp32 0 "frames=10/10 .*${_stub_ok}"
  --model stub --count 10 --color false --device-input false --fp16 false --width 640
  --height 480 ${_headless}
)
stereo_app_test(foundation_stereo_app_stub_realtime 0 "${_stub_ok}"
  --model stub --count 30 --realtime true --width 640 --height 480 --display false --period-ms 5
)
stereo_app_test(foundation_stereo_app_stub_without_cuda_graphs 0 "frames=5/5 .*${_stub_ok}"
  --model stub --count 5 --cuda-graphs false ${_headless}
)

# The display, rendered offscreen into GPU memory: the left image in the left half, the
# colorized disparity in the right half; with the color pair and with a MONO8 recording.
stereo_app_test(foundation_stereo_app_offscreen 0
  "${_stub_reference}.*offscreen: frames=[1-9][0-9]* failed=0 .*snapshot=.*\nfoundation_stereo: verdict=PASS"
  --model stub --count 20 --width 640 --height 480 --display offscreen --snapshot
  "${CMAKE_CURRENT_BINARY_DIR}/offscreen.ppm" --period-ms 20
)

# A recording of the synthetic scene, whose depth is the truth: the reader, its
# camera_info files, the recording source, the depth and its camera info, and the
# camera-depth comparison, with the stub, whose depth is only reported; with the real
# model, it is checked. Its depth is in millimeters.
set(_recording "${CMAKE_CURRENT_BINARY_DIR}/test_recording")
add_test(
  NAME foundation_stereo_app_recording_setup
  COMMAND $<TARGET_FILE:foundation_stereo_write_test_recording> "${_recording}" 6 640 480
)
set_tests_properties(
  foundation_stereo_app_recording_setup
  PROPERTIES FIXTURES_SETUP stereo_recording
)
stereo_app_test(foundation_stereo_app_recording_stub 0
  "frames=6/6 .*camera_depth: .*\\(stub model; reported only\\).*verdict=PASS"
  --model stub --source recording --recording "${_recording}" --camera-depth-scale 0.001
  ${_headless}
)
stereo_app_test(foundation_stereo_app_recording_without_depth 0 "frames=6/6 .*verdict=PASS"
  --model stub --source recording --recording "${_recording}" ${_headless}
)
stereo_app_test(foundation_stereo_app_recording_loop 0 "frames=9/9 .*verdict=PASS"
  --model stub --source recording --recording "${_recording}" --loop true --count 9
  ${_headless}
)
# In real time, with the camera's depth joined to each processed frame on drop-oldest
# edges: the run completes and passes, whatever it drops.
stereo_app_test(foundation_stereo_app_recording_stub_realtime 0
  "camera_depth: .*verdict=PASS"
  --model stub --source recording --recording "${_recording}" --loop true --count 30
  --camera-depth-scale 0.001 --realtime true --display false --period-ms 5
)
stereo_app_test(foundation_stereo_app_recording_offscreen 0
  "offscreen: frames=[1-9][0-9]* failed=0 .*\nfoundation_stereo: verdict=PASS"
  --model stub --source recording --recording "${_recording}" --display offscreen
  --period-ms 20
)
stereo_app_test(foundation_stereo_app_rejects_count_beyond_recording 2 "\nerror: .*--loop true"
  --model stub --source recording --recording "${_recording}" --count 7
)
# The whole lossless graph on one worker: CameraInfoSourceOp is a second source, and the
# camera's depth and the display fan out, which the automatic partitioning would split.
stereo_app_test(foundation_stereo_app_recording_one_worker 0
  "\nfoundation_stereo: plan=OK workers=1\n*$"
  --model stub --source recording --recording "${_recording}" --camera-depth-scale 0.001
  --display offscreen --compile-only true
)

# The real model: the synthetic scene against its truth, in color and in gray, and the
# synthetic recording against its true depth. Skipped unless the model is prepared.
set(_truth "truth: .*verdict=PASS")
set(_checked "camera_depth: .*\\(checked\\).*verdict=PASS")
stereo_real_test(foundation_stereo_app_real_synthetic "frames=20/20 .*${_truth}"
  --count 20 ${_headless}
)
stereo_real_test(foundation_stereo_app_real_synthetic_mono "frames=20/20 .*${_truth}"
  --count 20 --color false --device-input false --width 848 --height 480 ${_headless}
)
stereo_real_test(foundation_stereo_app_real_recording "frames=6/6 .*${_checked}"
  --source recording --recording "${_recording}" --camera-depth-scale 0.001 ${_headless}
)
set_tests_properties(
  foundation_stereo_app_real_recording
  PROPERTIES FIXTURES_REQUIRED stereo_recording
)
# The local recordings against the camera's own depth (0.000452 m per unit in these
# recordings): checked with the projector on; with it off the camera has depth on about 2 %
# of the pixels, so the run must complete and its accuracy is only reported. Local only:
# they also need HOLOSCAN_TEST_DATA. The clips' folder names are the recordings' own.
set(_local_scale --camera-depth-scale 0.000452)
foreach(_clip 720p_emitter_on 480p_emitter_on 720p_emitter_off 480p_emitter_off)
  if(_clip MATCHES "emitter_on")
    set(_expect "frames=60/60 .*${_checked}")
  else()
    set(_expect "frames=60/60 .*reported only.*verdict=PASS")
  endif()
  stereo_real_test(foundation_stereo_app_real_${_clip} "${_expect}"
    REQUIRE_ENV HOLOSCAN_TEST_DATA CLIP_SUFFIX ${_clip}
    --source recording ${_local_scale} ${_headless}
  )
endforeach()
# The same recording played in real time, as from the camera: frames the model cannot keep
# up with are dropped, and those it processes are checked.
stereo_real_test(foundation_stereo_app_real_realtime "camera_depth: .*verdict=PASS"
  REQUIRE_ENV HOLOSCAN_TEST_DATA CLIP_SUFFIX 720p_emitter_on
  --source recording ${_local_scale} --realtime true --count 30 --display false
)

get_property(_stub_tests DIRECTORY PROPERTY TESTS)
list(FILTER _stub_tests INCLUDE REGEX "^foundation_stereo_app_(stub_|offscreen|recording_)")
list(REMOVE_ITEM _stub_tests foundation_stereo_app_stub_setup foundation_stereo_app_recording_setup)
set_tests_properties(${_stub_tests} PROPERTIES FIXTURES_REQUIRED "stereo_stub")
set_tests_properties(
  foundation_stereo_app_recording_stub
  foundation_stereo_app_recording_without_depth
  foundation_stereo_app_recording_loop
  foundation_stereo_app_recording_stub_realtime
  foundation_stereo_app_recording_offscreen
  foundation_stereo_app_rejects_count_beyond_recording
  foundation_stereo_app_recording_one_worker
  PROPERTIES FIXTURES_REQUIRED "stereo_stub;stereo_recording"
)
set_tests_properties(
  foundation_stereo_app_stub_mono_host_fp32
  PROPERTIES RESOURCE_LOCK foundation_stereo_stub_fp32
)

# Invalid input fails fast with a usage error (exit status 2) instead of starting a run.
stereo_app_test(foundation_stereo_app_rejects_unknown_option 2 "^error: unknown option"
  --model stub --colour true
)
stereo_app_test(foundation_stereo_app_rejects_missing_model 2 "^error: --model PATH or --model stub"
  --count 1
)
stereo_app_test(foundation_stereo_app_rejects_absent_model 2 "^error: no model at .*prepare_model"
  --model "${CMAKE_CURRENT_BINARY_DIR}/no_such_model.onnx"
)
stereo_app_test(foundation_stereo_app_rejects_small_scene 2 "^error: synthetic images must be"
  --model stub --width 100 --height 100
)
stereo_app_test(foundation_stereo_app_rejects_missing_recording 2 "^error: .*left.yaml"
  --model stub --source recording --recording "${CMAKE_CURRENT_BINARY_DIR}/no_such_recording"
)
stereo_app_test(foundation_stereo_app_rejects_snapshot_without_offscreen 2
  "^error: --snapshot needs --display offscreen"
  --model stub --snapshot x.ppm
)
stereo_app_test(foundation_stereo_app_rejects_camera_depth_without_recording 2
  "^error: --recording, --clip, --loop and --camera-depth-scale apply to --source recording"
  --model stub --camera-depth-scale 0.001
)
add_test(
  NAME foundation_stereo_app_rejects_clip_without_data
  COMMAND
    ${CMAKE_COMMAND} -E env --unset=HOLOSCAN_TEST_DATA ${CMAKE_COMMAND} -DEXPECT_EXIT=2
    "-DEXPECT_OUTPUT=^error: --clip needs HOLOSCAN_TEST_DATA" -P "${_run_and_match}" --
    $<TARGET_FILE:foundation_stereo> --model stub --source recording --clip any_clip
)

# The graph compiles with a window, without a display server, and the ci mode's graph (the
# stub on the synthetic scene, rendered offscreen) runs on one worker.
stereo_app_test(foundation_stereo_app_compiles_with_display 0
  "^foundation_stereo: plan=OK workers=1\n*$"
  --model stub --display true --compile-only true
)
stereo_app_test(foundation_stereo_app_one_worker 0 "^foundation_stereo: plan=OK workers=1\n*$"
  --model stub --count 30 --width 640 --height 480 --period-ms 50 --display offscreen
  --compile-only true
)

# Another thread running before main(), as under a profiler or a sanitizer: the run goes
# ahead without managed signal handling, and says so.
add_test(
  NAME foundation_stereo_app_without_signal_domain
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS" -P "${_run_and_match}"
    -- ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:foundation_stereo_spawn_thread_preload>
    $<TARGET_FILE:foundation_stereo> --model stub --count 5 --width 320 --height 240 ${_headless}
)

set_tests_properties(
  foundation_stereo_app_without_signal_domain
  PROPERTIES FIXTURES_REQUIRED stereo_stub
)

get_property(_tests DIRECTORY PROPERTY TESTS)
foreach(_test IN LISTS _tests)
  get_test_property(${_test} TIMEOUT _timeout)
  if(NOT _timeout)
    set_tests_properties(${_test} PROPERTIES TIMEOUT 600)
  endif()
endforeach()
set_tests_properties(
  ${_tests}
  PROPERTIES
    LABELS "foundation_stereo;gpu"
    ENVIRONMENT "XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/cache"
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
