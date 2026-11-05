# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests build with the SDK, and all need a GPU and its driver. None needs a
# download: the module's stub model stands in for SAM 3, and the reporter checks every
# instance against the host reference of the whole pipeline. The tests of the real model
# are skipped (exit 77) unless it has been exported to SAM3_SEGMENTATION_DATA_DIR (see the
# README). The module's own suite covers the kernels and the operators.
set(
  SAM3_SEGMENTATION_DATA_DIR
  "${PROJECT_SOURCE_DIR}/../../data/sam3"
  CACHE PATH
  "Where prepare_model.py wrote the SAM 3 export, for the real-model tests"
)

add_executable(sam3_segmentation_make_test_images make_test_images.cpp)
target_link_libraries(sam3_segmentation_make_test_images PRIVATE sam3_segmentation_ops CUDA::nvjpeg)
add_library(sam3_segmentation_spawn_thread_preload SHARED spawn_thread_preload.cpp)

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: sam3_app_test(<name> <exit status> <output regex> <arguments>...). It runs in
# this directory, where `--model stub` writes the stub model and its engines are cached.
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(sam3_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:sam3_segmentation> ${ARGN}
  )
endfunction()

# Builds the stub model's engines once, FP16 and FP32, before the tests that run them, so
# that parallel tests (ctest -j) load the cached engines rather than each building and
# writing them at once.
add_test(
  NAME sam3_app_stub_setup
  COMMAND
    sh -c
    "\"$1\" --model stub --count 1 --period-ms 1 && \"$1\" --model stub --count 1 --period-ms 1 --fp16 false"
    sh $<TARGET_FILE:sam3_segmentation>
)
set_tests_properties(sam3_app_stub_setup PROPERTIES FIXTURES_SETUP sam3_stub)

# Synthetic frames with the stub model, every instance checked against the host
# reference: every encoding, GPU and host memory, wide, tall, small and large frames, FP16
# and FP32, text and box prompts, fewer slots than instances, and files written.
set(_stub --model stub --period-ms 5)
set(_checked "failed=0 worst_pixels=0.*verdict=PASS")
sam3_app_test(sam3_app_synthetic 0 "frames=20/20 prompts=2 instances=40 .*reference: checked=20 ${_checked}"
  ${_stub} --count 20
)
sam3_app_test(sam3_app_bgr8_fp32 0 "frames=20/20 .*reference: checked=20 ${_checked}"
  ${_stub} --count 20 --encoding bgr8 --fp16 false
)
sam3_app_test(sam3_app_rgba8_host 0 "reference: checked=10 ${_checked}"
  ${_stub} --count 10 --encoding rgba8 --host-input true
)
sam3_app_test(sam3_app_bgra8_tall 0 "reference: checked=10 ${_checked}"
  ${_stub} --count 10 --encoding bgra8 --width 720 --height 1280
)
sam3_app_test(sam3_app_small 0 "reference: checked=10 ${_checked}"
  ${_stub} --count 10 --width 64 --height 48
)
sam3_app_test(sam3_app_large 0 "reference: checked=5 ${_checked}"
  ${_stub} --count 5 --width 1920 --height 1080
)
sam3_app_test(sam3_app_box_prompt 0 "prompts=3 instances=30 .*reference: checked=10 ${_checked}"
  ${_stub} --count 10 --box-on triangle
)
sam3_app_test(sam3_app_one_slot 0 "prompts=2 instances=10 .*reference: checked=10 ${_checked}"
  ${_stub} --count 10 --max-instances 1
)
set_tests_properties(
  sam3_app_synthetic
  sam3_app_bgr8_fp32
  sam3_app_rgba8_host
  sam3_app_bgra8_tall
  sam3_app_small
  sam3_app_large
  sam3_app_box_prompt
  sam3_app_one_slot
  PROPERTIES FIXTURES_REQUIRED sam3_stub
)

# The real-time path with a consumer slower than the source: the reporter takes the
# newest instances, skipping frames, and still gets the last.
sam3_app_test(sam3_app_realtime_slow_consumer 0
  "frames=[0-9]+/40 .*skipped=[1-9][0-9]* reference: checked=[0-9]+ ${_checked}"
  ${_stub} --count 40 --period-ms 10 --realtime true --report-delay-ms 60
)
set_tests_properties(sam3_app_realtime_slow_consumer PROPERTIES FIXTURES_REQUIRED sam3_stub)

# Images with their instances written as files.
set(_images "${CMAKE_CURRENT_BINARY_DIR}/images")
add_test(
  NAME sam3_app_image_dir
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=files_written=3 .*verdict=PASS"
    "-DMAKE_DIRECTORY=${_images}" -P "${_run_and_match}" -- $<TARGET_FILE:sam3_segmentation>
    ${_stub} --count 3 --image-dir "${_images}"
)
set_tests_properties(sam3_app_image_dir PROPERTIES FIXTURES_REQUIRED sam3_stub)

# Image files of the synthetic scene, written by the fixture: a baseline JPEG, a PPM and a
# progressive JPEG, decoded as RGB8 or BGR8, each checked against the reference of the
# pixels the segmenter saw, and its instances printed.
set(_files "${CMAKE_CURRENT_BINARY_DIR}/test_images")
add_test(NAME sam3_app_images_setup COMMAND sam3_segmentation_make_test_images "${_files}")
set_tests_properties(sam3_app_images_setup PROPERTIES FIXTURES_SETUP sam3_images)
sam3_app_test(sam3_app_files 0
  "scene_0\\.jpg: 1280x720, 2 instances: red 0\\.90 \\[[0-9 ]+\\], blue 0\\.80 .*scene_1\\.ppm: 333x500, .*scene_2\\.JPEG: 500x333, .*frames=3/3 .*reference: checked=3 ${_checked}"
  ${_stub} --source files --images "${_files}"
)
sam3_app_test(sam3_app_files_bgr8_loop 0 "frames=7/7 .*reference: checked=7 ${_checked}"
  ${_stub} --source files --images "${_files}" --encoding bgr8 --loop true --count 7
)
sam3_app_test(sam3_app_rejects_count_beyond_files 2 "^error: .*--loop true"
  ${_stub} --source files --images "${_files}" --count 4
)
sam3_app_test(sam3_app_rejects_display_with_files 2 "^error: --display shows rgb8"
  ${_stub} --source files --images "${_files}" --display offscreen
)
set_tests_properties(
  sam3_app_files
  sam3_app_files_bgr8_loop
  PROPERTIES FIXTURES_REQUIRED "sam3_stub;sam3_images"
)
set_tests_properties(
  sam3_app_rejects_count_beyond_files
  sam3_app_rejects_display_with_files
  PROPERTIES FIXTURES_REQUIRED sam3_images
)

# HoloViz renders each image under its instances offscreen, into CUDA memory; with the
# stub model and a still scene, every rendered frame is checked against the known one.
set(_render "${CMAKE_CURRENT_BINARY_DIR}/render.ppm")
sam3_app_test(sam3_app_offscreen 0
  "rendered=[1-9][0-9]* render: checked=[1-9][0-9]* failed=0 .*verdict=PASS"
  ${_stub} --count 20 --period-ms 20 --box-on triangle --display offscreen --render-file "${_render}"
)
set_tests_properties(sam3_app_offscreen PROPERTIES FIXTURES_REQUIRED sam3_stub)

# Prompts typed while it runs: a line it can use, and one it cannot.
add_test(
  NAME sam3_app_interactive
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=prompts now: blue.*not in the prompt table.*frames=200/200 .*verdict=PASS" -P
    "${_run_and_match}" -- sh -c
    "(echo blue && sleep 0.3 && echo green) | \"$1\" --model stub --count 200 --period-ms 5 --verify false --interactive true"
    sh $<TARGET_FILE:sam3_segmentation>
)
set_tests_properties(sam3_app_interactive PROPERTIES FIXTURES_REQUIRED sam3_stub)

# Files that are no models fail before the run starts, when their engines are built.
set(_junk "${CMAKE_CURRENT_BINARY_DIR}/junk")
file(MAKE_DIRECTORY "${_junk}")
file(WRITE "${_junk}/sam3_encoder.onnx" "not a model")
file(WRITE "${_junk}/sam3_decoder.onnx" "not a model")
file(WRITE "${_junk}/sam3_prompts.bin" "not a table")
sam3_app_test(sam3_app_rejects_junk_table 1 "error: .*prompt table .*sam3_prompts\\.bin: not a prompt table"
  --model "${_junk}" --count 1
)

# Invalid input fails fast with a usage error (exit status 2) instead of starting a run.
sam3_app_test(sam3_app_rejects_unknown_option 2 "^error: unknown option" ${_stub} --colour 1)
sam3_app_test(sam3_app_rejects_no_model 2 "^error: --model is required" --count 1)
sam3_app_test(sam3_app_rejects_missing_model 2 "^error: --model: no sam3_encoder\\.onnx"
  --model "${CMAKE_CURRENT_BINARY_DIR}/missing"
)
sam3_app_test(sam3_app_rejects_verify_files_without_stub 2
  "^error: --verify with image files needs --model stub"
  --model "${_junk}" --source files --images "${CMAKE_CURRENT_SOURCE_DIR}" --verify true
)
sam3_app_test(sam3_app_rejects_bad_box 2 "^error: prompt boxes are" ${_stub} --box 0.5,0.1,0.2,0.4)
sam3_app_test(sam3_app_rejects_box_text 2 "^error: --box requires" ${_stub} --box 0.1,0.2)
sam3_app_test(sam3_app_rejects_no_prompts 2 "^error: 1\\.\\.16 prompts" ${_stub} --prompts " , ")
sam3_app_test(sam3_app_rejects_threshold 2 "^error: score_threshold" ${_stub} --score-threshold 1)
sam3_app_test(sam3_app_rejects_render_file 2 "^error: --render-file needs --display"
  ${_stub} --render-file x.ppm
)
sam3_app_test(sam3_app_rejects_small_frames 2 "^error: synthetic images are" ${_stub} --width 16)
sam3_app_test(sam3_app_rejects_zero_period 2 "^error: --period-ms" ${_stub} --period-ms 0)
sam3_app_test(sam3_app_rejects_encoding 2 "^error: --encoding" ${_stub} --encoding mono8)
sam3_app_test(sam3_app_rejects_interactive_verify 2 "^error: --interactive changes" ${_stub}
  --interactive true
)
sam3_app_test(sam3_app_rejects_still_with_files 2 "^error: --still, --realtime"
  ${_stub} --source files --images "${CMAKE_CURRENT_SOURCE_DIR}" --still true
)
sam3_app_test(sam3_app_rejects_image_dir 2 "^error: --image-dir is not a directory"
  ${_stub} --image-dir "${_junk}/sam3_encoder.onnx"
)
sam3_app_test(sam3_app_rejects_missing_camera 2 "^error: cannot open camera /dev/video99"
  ${_stub} --source camera --device /dev/video99
)
sam3_app_test(sam3_app_help 0 "^Usage: .*--model DIR\\|stub" --help)
# Only the plan's verdict: an accepted plan has no warnings.
sam3_app_test(sam3_app_compile_only 0 "^sam3_segmentation: plan=OK\n*$" ${_stub} --compile-only true)
sam3_app_test(sam3_app_window_plan 0 "^sam3_segmentation: plan=OK\n*$"
  ${_stub} --display window --compile-only true
)

# A stop request ends the run early with its own verdict, and the warning it leaves is
# shown. The signal goes out once the first frame is written, so the run has started. The
# test sends SIGTERM, as a shell starts background jobs with SIGINT ignored, and its
# script has no semicolons, where CMake would split it.
set(_stopped "${CMAKE_CURRENT_BINARY_DIR}/stopped_images")
add_test(
  NAME sam3_app_stopped
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=1
    "-DEXPECT_OUTPUT=warning RUNTIME_MANAGED_SIGNAL_RECEIVED.*frames=[0-9]+/100000 .*verdict=STOPPED"
    -P "${_run_and_match}" -- sh -c
    "rm -rf \"$2\" && mkdir -p \"$2\" || exit 1
     \"$1\" --model stub --count 100000 --period-ms 10 --width 64 --height 48 --image-dir \"$2\" &
     for i in $(seq 600)
     do
       [ -e \"$2/segment_000001.ppm\" ] && break
       sleep 0.05
     done
     kill -TERM $! && wait $!"
    sh $<TARGET_FILE:sam3_segmentation> "${_stopped}"
)

# Another thread running before main(), as under a profiler or a sanitizer: the run goes
# ahead without managed signal handling, and says so.
add_test(
  NAME sam3_app_without_signal_domain
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS" -P "${_run_and_match}"
    -- ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:sam3_segmentation_spawn_thread_preload>
    $<TARGET_FILE:sam3_segmentation> --model stub --count 10 --period-ms 5
)
set_tests_properties(
  sam3_app_stopped
  sam3_app_without_signal_domain
  PROPERTIES FIXTURES_REQUIRED sam3_stub
)

# The real SAM 3 on the synthetic scene: each prompt's best instance must cover its shape
# with an IoU of at least 0.9, by text and by an exemplar box, and rendered offscreen.
# Skipped (exit 77) unless prepare_model.py has written the export. The first run builds
# and caches the engines, which takes minutes, and longer on a Jetson.
set(_model "${SAM3_SEGMENTATION_DATA_DIR}")
function(sam3_real_test name output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=${output}"
      "-DSKIP_UNLESS_EXISTS=${_model}/sam3_encoder.onnx" -P "${_run_and_match}" --
      $<TARGET_FILE:sam3_segmentation> --model "${_model}" ${ARGN}
  )
  set_tests_properties(
    ${name}
    PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600 RUN_SERIAL TRUE
  )
endfunction()
sam3_real_test(sam3_real_synthetic
  "frames=30/30 prompts=3 .*truth: checked=30 failed=0 .*verdict=PASS" --count 30 --period-ms 10
)
sam3_real_test(sam3_real_box
  "frames=10/10 prompts=1 .*truth: checked=10 failed=0 .*verdict=PASS" --count 10 --period-ms 10
  --prompts " " --box-on circle
)
sam3_real_test(sam3_real_offscreen
  "truth: checked=20 failed=0 .*rendered=[1-9][0-9]*\n.*verdict=PASS" --count 20 --period-ms 20
  --display offscreen --render-file "${CMAKE_CURRENT_BINARY_DIR}/render_real.ppm"
)

get_property(_tests DIRECTORY PROPERTY TESTS)
foreach(_test IN LISTS _tests)
  get_test_property(${_test} TIMEOUT _timeout)
  if(NOT _timeout)
    set_tests_properties(${_test} PROPERTIES TIMEOUT 600)
  endif()
endforeach()
set_tests_properties(${_tests} PROPERTIES LABELS "sam3_segmentation;gpu")

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
