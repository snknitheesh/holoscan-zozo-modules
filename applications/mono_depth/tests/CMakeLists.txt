# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests build with the SDK, and all need a GPU and its driver. None needs a
# download: the module's stub model stands in for Depth Anything, and the reporter
# checks every depth value against the host reference of the whole pipeline. The tests
# of the real model are skipped (exit 77) unless it has been exported to
# MONO_DEPTH_DATA_DIR (see the README). The module's own suite covers the kernels and the
# operator.
set(
  MONO_DEPTH_DATA_DIR
  "${PROJECT_SOURCE_DIR}/../../data/mono_depth"
  CACHE PATH
  "Where prepare_model.py wrote the Depth Anything V2 model, for the real-model tests"
)

add_executable(mono_depth_make_test_images make_test_images.cpp)
target_link_libraries(mono_depth_make_test_images PRIVATE mono_depth_ops CUDA::nvjpeg)
add_library(mono_depth_spawn_thread_preload SHARED spawn_thread_preload.cpp)

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: mono_depth_app_test(<name> <exit status> <output regex> <arguments>...). It runs
# in this directory, where `--model stub` writes the stub model and its engines are cached.
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(mono_depth_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:mono_depth> ${ARGN}
  )
endfunction()

# Builds the stub model's engines once, FP16 and FP32, before the tests that run them, so
# that parallel tests (ctest -j) load the cached engines rather than each building and
# writing them at once. Other input sizes have caches of their own, each locked for its
# users.
add_test(
  NAME mono_depth_stub_setup
  COMMAND
    sh -c
    "\"$1\" --model stub --count 1 --period-ms 1 && \"$1\" --model stub --count 1 --period-ms 1 --fp16 false"
    sh $<TARGET_FILE:mono_depth>
)
set_tests_properties(mono_depth_stub_setup PROPERTIES FIXTURES_SETUP mono_depth_stub)

# Synthetic frames with the stub model, every depth value checked against the host
# reference: every encoding, GPU and host memory, wide, tall, small and large frames,
# letterboxed and stretched, FP16 and FP32, metric outputs, and colorized files.
set(_stub --model stub --period-ms 5)
set(_checked "failed=0 max_error=[0-9.e+-]+.*verdict=PASS")
mono_depth_app_test(mono_depth_app_synthetic 0 "frames=20/20 .*reference: checked=20 ${_checked}"
  ${_stub} --count 20
)
mono_depth_app_test(mono_depth_app_bgr8_fp32 0 "frames=20/20 .*reference: checked=20 ${_checked}"
  ${_stub} --count 20 --encoding bgr8 --fp16 false
)
mono_depth_app_test(mono_depth_app_rgba8_host 0 "reference: checked=10 ${_checked}"
  ${_stub} --count 10 --encoding rgba8 --host-input true
)
mono_depth_app_test(mono_depth_app_bgra8_tall 0 "reference: checked=10 ${_checked}"
  ${_stub} --count 10 --encoding bgra8 --width 720 --height 1280
)
mono_depth_app_test(mono_depth_app_small 0 "reference: checked=10 ${_checked}"
  ${_stub} --count 10 --width 64 --height 48
)
mono_depth_app_test(mono_depth_app_large 0 "reference: checked=5 ${_checked}"
  ${_stub} --count 5 --width 1920 --height 1200
)
mono_depth_app_test(mono_depth_app_metric 0 "meters=[0-9.]+\\.\\.[0-9.]+ reference: checked=10 ${_checked}"
  ${_stub} --count 10 --metric-scale 0.4 --metric-shift 0.1
)
mono_depth_app_test(mono_depth_app_metric_model 0 "meters=[0-9.]+\\.\\.[0-9.]+ reference: checked=10 ${_checked}"
  ${_stub} --count 10 --model-output metric
)
set_tests_properties(
  mono_depth_app_synthetic
  mono_depth_app_bgr8_fp32
  mono_depth_app_rgba8_host
  mono_depth_app_bgra8_tall
  mono_depth_app_small
  mono_depth_app_large
  mono_depth_app_metric
  mono_depth_app_metric_model
  PROPERTIES FIXTURES_REQUIRED mono_depth_stub
)
mono_depth_app_test(mono_depth_app_stretch_wide_input 0 "reference: checked=10 ${_checked}"
  ${_stub} --count 10 --resize stretch --input-width 518 --input-height 294
)
set_tests_properties(
  mono_depth_app_stretch_wide_input
  PROPERTIES RESOURCE_LOCK mono_depth_stub_518x294
)

# The real-time path with a consumer slower than the source: the reporter takes the
# newest depth, skipping frames, and still gets the last.
mono_depth_app_test(mono_depth_app_realtime_slow_consumer 0
  "frames=[0-9]+/40 .*skipped=[1-9][0-9]* reference: checked=[0-9]+ ${_checked}"
  ${_stub} --count 40 --period-ms 10 --realtime true --report-delay-ms 60
)
set_tests_properties(
  mono_depth_app_realtime_slow_consumer
  PROPERTIES FIXTURES_REQUIRED mono_depth_stub
)

# Colorized depth written as files.
set(_images "${CMAKE_CURRENT_BINARY_DIR}/images")
add_test(
  NAME mono_depth_app_image_dir
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=files_written=3 .*verdict=PASS"
    "-DMAKE_DIRECTORY=${_images}" -P "${_run_and_match}" -- $<TARGET_FILE:mono_depth> ${_stub}
    --count 3 --image-dir "${_images}"
)
set_tests_properties(mono_depth_app_image_dir PROPERTIES FIXTURES_REQUIRED mono_depth_stub)

# Image files of the synthetic scene, written by the fixture: a baseline JPEG, a PPM and a
# progressive JPEG, decoded as RGB8 or BGR8, each checked against the reference of the
# pixels the estimator saw.
set(_files "${CMAKE_CURRENT_BINARY_DIR}/test_images")
add_test(NAME mono_depth_images_setup COMMAND mono_depth_make_test_images "${_files}")
set_tests_properties(mono_depth_images_setup PROPERTIES FIXTURES_SETUP mono_depth_images)
mono_depth_app_test(mono_depth_app_files 0
  "scene_0\\.jpg: 1280x720 inverse depth [0-9.]+\\.\\.[0-9.]+\nmono_depth: scene_1\\.ppm: 333x500 .*scene_2\\.JPEG: 500x333 .*frames=3/3 .*reference: checked=3 ${_checked}"
  ${_stub} --source files --images "${_files}"
)
mono_depth_app_test(mono_depth_app_files_bgr8_loop 0 "frames=7/7 .*reference: checked=7 ${_checked}"
  ${_stub} --source files --images "${_files}" --encoding bgr8 --loop true --count 7
)
mono_depth_app_test(mono_depth_app_rejects_count_beyond_files 2 "^error: .*--loop true"
  ${_stub} --source files --images "${_files}" --count 4
)
mono_depth_app_test(mono_depth_app_rejects_rgba8_files 2 "^error: image files are decoded as rgb8"
  ${_stub} --source files --images "${_files}" --encoding rgba8
)
mono_depth_app_test(mono_depth_app_rejects_display_with_files 2 "^error: --display shows rgb8"
  ${_stub} --source files --images "${_files}" --display offscreen
)
set_tests_properties(
  mono_depth_app_files
  mono_depth_app_files_bgr8_loop
  PROPERTIES FIXTURES_REQUIRED "mono_depth_stub;mono_depth_images"
)
set_tests_properties(
  mono_depth_app_rejects_count_beyond_files
  mono_depth_app_rejects_rgba8_files
  mono_depth_app_rejects_display_with_files
  PROPERTIES FIXTURES_REQUIRED mono_depth_images
)

# HoloViz renders each image beside its colorized depth offscreen, into CUDA memory; with
# the stub model and a still scene, every rendered frame is checked against the known one.
set(_render "${CMAKE_CURRENT_BINARY_DIR}/render.ppm")
mono_depth_app_test(mono_depth_app_offscreen 0
  "rendered=[1-9][0-9]* render: checked=[1-9][0-9]* failed=0 .*verdict=PASS"
  ${_stub} --count 20 --period-ms 20 --still true --display offscreen --render-file "${_render}"
)
set_tests_properties(mono_depth_app_offscreen PROPERTIES FIXTURES_REQUIRED mono_depth_stub)

# A file that is no model fails before the run starts, when its engine is built.
set(_junk "${CMAKE_CURRENT_BINARY_DIR}/junk.onnx")
file(WRITE "${_junk}" "not a model")
mono_depth_app_test(mono_depth_app_rejects_junk_model 1 "error: MonoDepthOp: cannot load .*junk\\.onnx"
  --model "${_junk}" --count 1
)

# Invalid input fails fast with a usage error (exit status 2) instead of starting a run.
mono_depth_app_test(mono_depth_app_rejects_unknown_option 2 "^error: unknown option" ${_stub}
  --colour 1
)
mono_depth_app_test(mono_depth_app_rejects_no_model 2 "^error: --model is required" --count 1)
mono_depth_app_test(mono_depth_app_rejects_missing_model 2 "^error: --model: no such file"
  --model "${CMAKE_CURRENT_BINARY_DIR}/missing.onnx"
)
mono_depth_app_test(mono_depth_app_rejects_verify_files_without_stub 2
  "^error: --verify with image files needs --model stub"
  --model "${_junk}" --source files --images "${CMAKE_CURRENT_SOURCE_DIR}" --verify true
)
mono_depth_app_test(mono_depth_app_rejects_metric_twice 2 "^error: --metric-scale converts"
  ${_stub} --model-output metric --metric-scale 1
)
mono_depth_app_test(mono_depth_app_rejects_half_color_range 2 "^error: --color-min and --color-max"
  ${_stub} --color-min 1
)
mono_depth_app_test(mono_depth_app_rejects_color_range 2 "^error: color_range must be finite"
  ${_stub} --color-min 2 --color-max 1
)
mono_depth_app_test(mono_depth_app_rejects_render_file 2 "^error: --render-file needs --display"
  ${_stub} --render-file x.ppm
)
mono_depth_app_test(mono_depth_app_rejects_small_frames 2 "^error: synthetic images must be"
  ${_stub} --width 16
)
mono_depth_app_test(mono_depth_app_rejects_zero_period 2 "^error: --period-ms" ${_stub}
  --period-ms 0
)
mono_depth_app_test(mono_depth_app_rejects_encoding 2 "^error: --encoding" ${_stub} --encoding mono8)
mono_depth_app_test(mono_depth_app_rejects_still_with_files 2 "^error: --still, --realtime"
  ${_stub} --source files --images "${CMAKE_CURRENT_SOURCE_DIR}" --still true
)
mono_depth_app_test(mono_depth_app_rejects_image_dir 2 "^error: --image-dir is not a directory"
  ${_stub} --image-dir "${_junk}"
)
mono_depth_app_test(mono_depth_app_rejects_missing_camera 2 "^error: cannot open camera /dev/video99"
  ${_stub} --source camera --device /dev/video99
)
mono_depth_app_test(mono_depth_app_help 0 "^Usage: .*--model PATH\\|stub" --help)
# Only the plan's verdict: an accepted plan has no warnings.
mono_depth_app_test(mono_depth_app_compile_only 0 "^mono_depth: plan=OK\n*$"
  ${_stub} --compile-only true
)
mono_depth_app_test(mono_depth_app_window_plan 0 "^mono_depth: plan=OK\n*$"
  ${_stub} --display window --compile-only true
)

# A stop request ends the run early with its own verdict, and the warning it leaves is
# shown. The signal goes out once the first frame is written, so the run has started. The
# test sends SIGTERM, as a shell starts background jobs with SIGINT ignored, and its
# script has no semicolons, where CMake would split it.
set(_stopped "${CMAKE_CURRENT_BINARY_DIR}/stopped_images")
add_test(
  NAME mono_depth_app_stopped
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=1
    "-DEXPECT_OUTPUT=warning RUNTIME_MANAGED_SIGNAL_RECEIVED.*frames=[0-9]+/100000 .*verdict=STOPPED"
    -P "${_run_and_match}" -- sh -c
    "rm -rf \"$2\" && mkdir -p \"$2\" || exit 1
     \"$1\" --model stub --count 100000 --period-ms 10 --width 64 --height 48 --image-dir \"$2\" &
     for i in $(seq 600)
     do
       [ -e \"$2/depth_000001.ppm\" ] && break
       sleep 0.05
     done
     kill -TERM $! && wait $!"
    sh $<TARGET_FILE:mono_depth> "${_stopped}"
)

# Another thread running before main(), as under a profiler or a sanitizer: the run goes
# ahead without managed signal handling, and says so.
add_test(
  NAME mono_depth_app_without_signal_domain
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS" -P "${_run_and_match}"
    -- ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:mono_depth_spawn_thread_preload>
    $<TARGET_FILE:mono_depth> --model stub --count 10 --period-ms 5
)
set_tests_properties(
  mono_depth_app_stopped
  mono_depth_app_without_signal_domain
  PROPERTIES FIXTURES_REQUIRED mono_depth_stub
)

# The real Depth Anything V2 Small model on the synthetic scene: every frame's depth must
# rank the pixels as the true depth does, Spearman's correlation at least 0.85, and the
# near panel, the sphere and the far wall in that order. Letterboxed and stretched, and
# rendered offscreen. Skipped (exit 77) unless prepare_model.py has written the model. The
# first run builds and caches the engine, which takes minutes on a Jetson.
set(_model "${MONO_DEPTH_DATA_DIR}/depth_anything_v2_small_518x518.onnx")
function(mono_depth_real_test name output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=${output}" "-DSKIP_UNLESS_EXISTS=${_model}"
      -P "${_run_and_match}" -- $<TARGET_FILE:mono_depth> --model "${_model}" ${ARGN}
  )
  set_tests_properties(
    ${name}
    PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600 RUN_SERIAL TRUE
  )
endfunction()
mono_depth_real_test(mono_depth_real_synthetic
  "frames=30/30 .*ordering: checked=30 failed=0 .*verdict=PASS" --count 30 --period-ms 10
)
mono_depth_real_test(mono_depth_real_stretch
  "frames=30/30 .*ordering: checked=30 failed=0 .*verdict=PASS" --count 30 --period-ms 10
  --resize stretch
)
mono_depth_real_test(mono_depth_real_offscreen
  "ordering: checked=20 failed=0 .*rendered=[1-9][0-9]*\n.*verdict=PASS" --count 20 --period-ms 20
  --display offscreen --render-file "${CMAKE_CURRENT_BINARY_DIR}/render_real.ppm"
)

get_property(_tests DIRECTORY PROPERTY TESTS)
foreach(_test IN LISTS _tests)
  get_test_property(${_test} TIMEOUT _timeout)
  if(NOT _timeout)
    set_tests_properties(${_test} PROPERTIES TIMEOUT 600)
  endif()
endforeach()
set_tests_properties(${_tests} PROPERTIES LABELS "mono_depth_app;gpu")

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
