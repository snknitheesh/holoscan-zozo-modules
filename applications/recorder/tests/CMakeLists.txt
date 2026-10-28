# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The demo's tests: each runs the demo through run_and_match.cmake, which checks its exit
# status as well as its output: recorder_app_test(<name> <exit status> <output regex>
# <arguments>...). They need a GPU. The module's own suite covers the operators in depth.
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(recorder_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:recorder> ${ARGN}
  )
endfunction()

set(_out "${CMAKE_CURRENT_BINARY_DIR}")
# Record and replay, byte for byte: the ci mode, and a larger one.
recorder_app_test(recorder_app_ci 0
  "recorded=150/150 dropped=0 .*replayed to host: samples=150/150 mismatches=0\n.*replayed to device: samples=150/150 mismatches=0\n.*verdict=PASS"
  --mode ci --recording "${_out}/ci"
)
recorder_app_test(recorder_app_ci_720p 0 "recorded=50/50 dropped=0 .*verdict=PASS"
  --mode ci --frames 10 --width 1280 --height 720 --recording "${_out}/ci_720p"
)
# What ci recorded, replayed in real time and as fast as possible, headless: 30
# frames 33 ms apart take about a second in real time.
recorder_app_test(recorder_app_replay_realtime 0
  "frames=30/30 emitted=30 .*verdict=PASS" --mode replay --recording "${_out}/ci" --display false
)
recorder_app_test(recorder_app_replay_fast_loop 0 "frames=75/75 .*verdict=PASS"
  --mode replay --recording "${_out}/ci" --stream mono --pacing fast --loop true --count 75
  --memory host --display false
)
set_tests_properties(
  recorder_app_replay_realtime
  recorder_app_replay_fast_loop
  PROPERTIES DEPENDS recorder_app_ci FIXTURES_REQUIRED recorder_ci
)
set_tests_properties(recorder_app_ci PROPERTIES FIXTURES_SETUP recorder_ci)
# A live camera recorded through kDropOldest edges: whatever the disk takes, every frame
# is recorded or counted as dropped.
recorder_app_test(recorder_app_bench 0
  "bench 640x480 mono8 device at 100.0 fps: frames=100/100 .*verdict=PASS"
  --mode bench --width 640 --height 480 --encoding mono8 --fps 100 --frames 100
  --recording "${_out}/bench"
)
# A real stereo camera recording, when HOLOSCAN_TEST_DATA has one: every RGB frame,
# fast, and the depth images in real time.
set(_clip "$ENV{HOLOSCAN_TEST_DATA}/stereo-camera/d585_480p_emitter_on")
function(recorder_clip_test name output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=${output}"
      "-DSKIP_UNLESS_EXISTS=${_clip}/index.csv" -P "${_run_and_match}" -- $<TARGET_FILE:recorder>
      --mode replay --recording "${_clip}" --display false ${ARGN}
  )
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()
recorder_clip_test(recorder_app_clip_fast "replayed rgb .*frames=60/60 .*verdict=PASS" --pacing fast)
recorder_clip_test(recorder_app_clip_realtime "replayed depth .*frames=20/20 .*verdict=PASS"
  --stream depth --count 20
)

# Invalid input fails fast with a usage error (exit status 2).
recorder_app_test(recorder_app_help 0 "^Usage: recorder" --help)
recorder_app_test(recorder_app_rejects_mode 2 "^error: --mode takes" --mode stream)
recorder_app_test(recorder_app_rejects_option 2 "^error: unknown option" --colour 1)
recorder_app_test(recorder_app_rejects_replay_without_recording 2 "^error: --mode replay needs"
  --mode replay
)
recorder_app_test(recorder_app_rejects_missing_recording 2 "^error: .*no such folder"
  --mode replay --recording "${_out}/nowhere" --display false
)
recorder_app_test(recorder_app_rejects_odd_width 2 "^error: --width and --height" --width 33)
recorder_app_test(recorder_app_rejects_loop_without_count 2 "needs --count"
  --mode replay --recording "${_out}/ci" --loop true --display false
)
set_tests_properties(
  recorder_app_rejects_loop_without_count
  PROPERTIES FIXTURES_REQUIRED recorder_ci
)
# The graphs compile, the display's too, with or without a display to run it.
recorder_app_test(recorder_app_compile_only 0 "^recorder: plan=OK\n"
  --compile-only true --recording "${_out}/compile_only"
)
if(RECORDER_ENABLE_VISUALIZATION)
  recorder_app_test(recorder_app_display_plan 0 "^recorder: plan=OK\n*$"
    --mode replay --recording "${_out}/ci" --display true --compile-only true
  )
  set_tests_properties(recorder_app_display_plan PROPERTIES FIXTURES_REQUIRED recorder_ci)
endif()

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "recorder;gpu" TIMEOUT 300)
