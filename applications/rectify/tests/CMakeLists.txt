# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

add_library(rectify_spawn_thread_preload SHARED spawn_thread_preload.cpp)

# Runs the demo through run_and_match.cmake, which checks its exit status as well as
# its output: rectify_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(rectify_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:rectify> ${ARGN}
  )
endfunction()

# The synthetic distorted pair through every backend this machine can run and auto,
# from host and from GPU memory, and with a fisheye; each run checks every dot's row in
# both rectified images and, on a CI runner, that auto picked the platform's backend.
rectify_app_test(rectify_app_synthetic 0 "backend=auto:.*verdict=PASS"
  --backend all,auto --count 20 --period-ms 10 --display false
)
rectify_app_test(rectify_app_device_input 0 "backend=auto:.*verdict=PASS"
  --backend all,auto --count 10 --period-ms 10 --device-input true --display false
)
rectify_app_test(rectify_app_fisheye 0 "verdict=PASS"
  --backend all --distortion kannala_brandt4 --width 640 --height 480 --count 10 --period-ms 10
  --display false
)

# Invalid input fails fast with a usage error (exit status 2) instead of starting a run.
rectify_app_test(rectify_app_rejects_unknown_backend 2 "^error: " --backend gpu --display false)
rectify_app_test(rectify_app_rejects_zero_count 2 "^error: " --count 0 --display false)
rectify_app_test(rectify_app_rejects_repeated_backend 2 "error: .*listed twice"
  --backend all,cuda --display false
)
rectify_app_test(rectify_app_rejects_alpha 2 "^error: .*alpha" --alpha 2 --display false)

# A recording, rectified by its camera already, comes out unchanged. Needs
# $HOLOSCAN_TEST_DATA (the stereo camera recordings); skipped (77) without it.
add_test(
  NAME rectify_app_recording
  COMMAND
    sh -c
    "if [ -z \"$HOLOSCAN_TEST_DATA\" ]; then echo 'HOLOSCAN_TEST_DATA is not set; skipping'; exit 77; fi; exec \"$0\" -DEXPECT_EXIT=0 '-DEXPECT_OUTPUT=max_pixel_change=[01].*verdict=PASS' -P \"$1\" -- \"$2\" --recording \"$HOLOSCAN_TEST_DATA/stereo-camera/d585_480p_emitter_on\" --backend all,auto --count 10 --period-ms 10 --display false"
    ${CMAKE_COMMAND} "${_run_and_match}" $<TARGET_FILE:rectify>
)
set_tests_properties(rectify_app_recording PROPERTIES SKIP_RETURN_CODE 77)

# A stop request ends the run early with its own verdict. (A shell starts background
# jobs with SIGINT ignored, so the test sends SIGTERM.)
add_test(
  NAME rectify_app_stopped
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=1 "-DEXPECT_OUTPUT=frames=[0-9]+/1000 .*verdict=STOPPED" -P
    "${_run_and_match}" -- sh -c
    "\"$1\" --backend cuda --width 640 --height 480 --count 1000 --period-ms 10 --display false & sleep 4 && kill -TERM $! && wait $!"
    sh $<TARGET_FILE:rectify>
)

# Another thread running before main(), as under a profiler or a sanitizer: the run
# goes ahead without managed signal handling, and says so.
add_test(
  NAME rectify_app_without_signal_domain
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS" -P "${_run_and_match}"
    -- ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:rectify_spawn_thread_preload>
    $<TARGET_FILE:rectify> --backend cuda --width 640 --height 480 --count 5 --period-ms 10
    --display false
)

if(RECTIFY_ENABLE_VISUALIZATION)
  # The display graph compiles, with or without a display to run it.
  rectify_app_test(rectify_app_display_plan 0 "plan=OK"
    --backend all --display true --compile-only true
  )
endif()

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "rectify;gpu" TIMEOUT 600)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
