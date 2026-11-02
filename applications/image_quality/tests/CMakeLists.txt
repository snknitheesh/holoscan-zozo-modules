# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU; the offscreen one also HoloViz's Vulkan driver.

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: image_quality_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(image_quality_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:image_quality> ${ARGN}
  )
endfunction()

# Two cycles of the camera's segments, every frame's flags checked, from host and from GPU
# memory.
set(
  _every_segment
  "segment=sharp .*segment=blurred .*segment=dark .*segment=noisy .*segment=frozen .*segment=smudged .*segment=tinted .*wrong=0.*verdict=PASS"
)
image_quality_app_test(image_quality_app_cycles 0 "${_every_segment}"
  --count 280 --segment-frames 20 --obstruction-frames 10 --period-ms 5 --display false
)
image_quality_app_test(image_quality_app_device_input 0 "${_every_segment}"
  --count 140 --segment-frames 20 --obstruction-frames 10 --period-ms 5 --device-input true
  --display false
)
# A larger image, in fewer frames.
image_quality_app_test(image_quality_app_hd 0 "${_every_segment}"
  --count 105 --segment-frames 15 --obstruction-frames 5 --period-ms 5 --width 1280
  --height 720 --device-input true --display false
)

# HoloViz renders the image, the tile overlay and the metrics into GPU memory, where the
# checker finds each, and saves the frame with the most overlay.
image_quality_app_test(image_quality_app_offscreen 0
  "rendered=[1-9][0-9]* .*with_overlay=[1-9][0-9]*\n.*snapshot=.*verdict=PASS"
  --count 140 --segment-frames 20 --obstruction-frames 10 --period-ms 10 --display offscreen
  --snapshot "${CMAKE_CURRENT_BINARY_DIR}/image_quality_offscreen.ppm"
)
# The window graph compiles, with or without a display to run it.
image_quality_app_test(image_quality_app_window_plan 0 "plan=OK"
  --display true --compile-only true
)

# Invalid input fails fast with a usage error (exit status 2) instead of starting a run.
image_quality_app_test(image_quality_app_rejects_unknown_option 2 "^error: unknown option"
  --color true
)
image_quality_app_test(image_quality_app_rejects_zero_count 2 "^error: " --count 0)
image_quality_app_test(image_quality_app_rejects_short_segments 2 "^error: .*less than"
  --segment-frames 10 --obstruction-frames 10
)
image_quality_app_test(image_quality_app_rejects_snapshot 2 "^error: --snapshot"
  --snapshot x.ppm --display false
)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "image_quality;gpu" TIMEOUT 600)
