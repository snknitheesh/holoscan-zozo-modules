# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The demo end to end, through run_and_match.cmake, which checks its exit status and its output:
# codec_app_test(<name> <exit status> <output regex> <arguments>...). Each needs a GPU with a
# hardware encoder and decoder; without one the demo exits 77 and the test is skipped, except in
# CI (HOLOSCAN_CI_PLATFORM set), where that fails. The module's own suite covers the operators.
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(codec_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:codec_demo> ${ARGN}
  )
  set_tests_properties(
    ${name}
    PROPERTIES LABELS "codec;gpu" TIMEOUT 300 SKIP_RETURN_CODE 77
  )
endfunction()

# Every frame decoded, in order, at the PSNR its setting reaches.
codec_app_test(codec_app_h265_cqp 0 "frames=40/40 .*verdict=PASS"
  --display none --codec h265 --rate-control cqp --qp 25 --count 40 --period-ms 10 --min-psnr 34
)
codec_app_test(codec_app_h264_cbr_odd 0 "frames=40/40 .*verdict=PASS"
  --display none --codec h264 --rate-control cbr --bitrate 4000000 --width 641 --height 361
  --count 40 --period-ms 10 --min-psnr 28
)
# Without low latency, the decoder holds frames back; the stream's end hands them out.
codec_app_test(codec_app_reordering_decoder 0 "frames=40/40 .*verdict=PASS"
  --display none --low-latency false --rate-control cqp --qp 25 --count 40 --period-ms 10
  --min-psnr 34
)
# Rendered offscreen by HoloViz: the original and the decoded frame side by side, both checked;
# also with a source faster than the chain.
codec_app_test(codec_app_offscreen 0 "workers=1 frames=30/30 .*rendered=[1-9].*verdict=PASS"
  --display offscreen --count 30 --period-ms 20 --rate-control cqp --qp 25 --min-psnr 34
  --render-file "${CMAKE_CURRENT_BINARY_DIR}/codec_render.ppm"
)
codec_app_test(codec_app_offscreen_fast 0 "workers=1 frames=120/120 .*verdict=PASS"
  --display offscreen --count 120 --period-ms 5
)
# The live graph's edges and automatic partitioning, headless: the source's pacing keeps every
# frame, with a source faster than the chain.
codec_app_test(codec_app_live_edges 0 "frames=120/120 .*verdict=PASS"
  --display none --workers auto --count 120 --period-ms 5
)
# The lossless graphs run on one worker; the live window graph compiles with the automatic
# partitioning, with or without a display to run it.
codec_app_test(codec_app_one_worker 0 "^codec_demo: plan=OK workers=1\n*$"
  --display none --compile-only true
)
codec_app_test(codec_app_offscreen_one_worker 0 "^codec_demo: plan=OK workers=1\n*$"
  --display offscreen --compile-only true
)
codec_app_test(codec_app_window_plan 0 "^codec_demo: plan=OK workers=[1-9][0-9]*\n*$"
  --display window --compile-only true
)
# Unusable options are refused before anything runs.
codec_app_test(codec_app_bad_qp 2 "qp must be 0 to 51" --rate-control cqp --qp 60)
codec_app_test(codec_app_bad_display 2 "--render-file needs --display offscreen"
  --display none --render-file x.ppm
)
codec_app_test(codec_app_bad_workers 2 "--workers must be one or auto" --workers two)
