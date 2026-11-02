# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The demo end to end, through run_and_match.cmake, which checks its exit status and its output:
# codec_app_test(<name> <exit status> <output regex> <arguments>...). Each needs a GPU with a
# hardware encoder and decoder; without one the demo exits 77 and the test is skipped. The
# module's own suite covers the operators.
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
# Rendered offscreen by HoloViz: the original and the decoded frame side by side, both checked.
codec_app_test(codec_app_offscreen 0 "frames=30/30 .*rendered=[1-9].*verdict=PASS"
  --display offscreen --count 30 --period-ms 20 --rate-control cqp --qp 25 --min-psnr 34
  --render-file "${CMAKE_CURRENT_BINARY_DIR}/codec_render.ppm"
)
# Unusable options are refused before anything runs.
codec_app_test(codec_app_bad_qp 2 "qp must be 0 to 51" --rate-control cqp --qp 60)
codec_app_test(codec_app_bad_display 2 "--render-file needs --display offscreen"
  --display none --render-file x.ppm
)
