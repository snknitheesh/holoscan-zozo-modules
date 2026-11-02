# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: isp_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(isp_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:isp> ${ARGN}
  )
endfunction()

set(
  _fast
  --count
  10
  --period-ms
  10
  --width
  640
  --height
  480
  --display
  false
)
# Every frame checked against the camera's truth, with each demosaic method, bit depth,
# packing, pattern, sharpening and backend.
isp_app_test(isp_app_malvar 0 "demosaic=malvar.*verdict=PASS" ${_fast})
isp_app_test(isp_app_bilinear 0 "demosaic=bilinear.*verdict=PASS" ${_fast} --demosaic bilinear)
isp_app_test(isp_app_8bit 0 "verdict=PASS" ${_fast} --bits 8 --pattern BGGR)
isp_app_test(isp_app_16bit_sharpened 0 "verdict=PASS"
  ${_fast} --bits 16 --pattern GRBG --sharpen 0.3 --min-psnr 26
)
isp_app_test(isp_app_raw10 0 "verdict=PASS" ${_fast} --packing raw10 --pattern GBRG)
isp_app_test(isp_app_raw12 0 "verdict=PASS" ${_fast} --packing raw12)
isp_app_test(isp_app_1080p 0 "verdict=PASS" --count 5 --period-ms 20 --width 1920 --height 1080
  --display false
)
isp_app_test(isp_app_cpu 0 "backend=cpu.*verdict=PASS"
  --count 3 --period-ms 10 --width 320 --height 240 --backend cpu --display false
)
# Auto white balance finds the camera's gains: gray world on grays, white patch on the chart.
isp_app_test(isp_app_gray_world 0 "wb_gains=1.9[0-9]*,1.000,1.[56].*verdict=PASS"
  ${_fast} --scene gray --white-balance gray-world
)
isp_app_test(isp_app_white_patch 0 "verdict=PASS" ${_fast} --white-balance white-patch)
# HoloViz renders the raw and processed views offscreen, into CUDA memory.
isp_app_test(isp_app_offscreen 0 "showing=[1-9].*verdict=PASS"
  --count 10 --period-ms 30 --width 640 --height 480 --display offscreen
)
# The display graph compiles, without a display server.
isp_app_test(isp_app_display_graph 0 "plan=OK" --display true --compile-only true)

# Unusable arguments are usage errors.
isp_app_test(isp_app_rejects_size 2 "^error: synthetic frames" --width 8)
isp_app_test(isp_app_rejects_pattern 2 "^error: --pattern" --pattern RGBG)
isp_app_test(isp_app_rejects_bits 2 "^error: synthetic bit depth" --bits 17)
isp_app_test(isp_app_rejects_sharpen 2 "^error: sharpen" --sharpen 9)
isp_app_test(isp_app_rejects_option 2 "^error: unknown option" --frames 3)
isp_app_test(isp_app_rejects_cpu_display 2 "^error: the display needs the cuda backend"
  --backend cpu --display true --compile-only true
)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "isp_app;gpu" TIMEOUT 300)
