# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: messages_app_test(<name> <exit status> <output regex> <arguments>...). The runs
# need a GPU and its driver.
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(messages_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:messages_demo> ${ARGN}
  )
endfunction()

set(_pass "frames=40/40 .* wrong_camera_info=0 .* clouds=40/40 wrong_clouds=0.*verdict=PASS")
messages_app_test(messages_app_device_clouds 0 "${_pass}" --count 40 --period-ms 5)
messages_app_test(messages_app_host_clouds 0 "${_pass}"
  --count 40 --period-ms 5 --cloud-memory host
)
# Camera info slower than the frames: every frame uses the newest one.
messages_app_test(messages_app_slow_camera_info 0 "${_pass}"
  --count 40 --period-ms 5 --info-period-ms 500
)
# Only the plan's verdict: an accepted plan has no warnings. Three sources, and still one
# worker, as the demo lays the graph out; the automatic partitioning would split it onto four.
messages_app_test(messages_app_one_worker 0 "^messages: plan=OK workers=1\n*$" --compile-only true)
messages_app_test(messages_app_bad_option 2 "error: --cloud-memory takes device or host"
  --cloud-memory shared
)
messages_app_test(messages_app_bad_period 2 "error: --period-ms takes a number 1..1000"
  --period-ms 0
)
messages_app_test(messages_app_bad_compile_only 2 "error: --compile-only takes true or false"
  --compile-only yes
)
set_tests_properties(
  messages_app_device_clouds
  messages_app_host_clouds
  messages_app_slow_camera_info
  messages_app_one_worker
  messages_app_bad_option
  messages_app_bad_period
  messages_app_bad_compile_only
  PROPERTIES LABELS "messages;gpu" TIMEOUT 300
)
