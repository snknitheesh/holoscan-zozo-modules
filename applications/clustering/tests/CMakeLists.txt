# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU.

# Runs the demo through run_and_match.cmake, which checks its exit status as well as its
# output: clustering_app_test(<name> <exit status> <output regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(clustering_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:clustering> ${ARGN}
  )
endfunction()

# Every frame checked against the scene's truth, in each connectivity, at another size.
set(_pass "wrong_counts=0 .*wrong_objects=0 inconsistent=0.*verdict=PASS")
clustering_app_test(clustering_app_euclidean 0 "${_pass}" --count 10 --period-ms 10 --display false)
clustering_app_test(clustering_app_image 0 "${_pass}"
  --count 10 --period-ms 10 --display false --connectivity image
)
clustering_app_test(clustering_app_720p 0 "${_pass}"
  --count 5 --period-ms 20 --width 1280 --height 720 --display false
)
# The display graph compiles, without a display server.
clustering_app_test(clustering_app_display_graph 0 "plan=OK" --display true --compile-only true)

# Unusable arguments are usage errors.
clustering_app_test(clustering_app_rejects_size 2 "^error: scene images" --width 10)
clustering_app_test(clustering_app_rejects_tolerance 2 "^error: tolerance" --tolerance 0)
clustering_app_test(clustering_app_rejects_connectivity 2 "^error: --connectivity"
  --connectivity grid
)
clustering_app_test(clustering_app_rejects_option 2 "^error: unknown option" --frames 3)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "clustering_app;gpu" TIMEOUT 300)
