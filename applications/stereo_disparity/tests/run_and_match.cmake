# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Runs a command and passes only if it exits with EXPECT_EXIT (default 0) and
# its combined output matches EXPECT_OUTPUT. CTest's own PASS_REGULAR_EXPRESSION
# would ignore the exit status.
#
#   cmake -DEXPECT_EXIT=2 -DEXPECT_OUTPUT=regex [-DMAKE_DIRECTORY=dir]
#         -P run_and_match.cmake -- command [args...]
#
# MAKE_DIRECTORY is created first, for a test that needs one in its way. With
# REQUIRE_ENV=<variable>, an unset or empty variable runs nothing and exits with
# status 77, which the test's SKIP_RETURN_CODE reports as skipped. With
# CLIP_SUFFIX=<suffix>, the command gets `--clip <name>` for the one recording
# in $HOLOSCAN_TEST_DATA/stereo-camera/ whose folder name ends in it. With
# BACKENDS_PROBE=<program>, which prints the platform's backends one per line,
# and REQUIRE_BACKEND=<backend> or WITHOUT_BACKEND=<backend>, the script exits
# with status 77 where the platform lacks, or has, that backend: the platform
# is found when the test runs, not when it is configured.

if(NOT DEFINED EXPECT_EXIT)
  set(EXPECT_EXIT 0)
endif()
if(NOT DEFINED EXPECT_OUTPUT)
  message(FATAL_ERROR "run_and_match: EXPECT_OUTPUT is required")
endif()

set(_command)
set(_after_separator FALSE)
math(EXPR _last "${CMAKE_ARGC} - 1")
foreach(_index RANGE ${_last})
  if(_after_separator)
    list(APPEND _command "${CMAKE_ARGV${_index}}")
  elseif(CMAKE_ARGV${_index} STREQUAL "--")
    set(_after_separator TRUE)
  endif()
endforeach()
if(NOT _command)
  message(FATAL_ERROR "run_and_match: no command after --")
endif()

if(DEFINED REQUIRE_ENV AND "$ENV{${REQUIRE_ENV}}" STREQUAL "")
  message("skipped: ${REQUIRE_ENV} is not set")
  cmake_language(EXIT 77)
endif()

if(DEFINED BACKENDS_PROBE)
  execute_process(
    COMMAND "${BACKENDS_PROBE}"
    RESULT_VARIABLE _probe_status
    OUTPUT_VARIABLE _backends
    ERROR_VARIABLE _probe_errors
  )
  if(NOT _probe_status STREQUAL "0")
    message(
      FATAL_ERROR
      "run_and_match: ${BACKENDS_PROBE} failed (${_probe_status}): ${_probe_errors}"
    )
  endif()
  string(STRIP "${_backends}" _backends)
  string(REPLACE "\n" ";" _backends "${_backends}")
  if(DEFINED REQUIRE_BACKEND AND NOT REQUIRE_BACKEND IN_LIST _backends)
    message("skipped: this platform has no ${REQUIRE_BACKEND} backend")
    cmake_language(EXIT 77)
  endif()
  if(DEFINED WITHOUT_BACKEND AND WITHOUT_BACKEND IN_LIST _backends)
    message("skipped: this platform has the ${WITHOUT_BACKEND} backend")
    cmake_language(EXIT 77)
  endif()
endif()

if(DEFINED CLIP_SUFFIX)
  file(GLOB _clips LIST_DIRECTORIES true "$ENV{HOLOSCAN_TEST_DATA}/stereo-camera/*${CLIP_SUFFIX}")
  list(LENGTH _clips _clip_count)
  if(NOT _clip_count EQUAL 1)
    message(
      FATAL_ERROR
      "run_and_match: ${_clip_count} recordings in $ENV{HOLOSCAN_TEST_DATA}/stereo-camera/ end in"
      " ${CLIP_SUFFIX}; expected one"
    )
  endif()
  get_filename_component(_clip "${_clips}" NAME)
  list(APPEND _command --clip "${_clip}")
endif()

if(DEFINED MAKE_DIRECTORY)
  file(MAKE_DIRECTORY "${MAKE_DIRECTORY}")
endif()

execute_process(
  COMMAND ${_command}
  RESULT_VARIABLE _status
  OUTPUT_VARIABLE _output
  ERROR_VARIABLE _output
)
message("${_output}")
if(NOT _status STREQUAL "${EXPECT_EXIT}")
  message(FATAL_ERROR "exit status ${_status}, expected ${EXPECT_EXIT}")
endif()
if(NOT _output MATCHES "${EXPECT_OUTPUT}")
  message(FATAL_ERROR "output does not match: ${EXPECT_OUTPUT}")
endif()
