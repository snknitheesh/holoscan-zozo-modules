# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Runs a command and passes only if it exits with EXPECT_EXIT (default 0) and
# its combined output matches EXPECT_OUTPUT. CTest's own PASS_REGULAR_EXPRESSION
# would ignore the exit status.
#
#   cmake -DEXPECT_EXIT=2 -DEXPECT_OUTPUT=regex [-DREQUIRE_ENV=variable]
#         [-DSKIP_UNLESS_EXISTS=path1|path2] [-DCLIP_SUFFIX=suffix]
#         -P run_and_match.cmake -- command [args...]
#
# When REQUIRE_ENV names an unset or empty variable, or a path in
# SKIP_UNLESS_EXISTS is missing, the script runs nothing and exits with status
# 77, which the test's SKIP_RETURN_CODE reports as skipped: for data prepared
# separately. The paths are separated by |, as a ; would split the test's
# command. With CLIP_SUFFIX=<suffix>, the command gets `--clip <name>` for the
# one recording in $HOLOSCAN_TEST_DATA/stereo-camera/ whose folder name ends in
# it, so that the tests need not name the recordings' folders.

if(DEFINED REQUIRE_ENV AND "$ENV{${REQUIRE_ENV}}" STREQUAL "")
  message("skipped: ${REQUIRE_ENV} is not set")
  cmake_language(EXIT 77)
endif()
if(DEFINED SKIP_UNLESS_EXISTS)
  string(REPLACE "|" ";" _paths "${SKIP_UNLESS_EXISTS}")
  foreach(_path IN LISTS _paths)
    if(NOT EXISTS "${_path}")
      message("skipped: ${_path} is missing; see the README to prepare it")
      cmake_language(EXIT 77)
    endif()
  endforeach()
endif()
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
