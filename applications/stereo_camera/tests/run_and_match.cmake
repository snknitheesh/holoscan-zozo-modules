# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Runs a command and passes only if it exits with EXPECT_EXIT (default 0) and
# its combined output matches EXPECT_OUTPUT. CTest's own PASS_REGULAR_EXPRESSION
# would ignore the exit status.
#
#   cmake -DEXPECT_EXIT=2 -DEXPECT_OUTPUT=regex [-DMAKE_DIRECTORY=dir]
#         [-DTEST_DATA=stereo-camera/clip] -P run_and_match.cmake -- command [args...]
#
# MAKE_DIRECTORY is created first, for a test that needs one in its way. With
# TEST_DATA, the test needs that path under $HOLOSCAN_TEST_DATA, which @DATA@ in
# the command stands for; without it, or the variable, the script exits with
# status 77, which the test's SKIP_RETURN_CODE reports as skipped.

set(_data "")
if(DEFINED TEST_DATA)
  if("$ENV{HOLOSCAN_TEST_DATA}" STREQUAL "")
    message("skipped: HOLOSCAN_TEST_DATA is not set")
    cmake_language(EXIT 77)
  endif()
  set(_data "$ENV{HOLOSCAN_TEST_DATA}")
  if(NOT EXISTS "${_data}/${TEST_DATA}")
    message("skipped: ${_data}/${TEST_DATA} is missing")
    cmake_language(EXIT 77)
  endif()
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
    string(REPLACE "@DATA@" "${_data}" _argument "${CMAKE_ARGV${_index}}")
    list(APPEND _command "${_argument}")
  elseif(CMAKE_ARGV${_index} STREQUAL "--")
    set(_after_separator TRUE)
  endif()
endforeach()
if(NOT _command)
  message(FATAL_ERROR "run_and_match: no command after --")
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
