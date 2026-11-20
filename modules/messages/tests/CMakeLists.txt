# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. The graph tests need the SDK, CUDA and a GPU; the SDK-free
# tests are in host/.

foreach(_test message_graph contract camera_info_source_op)
  add_executable(messages_${_test}_test ${_test}_test.cpp)
  target_link_libraries(messages_${_test}_test PRIVATE holoscan::messages)
  add_test(NAME messages_${_test} COMMAND messages_${_test}_test)
  set_tests_properties(messages_${_test} PROPERTIES LABELS "messages;gpu" TIMEOUT 600)
endforeach()

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
