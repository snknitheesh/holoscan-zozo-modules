# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. The operator tests need the SDK, CUDA and a GPU, and write their
# recordings under this build folder. The real-data test is skipped (exit 77) unless
# HOLOSCAN_TEST_DATA names a folder with the stereo camera recordings in stereo-camera/.
foreach(_name IN ITEMS round_trip recorder_op replayer_op real_data)
  add_executable(recorder_${_name}_test ${_name}_test.cpp)
  target_link_libraries(recorder_${_name}_test PRIVATE holoscan::recorder CUDA::cudart)
  add_test(NAME recorder_${_name} COMMAND recorder_${_name}_test)
  set_tests_properties(recorder_${_name} PROPERTIES LABELS "recorder;gpu" TIMEOUT 600)
endforeach()
# Real-time pacing measures time: it runs alone.
set_tests_properties(recorder_replayer_op PROPERTIES RUN_SERIAL TRUE)
set_tests_properties(
  recorder_real_data
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "recorder;gpu;data" TIMEOUT 900
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
