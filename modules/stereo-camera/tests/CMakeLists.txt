# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. The operator tests need the SDK, CUDA and a GPU; they write their
# synthetic recordings at run time, so they need no data. The real-data test plays the
# recordings in $HOLOSCAN_TEST_DATA/stereo-camera/ and is skipped (exit 77) without them.
set(_work "${CMAKE_CURRENT_BINARY_DIR}/work")
file(MAKE_DIRECTORY "${_work}")

add_executable(stereo_camera_op_test stereo_camera_op_test.cpp)
target_link_libraries(stereo_camera_op_test PRIVATE holoscan::stereo_camera)
add_test(NAME stereo_camera_op COMMAND stereo_camera_op_test "${_work}")

add_executable(stereo_camera_real_data_op_test real_data_op_test.cpp)
target_link_libraries(stereo_camera_real_data_op_test PRIVATE holoscan::stereo_camera)
add_test(NAME stereo_camera_real_data_op COMMAND stereo_camera_real_data_op_test)

set_tests_properties(
  stereo_camera_op
  stereo_camera_real_data_op
  PROPERTIES LABELS "stereo_camera;gpu" TIMEOUT 600
)
set_tests_properties(
  stereo_camera_real_data_op
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "stereo_camera;gpu;data"
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
