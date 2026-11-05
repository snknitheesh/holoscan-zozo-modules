# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. The operator test needs the SDK, CUDA, VPI and a GPU; on a
# Jetson (HOLOSCAN_CI_PLATFORM=jetson) it also requires the ofa backend.

add_executable(stereo_disparity_op_test stereo_disparity_op_test.cpp)
target_link_libraries(stereo_disparity_op_test PRIVATE holoscan::stereo_disparity holoscan::testing)

add_test(NAME stereo_disparity_operator COMMAND stereo_disparity_op_test)
set_tests_properties(stereo_disparity_operator PROPERTIES LABELS "stereo_disparity;gpu" TIMEOUT 600)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
