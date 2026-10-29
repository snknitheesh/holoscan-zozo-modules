# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Every test needs the SDK, CUDA and a GPU.

# The registration and odometry on synthetic scenes, and the CPU reference.
add_executable(icp_registration_test registration_test.cpp cpu_reference.cpp)
target_include_directories(icp_registration_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(icp_registration_test PRIVATE holoscan::icp)

add_test(NAME icp_registration COMMAND icp_registration_test)
set_tests_properties(icp_registration PROPERTIES LABELS "icp;gpu" TIMEOUT 600)

# IcpOp through the SDK's operator test harness.
add_executable(icp_op_test icp_op_test.cpp)
target_link_libraries(icp_op_test PRIVATE holoscan::icp holoscan::testing)

add_test(NAME icp_op COMMAND icp_op_test)
set_tests_properties(icp_op PROPERTIES LABELS "icp;gpu" TIMEOUT 600)

# Odometry's time per frame at three resolutions, which the run prints; it fails only if
# frames fail to register.
add_executable(icp_benchmark icp_benchmark.cpp)
target_link_libraries(icp_benchmark PRIVATE holoscan::icp)

add_test(NAME icp_benchmark COMMAND icp_benchmark)
set_tests_properties(icp_benchmark PROPERTIES LABELS "icp;gpu" TIMEOUT 600)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
