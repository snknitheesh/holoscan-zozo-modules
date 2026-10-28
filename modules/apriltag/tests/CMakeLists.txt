# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Everything but the axes test needs the SDK, CUDA and a GPU.

add_executable(apriltag_backend_accuracy_test backend_accuracy_test.cpp)
target_include_directories(apriltag_backend_accuracy_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(apriltag_backend_accuracy_test PRIVATE holoscan::apriltag)

add_executable(apriltag_operator_test operator_test.cpp)
target_link_libraries(apriltag_operator_test PRIVATE holoscan::apriltag holoscan::testing)
target_compile_definitions(
  apriltag_operator_test
  PRIVATE HOLOSCAN_APRILTAG_WITH_OVERLAY=$<BOOL:${HOLOSCAN_APRILTAG_BUILD_OVERLAY}>
)
if(HOLOSCAN_APRILTAG_BUILD_OVERLAY)
  target_link_libraries(
    apriltag_operator_test
    PRIVATE holoscan::apriltag_overlay holoscan::visualization::op_holoviz
  )
endif()

add_test(NAME apriltag_backend_accuracy COMMAND apriltag_backend_accuracy_test)
add_test(NAME apriltag_operators COMMAND apriltag_operator_test)
set_tests_properties(
  apriltag_backend_accuracy
  apriltag_operators
  PROPERTIES LABELS "apriltag;gpu" TIMEOUT 600
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
