# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite: the GPU tests here, and the SDK-free tests in host/.

add_executable(fiducials_backend_test backend_test.cpp)
target_link_libraries(fiducials_backend_test PRIVATE holoscan::fiducials)

add_executable(fiducials_operator_test operator_test.cpp)
target_link_libraries(fiducials_operator_test PRIVATE holoscan::fiducials holoscan::testing)
target_compile_definitions(
  fiducials_operator_test
  PRIVATE HOLOSCAN_FIDUCIALS_WITH_OVERLAY=$<BOOL:${HOLOSCAN_FIDUCIALS_BUILD_OVERLAY}>
)
if(HOLOSCAN_FIDUCIALS_BUILD_OVERLAY)
  target_link_libraries(
    fiducials_operator_test
    PRIVATE holoscan::fiducials_overlay holoscan::visualization::op_holoviz
  )
endif()

add_test(NAME fiducials_backends COMMAND fiducials_backend_test)
add_test(NAME fiducials_operators COMMAND fiducials_operator_test)
set_tests_properties(
  fiducials_backends
  fiducials_operators
  PROPERTIES LABELS "fiducials;gpu" TIMEOUT 900
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
