# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. Everything but the host tests needs the SDK, CUDA and a GPU.

add_executable(rectify_remap_backends_test remap_backends_test.cpp)
target_link_libraries(rectify_remap_backends_test PRIVATE holoscan::rectify)

add_executable(rectify_operator_test operator_test.cpp)
target_link_libraries(rectify_operator_test PRIVATE holoscan::rectify holoscan::testing)
target_compile_definitions(
  rectify_operator_test
  PRIVATE HOLOSCAN_RECTIFY_TEST_WITH_VPI=$<BOOL:${HOLOSCAN_RECTIFY_VPI}>
)

# Needs $HOLOSCAN_TEST_DATA (the stereo camera recordings); exits 77, a skip, without it.
add_executable(rectify_real_data_test real_data_test.cpp)
target_link_libraries(rectify_real_data_test PRIVATE holoscan::rectify holoscan::testing)

add_test(NAME rectify_remap_backends COMMAND rectify_remap_backends_test)
add_test(NAME rectify_operators COMMAND rectify_operator_test)
add_test(NAME rectify_real_data COMMAND rectify_real_data_test)
set_tests_properties(rectify_real_data PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(
  rectify_remap_backends
  rectify_operators
  rectify_real_data
  PROPERTIES LABELS "rectify;gpu" TIMEOUT 600
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
