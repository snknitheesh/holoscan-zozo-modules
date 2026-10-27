# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. The kernel and operator tests need the SDK, CUDA and a GPU;
# none needs a download: the stub model stands in for YOLO26.

add_executable(yolo26_letterbox_test letterbox_test.cu)
add_executable(yolo26_decode_test decode_test.cu)
foreach(_test yolo26_letterbox_test yolo26_decode_test)
  target_link_libraries(${_test} PRIVATE holoscan::yolo26)
  target_compile_features(${_test} PRIVATE cuda_std_20)
  # The tests' host evaluations of the kernels' expressions must not contract either.
  target_compile_options(${_test} PRIVATE -fmad=false -Xcompiler=-ffp-contract=off)
endforeach()

add_executable(yolo26_operator_test operator_test.cpp)
target_link_libraries(yolo26_operator_test PRIVATE holoscan::yolo26 holoscan::testing)
target_compile_definitions(yolo26_operator_test PRIVATE
  HOLOSCAN_YOLO26_WITH_OVERLAY=$<BOOL:${HOLOSCAN_YOLO26_BUILD_OVERLAY}>)
if(HOLOSCAN_YOLO26_BUILD_OVERLAY)
  target_link_libraries(yolo26_operator_test PRIVATE
    holoscan::yolo26_overlay holoscan::visualization::op_holoviz)
endif()

add_test(NAME yolo26_letterbox COMMAND yolo26_letterbox_test)
add_test(NAME yolo26_decode COMMAND yolo26_decode_test)
# The stub models and their TensorRT engines go to a folder of their own.
add_test(NAME yolo26_operators
  COMMAND yolo26_operator_test "${CMAKE_CURRENT_BINARY_DIR}/operator_models")
set_tests_properties(yolo26_letterbox yolo26_decode yolo26_operators PROPERTIES
  LABELS "yolo26;gpu" TIMEOUT 600)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
