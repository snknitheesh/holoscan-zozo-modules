# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. The kernel and operator tests need the SDK, CUDA and a GPU;
# none needs a download: the stub models stand in for YOLO26 and YOLO26-seg.

add_executable(yolo26_letterbox_test letterbox_test.cu)
add_executable(yolo26_decode_test decode_test.cu)
add_executable(yolo26_masks_test masks_test.cu)
foreach(_test yolo26_letterbox_test yolo26_decode_test yolo26_masks_test)
  target_link_libraries(${_test} PRIVATE holoscan::yolo26)
  target_compile_features(${_test} PRIVATE cuda_std_20)
  # The tests' host evaluations of the kernels' expressions must not contract either.
  target_compile_options(${_test} PRIVATE -fmad=false -Xcompiler=-ffp-contract=off)
endforeach()

add_executable(yolo26_operator_test operator_test.cpp)
target_link_libraries(yolo26_operator_test PRIVATE holoscan::yolo26 holoscan::testing)
target_compile_definitions(
  yolo26_operator_test
  PRIVATE HOLOSCAN_YOLO26_WITH_OVERLAY=$<BOOL:${HOLOSCAN_YOLO26_BUILD_OVERLAY}>
)
if(HOLOSCAN_YOLO26_BUILD_OVERLAY)
  target_link_libraries(
    yolo26_operator_test
    PRIVATE holoscan::yolo26_overlay holoscan::visualization::op_holoviz
  )
endif()

add_executable(yolo26_segmentation_op_test segmentation_op_test.cpp)
target_link_libraries(yolo26_segmentation_op_test PRIVATE holoscan::yolo26 holoscan::testing)
target_compile_definitions(
  yolo26_segmentation_op_test
  PRIVATE HOLOSCAN_YOLO26_WITH_OVERLAY=$<BOOL:${HOLOSCAN_YOLO26_BUILD_OVERLAY}>
)
if(HOLOSCAN_YOLO26_BUILD_OVERLAY)
  target_link_libraries(
    yolo26_segmentation_op_test
    PRIVATE holoscan::yolo26_overlay holoscan::visualization::op_holoviz
  )
endif()

add_test(NAME yolo26_letterbox COMMAND yolo26_letterbox_test)
add_test(NAME yolo26_decode COMMAND yolo26_decode_test)
add_test(NAME yolo26_masks COMMAND yolo26_masks_test)
# The stub models and their TensorRT engines go to a folder of their own.
add_test(
  NAME yolo26_operators
  COMMAND yolo26_operator_test "${CMAKE_CURRENT_BINARY_DIR}/operator_models"
)
add_test(
  NAME yolo26_segmentation_operator
  COMMAND yolo26_segmentation_op_test "${CMAKE_CURRENT_BINARY_DIR}/segmentation_models"
)
set_tests_properties(
  yolo26_letterbox
  yolo26_decode
  yolo26_masks
  yolo26_operators
  yolo26_segmentation_operator
  PROPERTIES LABELS "yolo26;gpu" TIMEOUT 600
)

# The real YOLO26n-seg model on rendered stop signs. Skipped unless
# tools/prepare_model.py --task segment has written the model here; the first run builds
# and caches its engine beside it.
set(
  HOLOSCAN_YOLO26_SEG_MODEL
  "${PROJECT_SOURCE_DIR}/../../data/yolo26_detection/yolo26n-seg.onnx"
  CACHE FILEPATH
  "The YOLO26 segmentation model for the real-model test"
)
add_test(
  NAME yolo26_segmentation_real
  COMMAND
    yolo26_segmentation_op_test "${CMAKE_CURRENT_BINARY_DIR}/segmentation_models" --real
    "${HOLOSCAN_YOLO26_SEG_MODEL}"
)
set_tests_properties(
  yolo26_segmentation_real
  PROPERTIES LABELS "yolo26;gpu" TIMEOUT 900 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
