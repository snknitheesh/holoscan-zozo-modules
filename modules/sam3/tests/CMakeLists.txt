# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. The kernel and operator tests need the SDK, CUDA and a GPU;
# none needs a download: the stub model stands in for SAM 3, on every CI lane. The
# real-model test runs when HOLOSCAN_SAM3_MODEL_DIR holds an export of
# tools/prepare_model.py, and is skipped otherwise; it defaults to the sam3 folder of the
# test data ($HOLOSCAN_TEST_DATA) when that is set, and to data/sam3 otherwise.
if(NOT "$ENV{HOLOSCAN_TEST_DATA}" STREQUAL "")
  set(_sam3_model_dir "$ENV{HOLOSCAN_TEST_DATA}/sam3")
else()
  set(_sam3_model_dir "${PROJECT_SOURCE_DIR}/../../data/sam3")
endif()
set(
  HOLOSCAN_SAM3_MODEL_DIR
  "${_sam3_model_dir}"
  CACHE PATH
  "A folder with tools/prepare_model.py's SAM 3 export, for the real-model test"
)

add_executable(sam3_kernels_test kernels_test.cu)
target_link_libraries(sam3_kernels_test PRIVATE holoscan::sam3)
target_compile_features(sam3_kernels_test PRIVATE cuda_std_20)
# The tests' host references must not contract either.
target_compile_options(sam3_kernels_test PRIVATE -fmad=false -Xcompiler=-ffp-contract=off)

add_executable(sam3_operator_test operator_test.cpp)
add_executable(sam3_real_model_test real_model_test.cpp)
foreach(_test sam3_operator_test sam3_real_model_test)
  target_link_libraries(${_test} PRIVATE holoscan::sam3 holoscan::testing)
  target_compile_options(${_test} PRIVATE -ffp-contract=off)
endforeach()
if(TARGET holoscan_sam3_overlay)
  target_link_libraries(sam3_operator_test PRIVATE holoscan_sam3_overlay)
  target_compile_definitions(sam3_operator_test PRIVATE HOLOSCAN_SAM3_TEST_OVERLAY=1)
endif()

add_test(NAME sam3_kernels COMMAND sam3_kernels_test)
# The stub models and their TensorRT engines go to a folder of their own.
add_test(
  NAME sam3_operators
  COMMAND sam3_operator_test "${CMAKE_CURRENT_BINARY_DIR}/operator_models"
)
set_tests_properties(sam3_kernels sam3_operators PROPERTIES LABELS "sam3;gpu" TIMEOUT 900)
# The engines are built beside the models, once per GPU and TensorRT version, which
# takes minutes, and longer on Jetson Orin and Jetson Thor.
add_test(NAME sam3_real_model COMMAND sam3_real_model_test "${HOLOSCAN_SAM3_MODEL_DIR}")
set_tests_properties(
  sam3_real_model
  PROPERTIES LABELS "sam3;gpu;real_model" TIMEOUT 3600 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
