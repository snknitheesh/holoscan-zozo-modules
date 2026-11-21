# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. The kernel and operator tests need the SDK, CUDA and a GPU;
# none needs a download: the stub model stands in for Depth Anything. The real-model tests
# run when HOLOSCAN_MONO_DEPTH_MODEL names an exported model, and are skipped otherwise.
set(
  HOLOSCAN_MONO_DEPTH_MODEL
  ""
  CACHE FILEPATH
  "A Depth Anything V2 model from tools/prepare_model.py, for the real-model tests"
)

add_executable(mono_depth_kernels_test kernels_test.cu)
target_link_libraries(mono_depth_kernels_test PRIVATE holoscan::mono_depth)
target_compile_features(mono_depth_kernels_test PRIVATE cuda_std_20)
# The tests' host references must not contract either.
target_compile_options(mono_depth_kernels_test PRIVATE -fmad=false -Xcompiler=-ffp-contract=off)

add_executable(mono_depth_operator_test operator_test.cpp)
add_executable(mono_depth_real_model_test real_model_test.cpp)
foreach(_test mono_depth_operator_test mono_depth_real_model_test)
  target_link_libraries(${_test} PRIVATE holoscan::mono_depth holoscan::testing)
  target_compile_options(${_test} PRIVATE -ffp-contract=off)
endforeach()

add_test(NAME mono_depth_kernels COMMAND mono_depth_kernels_test)
# The stub models and their TensorRT engines go to a folder of their own.
add_test(
  NAME mono_depth_operators
  COMMAND mono_depth_operator_test "${CMAKE_CURRENT_BINARY_DIR}/operator_models"
)
set_tests_properties(
  mono_depth_kernels
  mono_depth_operators
  PROPERTIES LABELS "mono_depth;gpu" TIMEOUT 900
)
# The engine is built beside the model, once per GPU and TensorRT version, which takes
# minutes on Jetson Thor and Orin.
add_test(
  NAME mono_depth_real_model
  COMMAND mono_depth_real_model_test "${HOLOSCAN_MONO_DEPTH_MODEL}"
)
# The same model on the local recordings, when HOLOSCAN_TEST_DATA names them; skipped
# otherwise, as in CI. Their depth is in units of 0.000452 m.
add_test(
  NAME mono_depth_real_data
  COMMAND
    mono_depth_real_model_test "${HOLOSCAN_MONO_DEPTH_MODEL}" --recorded --depth-scale 0.000452
)
set_tests_properties(
  mono_depth_real_model
  mono_depth_real_data
  PROPERTIES LABELS "mono_depth;gpu;real_model" TIMEOUT 3600 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
