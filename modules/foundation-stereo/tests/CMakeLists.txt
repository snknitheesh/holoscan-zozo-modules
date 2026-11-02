# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The module's full suite. The kernel and operator tests need the SDK, CUDA and a GPU;
# none needs a download: the stub model stands in for Fast-FoundationStereo. The test of
# the real model is skipped (exit 77) unless tools/prepare_model.py has written it to
# HOLOSCAN_FOUNDATION_STEREO_MODEL.
set(
  HOLOSCAN_FOUNDATION_STEREO_MODEL
  "${PROJECT_SOURCE_DIR}/../../data/foundation_stereo/c_fast_foundationstereo_480x736.onnx"
  CACHE FILEPATH
  "A Fast-FoundationStereo ONNX model for the real-model test"
)

add_executable(foundation_stereo_kernels_test kernels_test.cu)
target_link_libraries(foundation_stereo_kernels_test PRIVATE holoscan::foundation_stereo)
target_compile_features(foundation_stereo_kernels_test PRIVATE cuda_std_20)

add_executable(foundation_stereo_operator_test operator_test.cpp)
add_executable(foundation_stereo_real_model_test real_model_test.cpp)
foreach(_test foundation_stereo_operator_test foundation_stereo_real_model_test)
  target_link_libraries(${_test} PRIVATE holoscan::foundation_stereo holoscan::testing)
endforeach()

add_test(NAME foundation_stereo_kernels COMMAND foundation_stereo_kernels_test)
# The stub models and their TensorRT engines go to a folder of their own.
add_test(
  NAME foundation_stereo_operator
  COMMAND foundation_stereo_operator_test "${CMAKE_CURRENT_BINARY_DIR}/operator_models"
)
# The first run builds the real model's engine, which takes minutes, and caches it.
add_test(
  NAME foundation_stereo_real_model
  COMMAND foundation_stereo_real_model_test "${HOLOSCAN_FOUNDATION_STEREO_MODEL}"
)
set_tests_properties(
  foundation_stereo_kernels
  foundation_stereo_operator
  foundation_stereo_real_model
  PROPERTIES LABELS "foundation_stereo;gpu" TIMEOUT 1200
)
# Building the engine takes about 2 minutes on an RTX 5090 and 45 on a Jetson Orin.
set_tests_properties(
  foundation_stereo_real_model
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 3600
)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
