# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The recorder demo, built against the INSTALLED holoscan-recorder package, as any other
# project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(recorder_demo LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RECORDER_ENABLE_VISUALIZATION "Build the HoloViz display of the replay mode" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(RECORDER_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-recorder 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The synthetic source and the sinks that check and count what is replayed.
add_library(recorder_demo_ops STATIC src/demo_ops.cpp)
target_compile_features(recorder_demo_ops PUBLIC cxx_std_20)
target_include_directories(recorder_demo_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  recorder_demo_ops
  PUBLIC holoscan::recorder holoscan::core holoscan::sensor_io CUDA::cudart
)

add_executable(recorder src/main.cpp)
target_link_libraries(recorder PRIVATE recorder_demo_ops)
target_compile_definitions(
  recorder
  PRIVATE RECORDER_ENABLE_VISUALIZATION=$<BOOL:${RECORDER_ENABLE_VISUALIZATION}>
)
if(RECORDER_ENABLE_VISUALIZATION)
  target_link_libraries(recorder PRIVATE holoscan::visualization::op_holoviz)
endif()
install(TARGETS recorder RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
