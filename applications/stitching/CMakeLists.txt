# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The stitching demo, built against the INSTALLED holoscan-stitching package, as any other
# project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(stitching_app LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(STITCHING_APP_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(STITCHING_APP_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-stitching 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The synthetic rig and the checks.
add_library(stitching_app_ops STATIC src/check_ops.cpp src/rig_source_op.cpp)
target_compile_features(stitching_app_ops PUBLIC cxx_std_20)
target_include_directories(stitching_app_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  stitching_app_ops
  PUBLIC holoscan::stitching holoscan::core holoscan::sensor_io CUDA::cudart
)

add_executable(stitching src/main.cpp)
target_link_libraries(stitching PRIVATE stitching_app_ops)
target_compile_definitions(
  stitching
  PRIVATE STITCHING_APP_ENABLE_VISUALIZATION=$<BOOL:${STITCHING_APP_ENABLE_VISUALIZATION}>
)
if(STITCHING_APP_ENABLE_VISUALIZATION)
  target_link_libraries(stitching PRIVATE holoscan::visualization::op_holoviz)
endif()
install(TARGETS stitching RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
