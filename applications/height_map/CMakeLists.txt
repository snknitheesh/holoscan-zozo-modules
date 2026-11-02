# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The Height Map demo, built against the INSTALLED holoscan-point-cloud and
# holoscan-height-map packages, as any other project would use them: put the modules'
# install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(height_map LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HEIGHT_MAP_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(HEIGHT_MAP_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-point-cloud 1.1 REQUIRED CONFIG)
find_package(holoscan-height-map 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The camera, the render check and the ground-truth reporter.
add_library(
  height_map_app_ops
  STATIC
  src/render_check_op.cpp
  src/report_op.cpp
  src/terrain_camera_op.cpp
)
target_compile_features(height_map_app_ops PUBLIC cxx_std_20)
target_include_directories(height_map_app_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  height_map_app_ops
  PUBLIC holoscan::height_map holoscan::core holoscan::sensor_io CUDA::cudart
)
# The reporter evaluates the terrain as the GPU renders it: without contracting.
target_compile_options(height_map_app_ops PUBLIC -ffp-contract=off)

add_executable(height_map src/main.cpp)
target_link_libraries(height_map PRIVATE height_map_app_ops holoscan::point_cloud)
target_compile_definitions(
  height_map
  PRIVATE HEIGHT_MAP_ENABLE_VISUALIZATION=$<BOOL:${HEIGHT_MAP_ENABLE_VISUALIZATION}>
)
if(HEIGHT_MAP_ENABLE_VISUALIZATION)
  target_link_libraries(height_map PRIVATE holoscan::visualization::op_holoviz)
endif()
install(TARGETS height_map RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
