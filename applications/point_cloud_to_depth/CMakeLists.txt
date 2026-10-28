# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The Point Cloud to Depth demo, built against the INSTALLED holoscan-point-cloud package,
# as any other project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(point_cloud_to_depth LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(BUILD_TESTING)
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-point-cloud 1.1 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The sources, the ground-truth reporter and the PLY, PGM and PPM files.
add_library(
  point_cloud_to_depth_ops
  STATIC
  src/cloud_io.cpp
  src/depth_image_report_op.cpp
  src/point_cloud_sources.cpp
)
target_compile_features(point_cloud_to_depth_ops PUBLIC cxx_std_20)
target_include_directories(point_cloud_to_depth_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  point_cloud_to_depth_ops
  PUBLIC holoscan::point_cloud holoscan::core holoscan::sensor_io CUDA::cudart
)
# Host code that makes or checks the synthetic clouds computes what the module computes
# on the GPU: without contracting, whatever the compiler's default. Everything that
# does links this library.
target_compile_options(point_cloud_to_depth_ops PUBLIC -ffp-contract=off)

add_executable(point_cloud_to_depth src/main.cpp)
target_link_libraries(point_cloud_to_depth PRIVATE point_cloud_to_depth_ops)
install(TARGETS point_cloud_to_depth RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
