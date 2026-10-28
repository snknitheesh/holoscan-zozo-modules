# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The Depth to Point Cloud demo, built against the INSTALLED holoscan-point-cloud package,
# as any other project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(depth_to_point_cloud LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(BUILD_TESTING)
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-point-cloud 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The sources, the ground-truth reporter and the PGM and PLY files.
add_library(
  depth_to_point_cloud_ops
  STATIC
  src/depth_sources.cpp
  src/image_io.cpp
  src/point_cloud_report_op.cpp
)
target_compile_features(depth_to_point_cloud_ops PUBLIC cxx_std_20)
target_include_directories(depth_to_point_cloud_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  depth_to_point_cloud_ops
  PUBLIC holoscan::point_cloud holoscan::core holoscan::sensor_io CUDA::cudart
)
# The ground truth evaluates the synthetic camera on the host, as the module renders it
# on the GPU: without contracting, whatever the compiler's default. Everything that
# computes the truth links this library.
target_compile_options(depth_to_point_cloud_ops PUBLIC -ffp-contract=off)

add_executable(depth_to_point_cloud src/main.cpp)
target_link_libraries(depth_to_point_cloud PRIVATE depth_to_point_cloud_ops)
install(TARGETS depth_to_point_cloud RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
