# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The Normals and Planes demo, built against the INSTALLED holoscan-point-cloud and
# holoscan-normals packages, as any other project would use them: put the modules' install
# prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(normals_planes LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NORMALS_PLANES_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(NORMALS_PLANES_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-point-cloud 1.1 REQUIRED CONFIG)
find_package(holoscan-normals 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The scene source, the view, the render check and the ground-truth reporter.
add_library(
  normals_planes_app_ops
  STATIC
  src/render_check_op.cpp
  src/report_op.cpp
  src/scene_source_op.cpp
  src/view_op.cpp
)
target_compile_features(normals_planes_app_ops PUBLIC cxx_std_20)
target_include_directories(normals_planes_app_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  normals_planes_app_ops
  PUBLIC holoscan::normals holoscan::core holoscan::sensor_io CUDA::cudart
)
# The reporter evaluates the scene as the GPU renders it: without contracting.
target_compile_options(normals_planes_app_ops PUBLIC -ffp-contract=off)

add_executable(normals_planes src/main.cpp)
target_link_libraries(normals_planes PRIVATE normals_planes_app_ops holoscan::point_cloud)
target_compile_definitions(
  normals_planes
  PRIVATE NORMALS_PLANES_ENABLE_VISUALIZATION=$<BOOL:${NORMALS_PLANES_ENABLE_VISUALIZATION}>
)
if(NORMALS_PLANES_ENABLE_VISUALIZATION)
  target_link_libraries(normals_planes PRIVATE holoscan::visualization::op_holoviz)
endif()
install(TARGETS normals_planes RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
