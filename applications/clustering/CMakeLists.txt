# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The Clustering demo, built against the INSTALLED holoscan-point-cloud and
# holoscan-clustering packages, as any other project would use them: put the modules'
# install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(clustering LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CLUSTERING_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(CLUSTERING_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-point-cloud 1.1 REQUIRED CONFIG)
find_package(holoscan-clustering 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The scene source, the ground-truth reporter and the display's drawing.
add_library(
  clustering_app_ops
  STATIC
  src/cluster_report_op.cpp
  src/cluster_view_op.cpp
  src/scene_source_op.cpp
)
target_compile_features(clustering_app_ops PUBLIC cxx_std_20)
target_include_directories(clustering_app_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  clustering_app_ops
  PUBLIC holoscan::clustering holoscan::core holoscan::sensor_io CUDA::cudart
)
# The reporter evaluates the scene as it renders: without contracting, whatever the
# compiler's default.
target_compile_options(clustering_app_ops PUBLIC -ffp-contract=off)

add_executable(clustering src/main.cpp)
target_link_libraries(clustering PRIVATE clustering_app_ops holoscan::point_cloud)
target_compile_definitions(
  clustering
  PRIVATE CLUSTERING_ENABLE_VISUALIZATION=$<BOOL:${CLUSTERING_ENABLE_VISUALIZATION}>
)
if(CLUSTERING_ENABLE_VISUALIZATION)
  target_link_libraries(clustering PRIVATE holoscan::visualization::op_holoviz)
endif()
install(TARGETS clustering RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
