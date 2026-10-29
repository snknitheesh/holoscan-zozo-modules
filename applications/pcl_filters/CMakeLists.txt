# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The PCL Filters demo, built against the INSTALLED holoscan-point-cloud and
# holoscan-pcl-filters packages, as any other project would use them: put the modules'
# install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(pcl_filters LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PCL_FILTERS_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(PCL_FILTERS_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-point-cloud 1.1 REQUIRED CONFIG)
find_package(holoscan-pcl-filters 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The scene source and the ground-truth reporter.
add_library(pcl_filters_app_ops STATIC src/filter_report_op.cpp src/scene_source_op.cpp)
target_compile_features(pcl_filters_app_ops PUBLIC cxx_std_20)
target_include_directories(pcl_filters_app_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  pcl_filters_app_ops
  PUBLIC holoscan::pcl_filters holoscan::core holoscan::sensor_io CUDA::cudart
)
# The reporter counts voxels as the GPU does and evaluates the scene as it renders: without
# contracting, whatever the compiler's default.
target_compile_options(pcl_filters_app_ops PUBLIC -ffp-contract=off)

add_executable(pcl_filters src/main.cpp)
target_link_libraries(pcl_filters PRIVATE pcl_filters_app_ops holoscan::point_cloud)
target_compile_definitions(
  pcl_filters
  PRIVATE PCL_FILTERS_ENABLE_VISUALIZATION=$<BOOL:${PCL_FILTERS_ENABLE_VISUALIZATION}>
)
if(PCL_FILTERS_ENABLE_VISUALIZATION)
  target_link_libraries(pcl_filters PRIVATE holoscan::visualization::op_holoviz)
endif()
install(TARGETS pcl_filters RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
