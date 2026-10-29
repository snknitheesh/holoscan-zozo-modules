# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The ICP Odometry demo, built against the INSTALLED holoscan-icp and holoscan-point-cloud
# packages, as any other project would use them: put the modules' install prefix on
# CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(icp_odometry LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ICP_ODOMETRY_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(ICP_ODOMETRY_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-icp 1.0 REQUIRED CONFIG)
find_package(holoscan-point-cloud 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The synthetic depth camera and the ground-truth reporter.
add_library(icp_odometry_ops STATIC src/depth_camera_op.cpp src/odometry_report_op.cpp)
target_compile_features(icp_odometry_ops PUBLIC cxx_std_20)
target_include_directories(icp_odometry_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  icp_odometry_ops
  PUBLIC holoscan::icp holoscan::core holoscan::sensor_io CUDA::cudart
)

add_executable(icp_odometry src/main.cpp)
target_link_libraries(icp_odometry PRIVATE icp_odometry_ops holoscan::point_cloud)
target_compile_definitions(
  icp_odometry
  PRIVATE ICP_ODOMETRY_ENABLE_VISUALIZATION=$<BOOL:${ICP_ODOMETRY_ENABLE_VISUALIZATION}>
)
if(ICP_ODOMETRY_ENABLE_VISUALIZATION)
  target_sources(icp_odometry PRIVATE src/trajectory_view_op.cpp)
  target_link_libraries(icp_odometry PRIVATE holoscan::visualization::op_holoviz)
endif()
install(TARGETS icp_odometry RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
