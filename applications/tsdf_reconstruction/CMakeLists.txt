# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The TSDF Reconstruction demo, built against the INSTALLED holoscan-tsdf package, as any other
# project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(tsdf_reconstruction LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TSDF_RECONSTRUCTION_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(TSDF_RECONSTRUCTION_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-tsdf 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The camera, the view, the render check and the ground-truth reporters.
add_library(
  tsdf_reconstruction_app_ops
  STATIC
  src/render_check_op.cpp
  src/report_op.cpp
  src/scene_camera_op.cpp
  src/view_op.cpp
)
target_compile_features(tsdf_reconstruction_app_ops PUBLIC cxx_std_20)
target_include_directories(tsdf_reconstruction_app_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  tsdf_reconstruction_app_ops
  PUBLIC holoscan::tsdf holoscan::core holoscan::sensor_io CUDA::cudart
)
# The reporters evaluate the scene as the GPU renders it: without contracting.
target_compile_options(tsdf_reconstruction_app_ops PUBLIC -ffp-contract=off)

add_executable(tsdf_reconstruction src/main.cpp)
target_link_libraries(tsdf_reconstruction PRIVATE tsdf_reconstruction_app_ops)
target_compile_definitions(
  tsdf_reconstruction
  PRIVATE
    TSDF_RECONSTRUCTION_ENABLE_VISUALIZATION=$<BOOL:${TSDF_RECONSTRUCTION_ENABLE_VISUALIZATION}>
)
if(TSDF_RECONSTRUCTION_ENABLE_VISUALIZATION)
  target_link_libraries(tsdf_reconstruction PRIVATE holoscan::visualization::op_holoviz)
endif()
install(TARGETS tsdf_reconstruction RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
