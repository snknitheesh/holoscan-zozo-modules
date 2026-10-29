# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The Stereo Camera demo, built against the INSTALLED holoscan-stereo-camera package, as any
# other project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(stereo_camera LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(STEREO_CAMERA_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(STEREO_CAMERA_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-stereo-camera 1.0 REQUIRED CONFIG)

# The reporter, which checks every frame.
add_library(stereo_camera_ops STATIC src/frame_report_op.cpp)
target_compile_features(stereo_camera_ops PUBLIC cxx_std_20)
target_include_directories(stereo_camera_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  stereo_camera_ops
  PUBLIC holoscan::stereo_camera holoscan::core holoscan::sensor_io
)

add_executable(stereo_camera src/main.cpp)
target_link_libraries(stereo_camera PRIVATE stereo_camera_ops)
target_compile_definitions(
  stereo_camera
  PRIVATE STEREO_CAMERA_ENABLE_VISUALIZATION=$<BOOL:${STEREO_CAMERA_ENABLE_VISUALIZATION}>
)
if(STEREO_CAMERA_ENABLE_VISUALIZATION)
  target_link_libraries(stereo_camera PRIVATE holoscan::visualization::op_holoviz)
endif()
install(TARGETS stereo_camera RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
