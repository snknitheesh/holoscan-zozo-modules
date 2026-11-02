# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The Mono Depth demo, built against the INSTALLED holoscan-mono-depth package, as any
# other project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(mono_depth LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MONO_DEPTH_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(MONO_DEPTH_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-mono-depth 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The image sources (synthetic, files, V4L2 camera), the image files, the reporter and
# the offscreen render check.
add_library(
  mono_depth_ops
  STATIC
  src/depth_report_op.cpp
  src/image_files.cpp
  src/image_sources.cpp
  src/render_check_op.cpp
)
target_compile_features(mono_depth_ops PUBLIC cxx_std_20)
target_include_directories(mono_depth_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  mono_depth_ops
  PUBLIC holoscan::mono_depth holoscan::core holoscan::sensor_io CUDA::cudart
  PRIVATE CUDA::nvjpeg
)

add_executable(mono_depth src/main.cpp)
target_link_libraries(mono_depth PRIVATE mono_depth_ops)
target_compile_definitions(
  mono_depth
  PRIVATE MONO_DEPTH_ENABLE_VISUALIZATION=$<BOOL:${MONO_DEPTH_ENABLE_VISUALIZATION}>
)
if(MONO_DEPTH_ENABLE_VISUALIZATION)
  target_link_libraries(mono_depth PRIVATE holoscan::visualization::op_holoviz)
endif()
install(TARGETS mono_depth RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
