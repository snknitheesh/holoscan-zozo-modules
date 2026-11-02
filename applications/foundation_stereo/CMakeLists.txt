# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The Stereo Disparity demo, built against the INSTALLED holoscan-foundation-stereo
# package, as any other project would use it: put the module's install prefix on
# CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
endif()
project(foundation_stereo LANGUAGES CXX CUDA)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(FOUNDATION_STEREO_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(FOUNDATION_STEREO_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
if(BUILD_TESTING)
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-foundation-stereo 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The recording reader, which needs no SDK; the host tests build it on its own too.
add_library(foundation_stereo_recording STATIC src/recording.cpp)
target_compile_features(foundation_stereo_recording PUBLIC cxx_std_20)
target_include_directories(foundation_stereo_recording PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

# The sources, the reporter and the color map.
add_library(
  foundation_stereo_ops
  STATIC
  src/colorize.cu
  src/disparity_color_op.cpp
  src/disparity_report_op.cpp
  src/render_check_op.cpp
  src/stereo_sources.cpp
)
target_compile_features(foundation_stereo_ops PUBLIC cxx_std_20 PRIVATE cuda_std_20)
# The reporter renders the synthetic scene on the host to check the stub model's output:
# without contracted multiply-adds, it computes the same bytes as the module's GPU renderer.
set_source_files_properties(
  src/disparity_report_op.cpp
  PROPERTIES COMPILE_OPTIONS -ffp-contract=off
)
target_include_directories(foundation_stereo_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  foundation_stereo_ops
  PUBLIC
    foundation_stereo_recording
    holoscan::foundation_stereo
    holoscan::core
    holoscan::sensor_io
    CUDA::cudart
)

add_executable(foundation_stereo src/main.cpp)
target_link_libraries(foundation_stereo PRIVATE foundation_stereo_ops)
target_compile_definitions(
  foundation_stereo
  PRIVATE FOUNDATION_STEREO_ENABLE_VISUALIZATION=$<BOOL:${FOUNDATION_STEREO_ENABLE_VISUALIZATION}>
)
if(FOUNDATION_STEREO_ENABLE_VISUALIZATION)
  target_link_libraries(foundation_stereo PRIVATE holoscan::visualization::op_holoviz)
endif()
install(TARGETS foundation_stereo RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
