# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The Stereo Disparity demo, built against the INSTALLED holoscan-stereo-disparity
# package, as any other project would use it: put the module's install prefix on
# CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
endif()
project(stereo_disparity LANGUAGES CXX CUDA)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(STEREO_DISPARITY_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(STEREO_DISPARITY_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
if(BUILD_TESTING)
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-stereo-disparity 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The recording reader, which needs no SDK; the host tests build it on its own too.
add_library(stereo_disparity_recording STATIC src/recording.cpp)
target_compile_features(stereo_disparity_recording PUBLIC cxx_std_20)
target_include_directories(stereo_disparity_recording PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

# The sources, the reporter and the color map.
add_library(
  stereo_disparity_ops
  STATIC
  src/colorize.cu
  src/disparity_color_op.cpp
  src/disparity_report_op.cpp
  src/stereo_sources.cpp
)
target_compile_features(stereo_disparity_ops PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(stereo_disparity_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  stereo_disparity_ops
  PUBLIC
    stereo_disparity_recording
    holoscan::stereo_disparity
    holoscan::core
    holoscan::sensor_io
    CUDA::cudart
)

add_executable(stereo_disparity src/main.cpp)
target_link_libraries(stereo_disparity PRIVATE stereo_disparity_ops)
target_compile_definitions(
  stereo_disparity
  PRIVATE STEREO_DISPARITY_ENABLE_VISUALIZATION=$<BOOL:${STEREO_DISPARITY_ENABLE_VISUALIZATION}>
)
if(STEREO_DISPARITY_ENABLE_VISUALIZATION)
  target_link_libraries(stereo_disparity PRIVATE holoscan::visualization::op_holoviz)
  # On Jetson, VPI loads Tegra's EGL driver, after which HoloViz cannot create its Vulkan
  # instance unless the Vulkan loader was loaded first: link the loader ahead of everything.
  find_library(STEREO_DISPARITY_VULKAN_LOADER NAMES vulkan REQUIRED)
  target_link_options(
    stereo_disparity
    BEFORE
    PRIVATE
      "LINKER:--push-state,--no-as-needed"
      "${STEREO_DISPARITY_VULKAN_LOADER}"
      "LINKER:--pop-state"
  )
endif()
install(TARGETS stereo_disparity RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
