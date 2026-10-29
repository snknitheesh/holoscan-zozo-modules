# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The Rectify demo, built against the INSTALLED holoscan-rectify package, as any other
# project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(rectify_demo LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RECTIFY_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(RECTIFY_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
if(BUILD_TESTING)
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-rectify 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The stereo source, the recording reader and the reporter.
add_library(rectify_demo_ops STATIC src/recording.cpp src/report_op.cpp src/stereo_source_op.cpp)
target_compile_features(rectify_demo_ops PUBLIC cxx_std_20)
target_include_directories(rectify_demo_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  rectify_demo_ops
  PUBLIC holoscan::rectify holoscan::core holoscan::sensor_io CUDA::cudart
)

add_executable(rectify src/main.cpp)
target_link_libraries(rectify PRIVATE rectify_demo_ops)
target_compile_definitions(
  rectify
  PRIVATE RECTIFY_ENABLE_VISUALIZATION=$<BOOL:${RECTIFY_ENABLE_VISUALIZATION}>
)
if(RECTIFY_ENABLE_VISUALIZATION)
  target_link_libraries(rectify PRIVATE holoscan::visualization::op_holoviz)
  # On Jetson, VPI loads Tegra's EGL driver, after which HoloViz cannot create its Vulkan
  # instance unless the Vulkan loader was loaded first: link the loader ahead of everything.
  if(holoscan-rectify_WITH_VPI)
    find_library(RECTIFY_VULKAN_LOADER NAMES vulkan REQUIRED)
    target_link_options(
      rectify
      BEFORE
      PRIVATE "LINKER:--push-state,--no-as-needed" "${RECTIFY_VULKAN_LOADER}" "LINKER:--pop-state"
    )
  endif()
endif()
install(TARGETS rectify RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
