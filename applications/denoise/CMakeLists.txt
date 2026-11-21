# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The denoise demo, built against the INSTALLED holoscan-denoise package, as any other
# project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
endif()
project(denoise_demo LANGUAGES CXX CUDA)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(DENOISE_DEMO_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(DENOISE_DEMO_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-denoise 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The scene source, the checker, the view and the render check.
add_library(
  denoise_demo_ops
  STATIC
  src/check_op.cpp
  src/render_check_op.cpp
  src/scene_source_op.cpp
  src/view_op.cu
)
target_compile_features(denoise_demo_ops PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(denoise_demo_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  denoise_demo_ops
  PUBLIC holoscan::denoise holoscan::core holoscan::sensor_io CUDA::cudart
)

add_executable(denoise src/main.cpp)
target_link_libraries(denoise PRIVATE denoise_demo_ops)
target_compile_definitions(
  denoise
  PRIVATE DENOISE_DEMO_ENABLE_VISUALIZATION=$<BOOL:${DENOISE_DEMO_ENABLE_VISUALIZATION}>
)
if(DENOISE_DEMO_ENABLE_VISUALIZATION)
  target_link_libraries(denoise PRIVATE holoscan::visualization::op_holoviz)
  # On Jetson, VPI loads Tegra's EGL driver, after which HoloViz cannot create its Vulkan
  # instance unless the Vulkan loader was loaded first: link the loader ahead of everything.
  if(holoscan-denoise_WITH_VPI)
    find_library(DENOISE_DEMO_VULKAN_LOADER NAMES vulkan REQUIRED)
    target_link_options(
      denoise
      BEFORE
      PRIVATE
        "LINKER:--push-state,--no-as-needed"
        "${DENOISE_DEMO_VULKAN_LOADER}"
        "LINKER:--pop-state"
    )
  endif()
endif()
install(TARGETS denoise RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
