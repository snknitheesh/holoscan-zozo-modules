# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The ISP demo, built against the INSTALLED holoscan-isp package, as any other project would
# use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
endif()
project(isp_demo LANGUAGES CXX CUDA)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(ISP_DEMO_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(ISP_DEMO_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-isp 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The raw source, the checker, the view and the render check.
add_library(
  isp_demo_ops
  STATIC
  src/check_op.cpp
  src/raw_source_op.cpp
  src/render_check_op.cpp
  src/view_op.cu
)
target_compile_features(isp_demo_ops PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(isp_demo_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  isp_demo_ops
  PUBLIC holoscan::isp holoscan::core holoscan::sensor_io CUDA::cudart
)
# The checker evaluates the synthetic camera as the GPU renders it: without contracting.
target_compile_options(
  isp_demo_ops
  PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
)

add_executable(isp src/main.cpp)
target_link_libraries(isp PRIVATE isp_demo_ops)
target_compile_definitions(
  isp
  PRIVATE ISP_DEMO_ENABLE_VISUALIZATION=$<BOOL:${ISP_DEMO_ENABLE_VISUALIZATION}>
)
if(ISP_DEMO_ENABLE_VISUALIZATION)
  target_link_libraries(isp PRIVATE holoscan::visualization::op_holoviz)
endif()
install(TARGETS isp RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
