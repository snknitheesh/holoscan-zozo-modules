# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The SAM 3 Segmentation demo, built against the INSTALLED holoscan-sam3 package, as any
# other project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(sam3_segmentation LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SAM3_SEGMENTATION_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
set(_sam3_components "")
if(SAM3_SEGMENTATION_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
  list(APPEND _sam3_components overlay)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-sam3 1.0 REQUIRED CONFIG COMPONENTS ${_sam3_components})
find_package(CUDAToolkit REQUIRED)

# The image sources (synthetic, files, V4L2 camera), the image files, the reporter and
# the offscreen render check.
add_library(
  sam3_segmentation_ops
  STATIC
  src/image_files.cpp
  src/image_sources.cpp
  src/render_check_op.cpp
  src/segmentation_report_op.cpp
)
target_compile_features(sam3_segmentation_ops PUBLIC cxx_std_20)
target_include_directories(sam3_segmentation_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  sam3_segmentation_ops
  PUBLIC holoscan::sam3 holoscan::core holoscan::sensor_io CUDA::cudart
  PRIVATE CUDA::nvjpeg
)

add_executable(sam3_segmentation src/main.cpp)
target_link_libraries(sam3_segmentation PRIVATE sam3_segmentation_ops)
target_compile_definitions(
  sam3_segmentation
  PRIVATE SAM3_SEGMENTATION_ENABLE_VISUALIZATION=$<BOOL:${SAM3_SEGMENTATION_ENABLE_VISUALIZATION}>
)
if(SAM3_SEGMENTATION_ENABLE_VISUALIZATION)
  target_link_libraries(
    sam3_segmentation
    PRIVATE holoscan::sam3_overlay holoscan::visualization::op_holoviz
  )
endif()
install(TARGETS sam3_segmentation RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
