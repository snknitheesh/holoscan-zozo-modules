# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The YOLO26 demo, built against the INSTALLED holoscan-yolo26 package, as any other
# project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(yolo26_detection LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(YOLO26_DETECTION_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
set(_yolo26_components)
if(YOLO26_DETECTION_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
  list(APPEND _yolo26_components overlay)
endif()
if(BUILD_TESTING)
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-yolo26 1.0 REQUIRED CONFIG COMPONENTS ${_yolo26_components})
find_package(CUDAToolkit REQUIRED)

# The image sources (synthetic, files, V4L2 camera), the image files, the expected
# detections and the reporter.
add_library(
  yolo26_detection_ops
  STATIC
  src/detection_report_op.cpp
  src/expectations.cpp
  src/image_files.cpp
  src/image_sources.cpp
)
target_compile_features(yolo26_detection_ops PUBLIC cxx_std_20)
target_include_directories(yolo26_detection_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  yolo26_detection_ops
  PUBLIC holoscan::yolo26 holoscan::core holoscan::sensor_io CUDA::cudart
  PRIVATE CUDA::nvjpeg
)

add_executable(yolo26_detection src/main.cpp)
target_link_libraries(yolo26_detection PRIVATE yolo26_detection_ops)
target_compile_definitions(
  yolo26_detection
  PRIVATE YOLO26_DETECTION_ENABLE_VISUALIZATION=$<BOOL:${YOLO26_DETECTION_ENABLE_VISUALIZATION}>
)
if(YOLO26_DETECTION_ENABLE_VISUALIZATION)
  target_link_libraries(
    yolo26_detection
    PRIVATE holoscan::yolo26_overlay holoscan::visualization::op_holoviz
  )
endif()
install(TARGETS yolo26_detection RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
