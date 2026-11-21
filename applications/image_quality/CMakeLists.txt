# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The Image Quality demo, built against the INSTALLED holoscan-image-quality package, as
# any other project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(image_quality LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io visualization)
if(BUILD_TESTING)
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-image-quality 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The synthetic camera, the reporter, the text layer and the offscreen checker.
add_library(
  image_quality_ops
  STATIC
  src/camera_op.cpp
  src/render_check_op.cpp
  src/report_op.cpp
  src/segments.cpp
)
target_compile_features(image_quality_ops PUBLIC cxx_std_20)
target_include_directories(image_quality_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  image_quality_ops
  PUBLIC
    holoscan::image_quality
    holoscan::core
    holoscan::sensor_io
    holoscan::visualization::op_holoviz
    CUDA::cudart
)

add_executable(image_quality src/main.cpp)
target_link_libraries(image_quality PRIVATE image_quality_ops)
install(TARGETS image_quality RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
