# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The fiducials demo, built against the INSTALLED holoscan-fiducials package, as any other
# project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(fiducial_detection LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FIDUCIAL_DETECTION_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
set(_fiducials_components)
if(FIDUCIAL_DETECTION_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
  list(APPEND _fiducials_components overlay)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-fiducials 1.0 REQUIRED CONFIG COMPONENTS ${_fiducials_components})
find_package(CUDAToolkit REQUIRED)

# The sources and the ground-truth reporter.
add_library(fiducial_detection_ops STATIC src/report_op.cpp src/sources.cpp)
target_compile_features(fiducial_detection_ops PUBLIC cxx_std_20)
target_include_directories(fiducial_detection_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  fiducial_detection_ops
  PUBLIC holoscan::fiducials holoscan::core holoscan::sensor_io CUDA::cudart
)

add_executable(fiducial_detection src/main.cpp)
target_link_libraries(fiducial_detection PRIVATE fiducial_detection_ops)
target_compile_definitions(
  fiducial_detection
  PRIVATE
    FIDUCIAL_DETECTION_ENABLE_VISUALIZATION=$<BOOL:${FIDUCIAL_DETECTION_ENABLE_VISUALIZATION}>
)
if(FIDUCIAL_DETECTION_ENABLE_VISUALIZATION)
  target_link_libraries(
    fiducial_detection
    PRIVATE holoscan::fiducials_overlay holoscan::visualization::op_holoviz
  )
endif()
install(TARGETS fiducial_detection RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
