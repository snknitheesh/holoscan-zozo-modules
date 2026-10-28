# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The AprilTag demo, built against the INSTALLED holoscan-apriltag package, as any other
# project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(apriltag_detection LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(APRILTAG_DETECTION_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
set(_apriltag_components)
if(APRILTAG_DETECTION_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
  list(APPEND _apriltag_components overlay)
endif()
if(BUILD_TESTING)
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-apriltag 1.0 REQUIRED CONFIG COMPONENTS ${_apriltag_components})
find_package(CUDAToolkit REQUIRED)

# The synthetic source and the ground-truth reporter.
add_library(apriltag_detection_ops STATIC src/detection_report_op.cpp src/frame_source_op.cpp)
target_compile_features(apriltag_detection_ops PUBLIC cxx_std_20)
target_include_directories(apriltag_detection_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  apriltag_detection_ops
  PUBLIC holoscan::apriltag holoscan::core holoscan::sensor_io CUDA::cudart
)

add_executable(apriltag_detection src/main.cpp)
target_link_libraries(apriltag_detection PRIVATE apriltag_detection_ops)
target_compile_definitions(
  apriltag_detection
  PRIVATE APRILTAG_DETECTION_ENABLE_VISUALIZATION=$<BOOL:${APRILTAG_DETECTION_ENABLE_VISUALIZATION}>
)
if(APRILTAG_DETECTION_ENABLE_VISUALIZATION)
  target_link_libraries(
    apriltag_detection
    PRIVATE holoscan::apriltag_overlay holoscan::visualization::op_holoviz
  )
  # On Jetson, VPI loads Tegra's EGL driver, after which HoloViz cannot create its Vulkan
  # instance unless the Vulkan loader was loaded first: link the loader ahead of everything.
  if(holoscan-apriltag_WITH_VPI)
    find_library(APRILTAG_DETECTION_VULKAN_LOADER NAMES vulkan REQUIRED)
    target_link_options(
      apriltag_detection
      BEFORE
      PRIVATE
        "LINKER:--push-state,--no-as-needed"
        "${APRILTAG_DETECTION_VULKAN_LOADER}"
        "LINKER:--pop-state"
    )
  endif()
endif()
install(TARGETS apriltag_detection RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
