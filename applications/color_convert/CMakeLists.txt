# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The colour conversion demo, built against the INSTALLED holoscan-color-convert package,
# as any other project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(color_convert_app LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(COLOR_CONVERT_APP_ENABLE_VISUALIZATION "Build the HoloViz display" ON)
option(BUILD_TESTING "Build the demo's tests" OFF)

set(_holoscan_components core sensor_io)
if(COLOR_CONVERT_APP_ENABLE_VISUALIZATION)
  list(APPEND _holoscan_components visualization)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(holoscan-color-convert 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# The synthetic camera and the checks.
add_library(color_convert_app_ops STATIC src/camera_op.cpp src/check_ops.cpp)
target_compile_features(color_convert_app_ops PUBLIC cxx_std_20)
target_include_directories(color_convert_app_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(
  color_convert_app_ops
  PUBLIC holoscan::color_convert holoscan::core holoscan::sensor_io CUDA::cudart
)

add_executable(color_convert src/main.cpp)
target_link_libraries(color_convert PRIVATE color_convert_app_ops)
target_compile_definitions(
  color_convert
  PRIVATE COLOR_CONVERT_APP_ENABLE_VISUALIZATION=$<BOOL:${COLOR_CONVERT_APP_ENABLE_VISUALIZATION}>
)
if(COLOR_CONVERT_APP_ENABLE_VISUALIZATION)
  target_link_libraries(color_convert PRIVATE holoscan::visualization::op_holoviz)
  # On Jetson, VPI loads Tegra's EGL driver, after which HoloViz cannot create its Vulkan
  # instance unless the Vulkan loader was loaded first: link the loader ahead of everything.
  if(holoscan-color-convert_WITH_VPI)
    find_library(COLOR_CONVERT_APP_VULKAN_LOADER NAMES vulkan REQUIRED)
    target_link_options(
      color_convert
      BEFORE
      PRIVATE
        "LINKER:--push-state,--no-as-needed"
        "${COLOR_CONVERT_APP_VULKAN_LOADER}"
        "LINKER:--pop-state"
    )
  endif()
endif()
install(TARGETS color_convert RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
