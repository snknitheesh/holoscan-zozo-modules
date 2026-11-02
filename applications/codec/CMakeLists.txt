# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The codec demo, built against the INSTALLED holoscan-codec package, as any other project would
# use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(codec_demo LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTING "Build the demo's tests" OFF)

find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io visualization)
find_package(holoscan-codec 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

add_executable(codec_demo src/main.cpp src/report_op.cpp src/test_source_op.cpp)
target_compile_features(codec_demo PRIVATE cxx_std_20)
target_link_libraries(
  codec_demo
  PRIVATE
    holoscan::codec
    holoscan::core
    holoscan::sensor_io
    holoscan::visualization::op_holoviz
    CUDA::cudart
)
install(TARGETS codec_demo RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
