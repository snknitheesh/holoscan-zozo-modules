# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# The holoscan-messages demo, built against the INSTALLED holoscan-messages package, as any
# other project would use it: put the module's install prefix on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 4.0)
project(messages_demo LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTING "Build the demo's tests" OFF)

find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core)
find_package(holoscan-messages 1.0 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

add_executable(messages_demo src/demo_ops.cpp src/main.cpp src/ros_files.cpp)
target_compile_features(messages_demo PRIVATE cxx_std_20)
target_link_libraries(messages_demo PRIVATE holoscan::messages holoscan::core CUDA::cudart)
install(TARGETS messages_demo RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
