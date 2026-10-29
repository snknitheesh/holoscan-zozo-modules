# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_point_cloud_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-point-cloud 1.0 REQUIRED CONFIG)
message(
  STATUS
  "Found holoscan-point-cloud ${holoscan-point-cloud_VERSION} in ${holoscan-point-cloud_DIR}"
)

add_executable(point_cloud_consumer main.cpp)
target_link_libraries(point_cloud_consumer PRIVATE holoscan::point_cloud)
set_target_properties(point_cloud_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME point_cloud_installed_consumer COMMAND point_cloud_consumer)
set_tests_properties(
  point_cloud_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "point_cloud_consumer: PASS" TIMEOUT 300
)
