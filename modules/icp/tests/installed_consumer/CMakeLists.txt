# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_icp_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-icp 1.0 REQUIRED CONFIG)
message(STATUS "Found holoscan-icp ${holoscan-icp_VERSION} in ${holoscan-icp_DIR}")

add_executable(icp_consumer main.cpp)
target_link_libraries(icp_consumer PRIVATE holoscan::icp)
set_target_properties(icp_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME icp_installed_consumer COMMAND icp_consumer)
set_tests_properties(
  icp_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "icp_consumer: PASS" TIMEOUT 300
)
