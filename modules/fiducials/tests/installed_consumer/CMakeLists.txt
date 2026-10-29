# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on CMAKE_PREFIX_PATH
# (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_fiducials_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-fiducials 1.0 REQUIRED CONFIG)
message(STATUS "Found holoscan-fiducials ${holoscan-fiducials_VERSION} in ${holoscan-fiducials_DIR}")

add_executable(fiducials_consumer main.cpp)
target_link_libraries(fiducials_consumer PRIVATE holoscan::fiducials)

add_test(NAME fiducials_installed_consumer COMMAND fiducials_consumer)
set_tests_properties(
  fiducials_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "fiducials_consumer: PASS" TIMEOUT 300
)
