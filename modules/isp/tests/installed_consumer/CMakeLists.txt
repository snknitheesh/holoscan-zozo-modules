# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_isp_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-isp 1.0 REQUIRED CONFIG)
message(STATUS "Found holoscan-isp ${holoscan-isp_VERSION} in ${holoscan-isp_DIR}")

add_executable(isp_consumer main.cpp)
target_link_libraries(isp_consumer PRIVATE holoscan::isp)
set_target_properties(isp_consumer PROPERTIES CXX_EXTENSIONS OFF)
# It evaluates the synthetic camera on the host, as the module does: without contracting.
target_compile_options(isp_consumer PRIVATE -ffp-contract=off)

add_test(NAME isp_installed_consumer COMMAND isp_consumer)
set_tests_properties(
  isp_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "isp_consumer: PASS" TIMEOUT 300
)
