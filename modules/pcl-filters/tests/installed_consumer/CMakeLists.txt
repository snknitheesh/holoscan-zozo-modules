# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_pcl_filters_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-pcl-filters 1.0 REQUIRED CONFIG)
message(
  STATUS
  "Found holoscan-pcl-filters ${holoscan-pcl-filters_VERSION} in ${holoscan-pcl-filters_DIR}"
)

add_executable(pcl_filters_consumer main.cpp)
target_link_libraries(pcl_filters_consumer PRIVATE holoscan::pcl_filters)
set_target_properties(pcl_filters_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME pcl_filters_installed_consumer COMMAND pcl_filters_consumer)
set_tests_properties(
  pcl_filters_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "pcl_filters_consumer: PASS" TIMEOUT 300
)
