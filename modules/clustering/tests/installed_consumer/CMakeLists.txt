# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_clustering_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-clustering 1.0 REQUIRED CONFIG)
message(
  STATUS
  "Found holoscan-clustering ${holoscan-clustering_VERSION} in ${holoscan-clustering_DIR}"
)

add_executable(clustering_consumer main.cpp)
target_link_libraries(clustering_consumer PRIVATE holoscan::clustering)
set_target_properties(clustering_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME clustering_installed_consumer COMMAND clustering_consumer)
set_tests_properties(
  clustering_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "clustering_consumer: PASS" TIMEOUT 300
)
