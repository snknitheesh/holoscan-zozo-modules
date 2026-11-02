# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_image_quality_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-image-quality 1.0 REQUIRED CONFIG)
message(
  STATUS
  "Found holoscan-image-quality ${holoscan-image-quality_VERSION} in ${holoscan-image-quality_DIR}"
)

add_executable(image_quality_consumer main.cpp)
target_link_libraries(image_quality_consumer PRIVATE holoscan::image_quality)
set_target_properties(image_quality_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME image_quality_installed_consumer COMMAND image_quality_consumer)
set_tests_properties(
  image_quality_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "image_quality_consumer: PASS" TIMEOUT 300
)
