# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_messages_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-messages 1.0 REQUIRED CONFIG)
message(STATUS "Found holoscan-messages ${holoscan-messages_VERSION} in ${holoscan-messages_DIR}")

add_executable(messages_consumer main.cpp)
target_link_libraries(messages_consumer PRIVATE holoscan::messages)
set_target_properties(messages_consumer PROPERTIES CXX_EXTENSIONS OFF)

# Passes on exit status 0, which the consumer returns only when it prints PASS (a
# PASS_REGULAR_EXPRESSION would ignore the exit status).
add_test(NAME messages_installed_consumer COMMAND messages_consumer)
set_tests_properties(messages_installed_consumer PROPERTIES TIMEOUT 300)
