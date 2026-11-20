# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on CMAKE_PREFIX_PATH
# (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_fiducials_consumer LANGUAGES CXX)
enable_testing()

# The overlay component when the package has it, to exercise FiducialOverlayOp too.
find_package(holoscan-fiducials 1.0 REQUIRED CONFIG OPTIONAL_COMPONENTS overlay)
message(
  STATUS
  "Found holoscan-fiducials ${holoscan-fiducials_VERSION} in ${holoscan-fiducials_DIR}"
)

add_executable(fiducials_consumer main.cpp)
target_link_libraries(fiducials_consumer PRIVATE holoscan::fiducials)
set_target_properties(fiducials_consumer PROPERTIES CXX_EXTENSIONS OFF)
if(TARGET holoscan::fiducials_overlay)
  target_link_libraries(fiducials_consumer PRIVATE holoscan::fiducials_overlay)
  target_compile_definitions(fiducials_consumer PRIVATE FIDUCIALS_CONSUMER_WITH_OVERLAY=1)
else()
  target_compile_definitions(fiducials_consumer PRIVATE FIDUCIALS_CONSUMER_WITH_OVERLAY=0)
endif()

# Passes on exit status 0, which the consumer returns only when it prints PASS (a
# PASS_REGULAR_EXPRESSION would ignore the exit status), and only on one worker.
add_test(NAME fiducials_installed_consumer COMMAND fiducials_consumer)
set_tests_properties(fiducials_installed_consumer PROPERTIES LABELS "fiducials;gpu" TIMEOUT 300)
