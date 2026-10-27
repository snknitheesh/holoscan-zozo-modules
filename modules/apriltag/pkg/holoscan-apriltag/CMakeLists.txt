# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Entered when PKG_holoscan_apriltag is ON, as `holoscan package holoscan-apriltag` sets
# it. Writes ${CMAKE_BINARY_DIR}/pkg/CPackConfig-holoscan-apriltag.cmake, which packages
# everything the module installs.
include("${PROJECT_SOURCE_DIR}/cmake/holohub_configure_deb.cmake")

# "holoscan" is the Debian package of the SDK 5.0 on its platforms.
set(_depends "holoscan (>= 5.0.0)")
if(HOLOSCAN_APRILTAG_VPI)
  string(APPEND _depends ", libnvvpi4 (>= 4.0)")
endif()

holohub_configure_deb(
  NAME "holoscan-apriltag"
  DESCRIPTION "${PROJECT_DESCRIPTION}"
  VERSION "${PROJECT_VERSION}"
  VENDOR "zozo"
  CONTACT "zozo <snknitheesh@gmail.com>"
  DEPENDS "${_depends}")
