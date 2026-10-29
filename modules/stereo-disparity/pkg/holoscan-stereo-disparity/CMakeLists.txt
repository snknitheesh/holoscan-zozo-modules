# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Entered when PKG_holoscan_stereo_disparity is ON, as `holoscan package
# holoscan-stereo-disparity` sets it. Writes
# ${CMAKE_BINARY_DIR}/pkg/CPackConfig-holoscan-stereo-disparity.cmake, which packages
# everything the module installs.
include("${PROJECT_SOURCE_DIR}/cmake/holohub_configure_deb.cmake")

# "holoscan" is the Debian package of the SDK 5.0 on its platforms; libnvvpi4 is VPI 4's
# runtime, from NVIDIA's Jetson package repository.
holohub_configure_deb(
  NAME "holoscan-stereo-disparity"
  DESCRIPTION "${PROJECT_DESCRIPTION}"
  VERSION "${PROJECT_VERSION}"
  VENDOR "zozo"
  CONTACT "zozo <snknitheesh@gmail.com>"
  DEPENDS "holoscan (>= 5.0.0), libnvvpi4 (>= 4.0)"
)
