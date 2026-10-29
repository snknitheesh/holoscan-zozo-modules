# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Entered when PKG_holoscan_stereo_camera is ON, as `holoscan package holoscan-stereo-camera`
# sets it. Writes ${CMAKE_BINARY_DIR}/pkg/CPackConfig-holoscan-stereo-camera.cmake, which
# packages everything the module installs.
include("${PROJECT_SOURCE_DIR}/cmake/holohub_configure_deb.cmake")

# "holoscan" is the Debian package of the SDK 5.0 on its platforms. A build with the realsense
# backend also needs librealsense.
set(_depends "holoscan (>= 5.0.0)")
if(HOLOSCAN_STEREO_CAMERA_REALSENSE)
  string(APPEND _depends ", librealsense2 (>= 2.50)")
endif()
holohub_configure_deb(
  NAME "holoscan-stereo-camera"
  DESCRIPTION "${PROJECT_DESCRIPTION}"
  VERSION "${PROJECT_VERSION}"
  VENDOR "zozo"
  CONTACT "zozo <snknitheesh@gmail.com>"
  DEPENDS "${_depends}"
)
