# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Entered when PKG_holoscan_yolo26 is ON, as `holoscan package holoscan-yolo26` sets it.
# Writes ${CMAKE_BINARY_DIR}/pkg/CPackConfig-holoscan-yolo26.cmake, which packages
# everything the module installs.
include("${PROJECT_SOURCE_DIR}/cmake/holohub_configure_deb.cmake")

# "holoscan" is the Debian package of the SDK 5.0 on its platforms; it carries HoloInfer.
holohub_configure_deb(
  NAME "holoscan-yolo26"
  DESCRIPTION "${PROJECT_DESCRIPTION}"
  VERSION "${PROJECT_VERSION}"
  VENDOR "zozo"
  CONTACT "zozo <snknitheesh@gmail.com>"
  DEPENDS "holoscan (>= 5.0.0)"
)
