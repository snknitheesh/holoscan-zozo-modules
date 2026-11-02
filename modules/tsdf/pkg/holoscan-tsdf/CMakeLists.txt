# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Entered when PKG_holoscan_tsdf is ON, as `holoscan package holoscan-tsdf`
# sets it. Writes ${CMAKE_BINARY_DIR}/pkg/CPackConfig-holoscan-tsdf.cmake, which
# packages everything the module installs.
include("${PROJECT_SOURCE_DIR}/cmake/holohub_configure_deb.cmake")

# "holoscan" is the Debian package of the SDK 5.0 on its platforms.
holohub_configure_deb(
  NAME "holoscan-tsdf"
  DESCRIPTION "${PROJECT_DESCRIPTION}"
  VERSION "${PROJECT_VERSION}"
  VENDOR "zozo"
  CONTACT "zozo <snknitheesh@gmail.com>"
  DEPENDS "holoscan (>= 5.0.0)"
)
