# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Entered when PKG_holoscan_point_cloud is ON, as `holoscan package holoscan-point-cloud`
# sets it. Writes ${CMAKE_BINARY_DIR}/pkg/CPackConfig-holoscan-point-cloud.cmake, which
# packages everything the module installs.
include("${PROJECT_SOURCE_DIR}/cmake/holohub_configure_deb.cmake")

# "holoscan" is the Debian package of the SDK 5.0 on its platforms; "holoscan-messages" the
# messages module's, whose types the operators' ports carry: 1.x only, as the CMake package
# finds it (SameMajorVersion) and the library links its major version's soname.
holohub_configure_deb(
  NAME "holoscan-point-cloud"
  DESCRIPTION "${PROJECT_DESCRIPTION}"
  VERSION "${PROJECT_VERSION}"
  VENDOR "zozo"
  CONTACT "zozo <snknitheesh@gmail.com>"
  DEPENDS "holoscan (>= 5.0.0), holoscan-messages (>= 1.0.0), holoscan-messages (<< 2)"
)
