# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Entered when PKG_holoscan_foundation_stereo is ON, as `holoscan package
# holoscan-foundation-stereo` sets it. Writes
# ${CMAKE_BINARY_DIR}/pkg/CPackConfig-holoscan-foundation-stereo.cmake, which packages everything
# the module installs. It contains no model.
include("${PROJECT_SOURCE_DIR}/cmake/holohub_configure_deb.cmake")

# "holoscan" is the Debian package of the SDK 5.0 on its platforms; it carries HoloInfer.
holohub_configure_deb(
  NAME "holoscan-foundation-stereo"
  DESCRIPTION "${PROJECT_DESCRIPTION}"
  VERSION "${PROJECT_VERSION}"
  VENDOR "zozo"
  CONTACT "zozo <snknitheesh@gmail.com>"
  DEPENDS "holoscan (>= 5.0.0)"
)
