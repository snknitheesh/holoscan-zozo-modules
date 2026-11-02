# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Entered when PKG_holoscan_codec is ON, as `holoscan package holoscan-codec` sets it. Writes
# ${CMAKE_BINARY_DIR}/pkg/CPackConfig-holoscan-codec.cmake, which packages everything the module
# installs. The codec libraries come with the NVIDIA driver (and VPI with JetPack), so the package
# depends on the SDK only.
include("${PROJECT_SOURCE_DIR}/cmake/holohub_configure_deb.cmake")

holohub_configure_deb(
  NAME "holoscan-codec"
  DESCRIPTION "${PROJECT_DESCRIPTION}"
  VERSION "${PROJECT_VERSION}"
  VENDOR "zozo"
  CONTACT "zozo <snknitheesh@gmail.com>"
  DEPENDS "holoscan (>= 5.0.0)"
)
