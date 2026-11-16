# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Entered when PKG_holoscan_messages is ON, as `holoscan package holoscan-messages`
# sets it. Writes ${CMAKE_BINARY_DIR}/pkg/CPackConfig-holoscan-messages.cmake, which
# packages everything the module installs.
include("${PROJECT_SOURCE_DIR}/cmake/holohub_configure_deb.cmake")

# "holoscan" is the Debian package of the SDK 5.0 on its platforms. The library itself needs
# only the SDK's (it links no CUDA library at run time), but its CMake package finds the CUDA
# toolkit and links CUDA::cudart, and its headers use CUDA's: "libcudart.so.13-dev", which the
# CUDA 13 runtime's development package provides on every platform, as the SDK's own package
# names it.
holohub_configure_deb(
  NAME "holoscan-messages"
  DESCRIPTION "${PROJECT_DESCRIPTION}"
  VERSION "${PROJECT_VERSION}"
  VENDOR "zozo"
  CONTACT "zozo <snknitheesh@gmail.com>"
  DEPENDS "holoscan (>= 5.0.0), libcudart.so.13-dev"
)
