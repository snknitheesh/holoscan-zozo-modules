# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Entered when PKG_holoscan_icp is ON, as `holoscan package holoscan-icp`
# sets it. Writes ${CMAKE_BINARY_DIR}/pkg/CPackConfig-holoscan-icp.cmake, which
# packages everything the module installs.
include("${PROJECT_SOURCE_DIR}/cmake/holohub_configure_deb.cmake")

# "holoscan" is the Debian package of the SDK 5.0 on its platforms; "holoscan-messages" the
# messages module's, whose PointCloud the operator's ports carry: 1.x only, as the CMake
# package finds it (SameMajorVersion) and the library links its major version's soname. The
# library links the CUDA runtime (NEEDED libcudart.so.13), and its CMake package finds the
# CUDA toolkit and links CUDA::cudart: "libcudart.so.13-dev", which the CUDA 13 runtime's
# development package provides on every platform, as the SDK's own package names it.
holohub_configure_deb(
  NAME "holoscan-icp"
  DESCRIPTION "${PROJECT_DESCRIPTION}"
  VERSION "${PROJECT_VERSION}"
  VENDOR "zozo"
  CONTACT "zozo <snknitheesh@gmail.com>"
  DEPENDS
    "holoscan (>= 5.0.0), holoscan-messages (>= 1.0.0), holoscan-messages (<< 2), libcudart.so.13-dev"
)
