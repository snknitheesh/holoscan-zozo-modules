# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Entered when PKG_holoscan_normals is ON, as `holoscan package holoscan-normals`
# sets it. Writes ${CMAKE_BINARY_DIR}/pkg/CPackConfig-holoscan-normals.cmake, which
# packages everything the module installs.
include("${PROJECT_SOURCE_DIR}/cmake/holohub_configure_deb.cmake")

# "holoscan" is the Debian package of the SDK 5.0 on its platforms; "holoscan-messages" the
# messages module's, whose types the operators' ports carry: 1.x only, as the CMake package
# finds it (SameMajorVersion) and the library links its major version's soname. The CMake
# package finds CUDAToolkit, and the public headers include the CUDA runtime's, so the
# runtime's development files are needed too.
holohub_configure_deb(
  NAME "holoscan-normals"
  DESCRIPTION "${PROJECT_DESCRIPTION}"
  VERSION "${PROJECT_VERSION}"
  VENDOR "zozo"
  CONTACT "zozo <snknitheesh@gmail.com>"
  DEPENDS
    "holoscan (>= 5.0.0), holoscan-messages (>= 1.0.0), holoscan-messages (<< 2), libcudart.so.13-dev"
)
