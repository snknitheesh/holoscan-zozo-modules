# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
#
# The SDK-free part of holoscan-messages as a static library, messages_host_core: the plain
# messages and their checks, the camera_info YAML files, the point cloud layout and the
# synthetic calibration, which need no Holoscan SDK, CUDA or GPU. For the host tests
# (tests/host/) of this module and of the modules that depend on it: include() this file
# from the messages folder of the same checkout, then link messages_host_core.
include_guard(GLOBAL)

set(_messages_root "${CMAKE_CURRENT_LIST_DIR}/..")
add_library(
  messages_host_core
  STATIC
  "${_messages_root}/src/camera_info.cpp"
  "${_messages_root}/src/camera_info_yaml.cpp"
  "${_messages_root}/src/point_cloud_layout.cpp"
  "${_messages_root}/src/synthetic_calibration.cpp"
  "${_messages_root}/src/yaml_reader.cpp"
)
target_include_directories(
  messages_host_core
  PUBLIC "${_messages_root}/include"
  PRIVATE "${_messages_root}/src"
)
target_compile_features(messages_host_core PUBLIC cxx_std_20)
set_target_properties(messages_host_core PROPERTIES CXX_EXTENSIONS OFF)
unset(_messages_root)
