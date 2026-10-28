# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# stereo_camera_record: records a camera through the module's backends. Part of the module's
# build, linked to its library; or a project of its own, which needs no Holoscan SDK, for a
# machine with the camera and librealsense but no SDK:
#
#   cmake -S modules/stereo-camera/tools -B build/recorder -DCMAKE_BUILD_TYPE=Release
cmake_minimum_required(VERSION 4.0)
project(holoscan_stereo_camera_tools LANGUAGES CXX)

add_executable(stereo_camera_record stereo_camera_record.cpp)
target_compile_features(stereo_camera_record PRIVATE cxx_std_20)
if(TARGET holoscan_stereo_camera)
  target_link_libraries(stereo_camera_record PRIVATE holoscan_stereo_camera)
else()
  # On its own: the module's SDK-free sources, with the realsense backend when librealsense is
  # installed.
  set(_module "${CMAKE_CURRENT_SOURCE_DIR}/..")
  find_package(realsense2 2.50 QUIET CONFIG)
  target_sources(
    stereo_camera_record
    PRIVATE
      "${_module}/src/calibration.cpp"
      "${_module}/src/image.cpp"
      "${_module}/src/json.cpp"
      "${_module}/src/netpbm.cpp"
      "${_module}/src/playback_source.cpp"
      "${_module}/src/recording.cpp"
      "${_module}/src/side_by_side.cpp"
      "${_module}/src/source.cpp"
      "${_module}/src/synthetic.cpp"
      "${_module}/src/v4l2_source.cpp"
  )
  target_include_directories(stereo_camera_record PRIVATE "${_module}/include" "${_module}/src")
  target_compile_definitions(
    stereo_camera_record
    PRIVATE HOLOSCAN_STEREO_CAMERA_WITH_REALSENSE=$<BOOL:${realsense2_FOUND}>
  )
  if(realsense2_FOUND)
    target_sources(stereo_camera_record PRIVATE "${_module}/src/realsense_source.cpp")
    target_link_libraries(stereo_camera_record PRIVATE realsense2::realsense2)
    message(STATUS "stereo_camera_record: with librealsense ${realsense2_VERSION}")
  else()
    message(STATUS "stereo_camera_record: librealsense not found; v4l2-sbs only")
  endif()
endif()
