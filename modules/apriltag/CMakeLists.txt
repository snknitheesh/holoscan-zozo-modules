# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# `holoscan package` sets PKG_holoscan_apriltag; a package is built for every major GPU
# architecture, a development build for the GPU it runs on, unless the caller chose.
option(PKG_holoscan_apriltag "Generate the Debian package configuration" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(PKG_holoscan_apriltag)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures to compile")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
  endif()
endif()

project(
  holoscan_apriltag
  VERSION 1.0.0
  DESCRIPTION "AprilTag detection on the CPU, the GPU or the PVA, and its HoloViz overlay"
  LANGUAGES C CXX CUDA
)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTING "Build the module's tests" OFF)
option(HOLOSCAN_APRILTAG_WITH_VPI "Build the vpi backend when VPI 4 is installed" ON)
option(HOLOSCAN_APRILTAG_BUILD_OVERLAY "Build AprilTagOverlayOp (needs Holoscan Visualization)" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_holoscan_components core sensor_io)
if(HOLOSCAN_APRILTAG_BUILD_OVERLAY)
  list(APPEND _holoscan_components visualization)
endif()
if(BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# AprilTag 3 (BSD-2-Clause): an installed copy if there is one, otherwise the pinned
# release, checked against its SHA-256 and linked into the library.
find_package(apriltag 3.4 QUIET CONFIG)
if(apriltag_FOUND)
  set(_apriltag_target apriltag::apriltag)
  message(STATUS "holoscan-apriltag: using AprilTag ${apriltag_VERSION} from ${apriltag_DIR}")
else()
  include(FetchContent)
  # The library's options apply to it alone, and its tests stay out of this suite.
  block(PROPAGATE apriltag_SOURCE_DIR)
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(BUILD_EXAMPLES OFF)
    set(BUILD_PYTHON_WRAPPER OFF)
    set(BUILD_TESTING OFF)
    FetchContent_Declare(
      apriltag
      URL https://github.com/AprilRobotics/apriltag/archive/refs/tags/v3.4.5.tar.gz
      URL_HASH SHA256=3749f48220db78c29c375a373861f4e1f4138e5dbae3c298584e96cfcc556515
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(apriltag)
  endblock()
  set(_apriltag_target apriltag)
  message(STATUS "holoscan-apriltag: building AprilTag 3.4.5 into the library")
  # The library carries AprilTag's code, so its BSD-2-Clause notice travels with it.
  install(
    FILES "${apriltag_SOURCE_DIR}/LICENSE.md"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-apriltag"
    RENAME apriltag-LICENSE.md
  )
endif()

set(HOLOSCAN_APRILTAG_VPI OFF)
if(HOLOSCAN_APRILTAG_WITH_VPI)
  find_package(vpi 4 QUIET CONFIG PATHS /opt/nvidia/vpi4)
  if(vpi_FOUND)
    set(HOLOSCAN_APRILTAG_VPI ON)
    message(STATUS "holoscan-apriltag: building the vpi backend with VPI ${vpi_VERSION}")
  else()
    message(STATUS "holoscan-apriltag: VPI 4 not found; building without the vpi backend")
  endif()
endif()

# The detections message, generated from its FlatBuffers schema into the public
# include tree.
set(_generated_include_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
holoscan_add_flatbuffer_schema(
  TARGET holoscan_apriltag_schema
  SCHEMA schemas/apriltag_detections.fbs
  OUTPUT_DIR "${_generated_include_dir}/holoscan/apriltag"
  MAX_SERIALIZED_SIZE 16384
)
get_target_property(
  _generated_headers
  holoscan_apriltag_schema
  HOLOSCAN_FLATBUFFER_GENERATED_HEADERS
)

# holoscan::apriltag: the detector backends, pose estimation, the synthetic scenes and
# AprilTagDetectorOp.
add_library(
  holoscan_apriltag
  src/apriltag_detector_op.cpp
  src/cpu_backend.cpp
  src/cuda_backend.cu
  src/detector_backend.cpp
  src/gpu_segmenter.cu
  src/gray_converter.cu
  src/pose.cpp
  src/synthetic_scene.cpp
  src/tag_family.cpp
)
add_library(holoscan::apriltag ALIAS holoscan_apriltag)
# No FMA contraction in the quad fit and decoder, so they round as on a CPU.
set_source_files_properties(src/cuda_backend.cu PROPERTIES COMPILE_OPTIONS --fmad=false)
target_compile_features(holoscan_apriltag PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_include_directories(
  holoscan_apriltag
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${_generated_include_dir}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(
  holoscan_apriltag
  PUBLIC
    holoscan::core
    holoscan::sensor_io
    CUDA::cudart
    "$<BUILD_INTERFACE:holoscan_apriltag_schema>"
  PRIVATE ${_apriltag_target}
)
target_compile_definitions(
  holoscan_apriltag
  PRIVATE HOLOSCAN_APRILTAG_WITH_VPI=$<BOOL:${HOLOSCAN_APRILTAG_VPI}>
)
if(HOLOSCAN_APRILTAG_VPI)
  target_sources(holoscan_apriltag PRIVATE src/vpi_backend.cpp)
  target_link_libraries(holoscan_apriltag PRIVATE vpi)
endif()
if(NOT apriltag_FOUND)
  # Keep the bundled AprilTag's symbols inside the library, so a consumer that links
  # its own copy cannot mix the two.
  target_link_options(holoscan_apriltag PRIVATE "LINKER:--exclude-libs,ALL")
endif()
set_target_properties(
  holoscan_apriltag
  PROPERTIES
    EXPORT_NAME apriltag
    OUTPUT_NAME holoscan_apriltag
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
set(_targets holoscan_apriltag)

# holoscan::apriltag_overlay: AprilTagOverlayOp, which draws detections as HoloViz layers.
if(HOLOSCAN_APRILTAG_BUILD_OVERLAY)
  add_library(holoscan_apriltag_overlay src/apriltag_overlay_op.cpp src/image_output.cpp)
  add_library(holoscan::apriltag_overlay ALIAS holoscan_apriltag_overlay)
  target_compile_features(holoscan_apriltag_overlay PUBLIC cxx_std_20)
  target_include_directories(holoscan_apriltag_overlay PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(
    holoscan_apriltag_overlay
    PUBLIC holoscan_apriltag
    PRIVATE holoscan::visualization::op_holoviz
  )
  set_target_properties(
    holoscan_apriltag_overlay
    PROPERTIES
      EXPORT_NAME apriltag_overlay
      OUTPUT_NAME holoscan_apriltag_overlay
      POSITION_INDEPENDENT_CODE ON
      VERSION ${PROJECT_VERSION}
      SOVERSION ${PROJECT_VERSION_MAJOR}
  )
  list(APPEND _targets holoscan_apriltag_overlay)
endif()

# Installation: libraries, public and generated headers, the schema, and the package.
set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan-apriltag")
install(
  TARGETS ${_targets}
  EXPORT holoscan-apriltag-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/holoscan/apriltag DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan")
install(FILES ${_generated_headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/apriltag")
install(
  FILES schemas/apriltag_detections.fbs
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/holoscan-apriltag/schemas"
)
install(
  FILES ../../LICENSE NOTICE README.md
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/holoscan-apriltag"
)
install(
  EXPORT holoscan-apriltag-targets
  FILE holoscan-apriltag-targets.cmake
  NAMESPACE holoscan::
  DESTINATION "${_cmake_dir}"
)
set(HOLOSCAN_APRILTAG_HOLOSCAN_COMPONENTS core sensor_io)
if(HOLOSCAN_APRILTAG_BUILD_OVERLAY)
  list(APPEND HOLOSCAN_APRILTAG_HOLOSCAN_COMPONENTS visualization)
endif()
configure_package_config_file(
  cmake/holoscan-apriltag-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-apriltag-config.cmake"
  INSTALL_DESTINATION "${_cmake_dir}"
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/holoscan-apriltag-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-apriltag-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-apriltag-config-version.cmake"
  DESTINATION "${_cmake_dir}"
)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PKG_holoscan_apriltag)
  add_subdirectory(pkg/holoscan-apriltag)
endif()
