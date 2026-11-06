# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, CUDA and a GPU, except the SDK-free ones in host/. The
# real-data test also needs the local recordings: it runs when HOLOSCAN_TEST_DATA names
# the folder that holds them, and is skipped otherwise, as in CI.

add_executable(depth_report_op_test report_op_test.cpp)
target_link_libraries(depth_report_op_test PRIVATE depth_to_point_cloud_ops holoscan::testing)
add_test(
  NAME depth_report_op
  COMMAND depth_report_op_test "${CMAKE_CURRENT_BINARY_DIR}/report_op_files"
)

add_executable(depth_write_test_images write_test_images.cpp)
target_link_libraries(depth_write_test_images PRIVATE depth_to_point_cloud_ops)

add_library(depth_spawn_thread_preload SHARED spawn_thread_preload.cpp)

# Runs the demo through run_and_match.cmake, which checks its exit status as
# well as its output: depth_app_test(<name> <exit status> <output regex>
# <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(depth_app_test name status output)
  add_test(
    NAME ${name}
    COMMAND
      ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}" -P "${_run_and_match}" --
      $<TARGET_FILE:depth_to_point_cloud> ${ARGN}
  )
endfunction()

# The demo on the synthetic camera, every point and color checked against ground truth:
# float32 and uint16 depth, from GPU and host memory, with the camera from a camera_info
# file or with every frame. The frames before the first camera info give no cloud; the
# demo reports how many (before_camera_info), at most one, and checks every cloud after
# them.
set(_pass "before_camera_info=[01] .*wrong_points=0 wrong_colors=0 .*verdict=PASS")
depth_app_test(depth_app_meters 0 "${_pass}" --count 20 --period-ms 5)
depth_app_test(depth_app_millimeters 0 "${_pass}"
  --count 20 --period-ms 5 --depth-format millimeters
)
depth_app_test(depth_app_host_input 0 "${_pass}" --count 20 --period-ms 5 --host-input true)
# With every frame, the camera's intrinsics change from frame to frame, and every frame
# gives a cloud.
depth_app_test(depth_app_camera_info_per_frame 0
  "frames=20/20 before_camera_info=0 .*wrong_points=0 wrong_colors=0 .*verdict=PASS"
  --count 20 --period-ms 5 --camera-info-per-frame true
)
# A run of one frame: the source sends it a period after the camera info, which
# CameraInfoSourceOp sends when the run starts, so it gives a cloud and ends the run.
depth_app_test(depth_app_one_frame 0
  "frames=1/1 before_camera_info=0 .*wrong_points=0 wrong_colors=0 .*verdict=PASS"
  --count 1 --period-ms 5 --timeout-seconds 30
)
depth_app_test(depth_app_without_color 0 "before_camera_info=[01] .*wrong_points=0 .*verdict=PASS"
  --count 20 --period-ms 5 --color false
)
# A narrower valid range reaches the operator: the truth applies the same one,
# and the reported depths stay inside it.
depth_app_test(depth_app_depth_range 0
  "before_camera_info=[01] .*depth_m=\\[1\\.200, 1\\.800\\] truth: wrong_points=0 wrong_colors=0 .*verdict=PASS"
  --count 20 --period-ms 5 --depth-min 1.2 --depth-max 1.8
)
# Only the plan's verdict: an accepted plan has no warnings. With CameraInfoSourceOp
# the graph has two sources and still runs on one worker, as the demo lays it out.
depth_app_test(depth_app_compile_only 0 "^depth_to_point_cloud: plan=OK workers=1\n*$"
  --compile-only true
)

# PLY output: one file per cloud, which the verdict checks.
set(_ply "${CMAKE_CURRENT_BINARY_DIR}/ply")
add_test(
  NAME depth_app_ply
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=ply_files=[23] .*verdict=PASS"
    "-DMAKE_DIRECTORY=${_ply}" -P "${_run_and_match}" -- $<TARGET_FILE:depth_to_point_cloud> --count
    3 --period-ms 5 --ply-dir "${_ply}"
)

# A folder of 16-bit PGM depth images written by the fixture, with their camera_info
# file. The depths are those of the synthetic frames 0..4 at 320x240 in millimeters,
# valid in 0.1..10 m: 73984 valid points per frame (or one frame less, when the first
# frame comes before the first camera info).
set(_images "${CMAKE_CURRENT_BINARY_DIR}/depth_images")
add_test(
  NAME depth_images_setup
  COMMAND $<TARGET_FILE:depth_write_test_images> "${_images}" 5 320 240
)
set_tests_properties(depth_images_setup PROPERTIES FIXTURES_SETUP depth_images)
set(_camera_info --camera-info "${_images}/camera_info.yaml")
depth_app_test(depth_app_images 0
  "(frames=5/5 before_camera_info=0 points=384000 valid=369920 depth_m=\\[1\\.040|frames=4/5 before_camera_info=1 points=307200 valid=295936 depth_m=\\[1\\.055), 1\\.956\\].*verdict=PASS"
  --source images --images "${_images}" ${_camera_info} --period-ms 5
)
# --loop starts over after the last image: 12 frames from 5 images.
depth_app_test(depth_app_images_loop 0
  "(frames=12/12 before_camera_info=0 points=921600 valid=887808|frames=11/12 before_camera_info=1 points=844800 valid=813824) depth_m=\\[1\\.040, 1\\.956\\].*verdict=PASS"
  --source images --images "${_images}" ${_camera_info} --period-ms 5
  --loop true --count 12
)
depth_app_test(depth_app_rejects_count_beyond_images 2 "^error: .*--loop true"
  --source images --images "${_images}" ${_camera_info} --count 6
)
# The camera: required, readable, for the images' size, and one the operator can model.
depth_app_test(depth_app_rejects_images_without_camera_info 2 "^error: .*--camera-info FILE"
  --source images --images "${_images}"
)
depth_app_test(depth_app_rejects_missing_camera_info 2 "^error: .*no_such_camera_info\\.yaml"
  --source images --images "${_images}" --camera-info "${_images}/no_such_camera_info.yaml"
)
depth_app_test(depth_app_rejects_camera_of_other_size 2
  "^error: .*the camera is for 160x120 images, the depth images are 320x240"
  --source images --images "${_images}" --camera-info "${_images}/other_size.yaml"
)
depth_app_test(depth_app_rejects_raw_camera 2 "^error: .*raw\\.yaml: the camera describes raw images"
  --source images --images "${_images}" --camera-info "${_images}/raw.yaml"
)
depth_app_test(depth_app_rejects_skewed_camera 2 "^error: .*skewed\\.yaml: the camera has a skew"
  --source images --images "${_images}" --camera-info "${_images}/skewed.yaml"
)
# Options that do not apply to the input are refused, not ignored.
depth_app_test(depth_app_rejects_verify_with_images 2 "^error: .*--verify apply to synthetic"
  --source images --images "${_images}" ${_camera_info} --verify true
)
depth_app_test(depth_app_rejects_size_with_images 2 "^error: .*--width, --height"
  --source images --images "${_images}" ${_camera_info} --width 320
)
depth_app_test(depth_app_rejects_zero_depth_scale 2 "^error: depth_scale"
  --source images --images "${_images}" ${_camera_info} --depth-scale 0
)
set_tests_properties(
  depth_app_images
  depth_app_images_loop
  depth_app_rejects_count_beyond_images
  depth_app_rejects_images_without_camera_info
  depth_app_rejects_missing_camera_info
  depth_app_rejects_camera_of_other_size
  depth_app_rejects_raw_camera
  depth_app_rejects_skewed_camera
  depth_app_rejects_verify_with_images
  depth_app_rejects_size_with_images
  depth_app_rejects_zero_depth_scale
  PROPERTIES FIXTURES_REQUIRED depth_images
)
# Invalid input fails fast with a usage error (exit status 2) instead of
# starting a run.
depth_app_test(depth_app_rejects_unknown_option 2 "^error: unknown option" --colour true)
depth_app_test(depth_app_rejects_zero_count 2 "^error: synthetic runs need --count > 0" --count 0)
depth_app_test(depth_app_rejects_loop_with_synthetic 2 "^error: --loop applies to --source images"
  --loop true
)
# An 8-bit PGM image needs its depth scale: the default, millimeters, would put every depth
# below 0.256 m.
set(_images_8bit "${CMAKE_CURRENT_BINARY_DIR}/depth_images_8bit")
file(MAKE_DIRECTORY "${_images_8bit}")
file(WRITE "${_images_8bit}/depth_0000.pgm" "P5\n4 2\n255\nAAAAAAAA")
depth_app_test(depth_app_rejects_8bit_without_depth_scale 2
  "^error: .*depth_0000\\.pgm: 8-bit depth images need --depth-scale"
  --source images --images "${_images_8bit}" --camera-info "${_images}/camera_info.yaml"
)
set_tests_properties(
  depth_app_rejects_8bit_without_depth_scale
  PROPERTIES FIXTURES_REQUIRED depth_images
)
depth_app_test(depth_app_rejects_camera_info_for_synthetic 2 "^error: .*drop --camera-info"
  --camera-info "${CMAKE_CURRENT_BINARY_DIR}/camera_info.yaml"
)
depth_app_test(depth_app_rejects_zero_width 2 "^error: synthetic depth images" --width 0)
depth_app_test(depth_app_rejects_inverted_range 2 "^error: depth range"
  --depth-min 5 --depth-max 1
)
depth_app_test(depth_app_rejects_missing_images 2 "^error: .*cannot list"
  --source images --images "${CMAKE_CURRENT_BINARY_DIR}/no_such_folder" --camera-info
  "${CMAKE_CURRENT_BINARY_DIR}/camera_info.yaml"
)
set(_no_images "${CMAKE_CURRENT_BINARY_DIR}/no_images")
add_test(
  NAME depth_app_rejects_empty_images
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=2 "-DEXPECT_OUTPUT=^error: .*no \\.pgm images"
    "-DMAKE_DIRECTORY=${_no_images}" -P "${_run_and_match}" -- $<TARGET_FILE:depth_to_point_cloud>
    --source images --images "${_no_images}" --camera-info
    "${CMAKE_CURRENT_BINARY_DIR}/camera_info.yaml"
)

# A failed run reports its cause and exits 1. The second frame's file is blocked: it is
# written whether or not the first frame came before the first camera info.
set(_blocked "${CMAKE_CURRENT_BINARY_DIR}/blocked_ply")
add_test(
  NAME depth_app_failed_run
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=1 "-DEXPECT_OUTPUT=frame_000001.ply.*verdict=FAIL"
    "-DMAKE_DIRECTORY=${_blocked}/frame_000001.ply" -P "${_run_and_match}" --
    $<TARGET_FILE:depth_to_point_cloud> --count 5 --period-ms 5 --ply-dir "${_blocked}"
)

# A run that outlives its deadline is stopped and fails: 100 frames of
# 4096x4096, each checked point by point, take far longer than the one second
# allowed beyond the 0.2 s the frames need.
depth_app_test(depth_app_timed_out 1 "timed out.*frames=[0-9]+/100 .*verdict=FAIL"
  --width 4096 --height 4096 --count 100 --period-ms 1 --timeout-seconds 1
)

# A stop request ends the run early with its own verdict, and the warning it
# leaves is shown. The signal goes out once the first cloud is written, so the
# run has started. The test sends SIGTERM, as a shell starts background jobs
# with SIGINT ignored, and its script has no semicolons, where CMake would
# split it.
set(_stopped "${CMAKE_CURRENT_BINARY_DIR}/stopped_ply")
add_test(
  NAME depth_app_stopped
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=1
    "-DEXPECT_OUTPUT=warning RUNTIME_MANAGED_SIGNAL_RECEIVED.*frames=[0-9]+/100000 .*verdict=STOPPED"
    -P "${_run_and_match}" -- sh -c
    "rm -rf \"$2\" && mkdir -p \"$2\" || exit 1
     \"$1\" --count 100000 --period-ms 10 --width 64 --height 48 --ply-dir \"$2\" &
     for i in $(seq 600)
     do
       [ -e \"$2/frame_000000.ply\" ] || [ -e \"$2/frame_000001.ply\" ] && break
       sleep 0.05
     done
     kill -TERM $! && wait $!"
    sh $<TARGET_FILE:depth_to_point_cloud> "${_stopped}"
)

# Another thread running before main(), as under a profiler or a sanitizer:
# the run goes ahead without managed signal handling, and says so.
add_test(
  NAME depth_app_without_signal_domain
  COMMAND
    ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS" -P "${_run_and_match}"
    -- ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:depth_spawn_thread_preload>
    $<TARGET_FILE:depth_to_point_cloud> --count 10 --period-ms 5
)

# The local recordings: each clip's depth images with its depth.yaml, through the demo.
# Their depth is in units of 0.452 mm.
add_test(
  NAME depth_app_real_recordings
  COMMAND
    ${CMAKE_COMMAND} "-DDEMO=$<TARGET_FILE:depth_to_point_cloud>" -DDEPTH_SCALE=0.000452 -P
    "${CMAKE_CURRENT_SOURCE_DIR}/real_recordings.cmake"
)
set_tests_properties(depth_app_real_recordings PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED: ")

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "depth_to_point_cloud;gpu" TIMEOUT 300)

# The SDK-free tests, which the hosted CI job also builds on their own.
add_subdirectory(host)
