// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A stereo calibration's two camera_info files, written as ROS's
// camera_calibration writes them, from a raw pair with known extrinsics.

#pragma once

#include <filesystem>

#include <holoscan/messages/camera_info.hpp>

namespace messages_demo {

// Writes `pair`'s left and right camera_info files: K and D of each camera,
// its rectification as R, and P of the rectified cameras, with focal length
// `f` and principal point (cx, cy), the right one's Tx = -f * baseline. As
// the calibrator does, K and P have 5 decimals, D 6 and R 8, a matrix row per
// line.
// @throws std::runtime_error for a file that cannot be written.
void write_ros_stereo_files(const holoscan::messages::StereoCameraInfo& pair, double f, double cx,
                            double cy, const std::filesystem::path& left,
                            const std::filesystem::path& right);

// `camera` as its file holds it: K and D rounded to the decimals written.
// (R, rounded to 8 decimals, reads back as the rotation nearest to it, which
// is the true one to about 1e-8.)
[[nodiscard]] holoscan::messages::CameraInfo as_written(holoscan::messages::CameraInfo camera);

}  // namespace messages_demo
