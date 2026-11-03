// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The one calibration file format: the ROS camera_info YAML file, as ROS's
// camera_calibration writes it and camera_calibration_parsers reads it.
//
//   image_width: 640
//   image_height: 480
//   camera_name: left
//   camera_matrix: {rows: 3, cols: 3, data: [fx, skew, cx, 0, fy, cy, 0, 0, 1]}
//   distortion_model: plumb_bob
//   distortion_coefficients: {rows: 1, cols: 5, data: [k1, k2, p1, p2, k3]}
//   rectification_matrix: {rows: 3, cols: 3, data: [R, row-major]}
//   projection_matrix: {rows: 3, cols: 4, data: [fx', s', cx', Tx, 0, fy', cy', Ty, 0, 0, 1, 0]}
//
// (block or flow style alike). image_width, image_height and camera_matrix
// are required; the others are optional, and any other key is refused, so
// that a misspelt key cannot silently drop a matrix.
//
// ROS leaves implicit what the file's images are; reading makes it explicit:
// - camera_name is the header's frame_id.
// - K and D describe the camera; a camera whose D has a coefficient that is
//   not zero is kRaw, else kUndistorted (a distortion_model of "none" or ""
//   with no coefficients is kNone).
// - R, when present and not the identity, is the camera's rectification.
// - P is read only to recover a stereo pair's extrinsics (below); a camera's
//   own images are described by K.
// Writing is the inverse: R is the rectification or the identity, and P is
// [K | 0] for a single camera.
//
// Values are written with the fewest digits that read back as the same
// double, so write then read gives back the same CameraInfo (a kRectified
// camera reads back as kUndistorted on its own: one file cannot say it
// belongs to a rectified pair; read it as a pair).

#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

#include "holoscan/messages/camera_info.hpp"

namespace holoscan::messages {

// The largest camera_info file read, far above any real one.
inline constexpr std::uintmax_t kMaxCameraInfoFileBytes = 1U << 20;

// Reads and validates one camera_info file.
// @throws std::invalid_argument starting with the path, naming the line or
// key that is wrong, for a file that cannot be read or is not valid.
[[nodiscard]] CameraInfo read_camera_info_yaml(const std::filesystem::path& path);
// The same, from the file's text.
[[nodiscard]] CameraInfo parse_camera_info_yaml(std::string_view text);

// Writes `info` (validated first) as a camera_info file, replacing `path`.
// @throws std::invalid_argument for an invalid `info` or a file that cannot be written.
void write_camera_info_yaml(const CameraInfo& info, const std::filesystem::path& path);
[[nodiscard]] std::string format_camera_info_yaml(const CameraInfo& info);

// Reads a stereo pair from its two camera_info files, as a ROS stereo
// calibration writes them: K and D of each raw camera, R rotating each into
// the rectified frame, and P of each rectified camera, the right one's with
// Tx = -fx' * baseline. The extrinsics are recovered from them:
//   right_from_left.rotation = R_right^T * R_left
//   right_from_left.translation = R_right^T * (Tx_r / fx'_r - Tx_l / fx'_l,
//                                              Ty_r / fy'_r - Ty_l / fy'_l, 0)
// which for a usual file (left Tx = 0) is -R_right^T * (baseline, 0, 0).
// A pair whose files describe rectified images (no distortion, R the
// identity, P's left 3x3 equal to K, one fx, fy and cy, Ty = 0) is a
// kRectified pair, with baseline = -Tx_r / fx_r and doffs = cx_r - cx_l.
// @throws std::invalid_argument naming the file and what is wrong, including
// a right projection without a baseline.
[[nodiscard]] StereoCameraInfo read_stereo_camera_info_yaml(
    const std::filesystem::path& left_path, const std::filesystem::path& right_path);
[[nodiscard]] StereoCameraInfo parse_stereo_camera_info_yaml(std::string_view left_text,
                                                             std::string_view right_text);

// Writes a stereo pair (validated first) as its two camera_info files, in
// the form read_stereo_camera_info_yaml() reads. The ROS format holds the
// extrinsics only as R and P, so a pair that is not rectified needs both
// cameras' rectifications (an absent one is the identity) to agree with
// right_from_left (see validate()), and its P are written with the left
// camera's fy as fx' and fy' and its principal point.
// @throws std::invalid_argument for a pair that is not valid or cannot be written.
void write_stereo_camera_info_yaml(const StereoCameraInfo& info,
                                   const std::filesystem::path& left_path,
                                   const std::filesystem::path& right_path);
// The left and right files' text.
[[nodiscard]] std::pair<std::string, std::string> format_stereo_camera_info_yaml(
    const StereoCameraInfo& info);

}  // namespace holoscan::messages
