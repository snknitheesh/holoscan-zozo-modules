// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Between the plain values of camera_info.hpp and the generated message types
// that ports carry (schema::CameraInfoT, schema::StereoCameraInfoT). Both
// directions validate, so that no invalid calibration is sent or used.

#pragma once

#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/camera_info_schema_traits.hpp>
#include <holoscan/messages/stereo_camera_info_schema_traits.hpp>
#include <holoscan/schemas/header_schema_traits.hpp>

namespace holoscan::messages {

// @throws std::invalid_argument for a value validate() refuses.
[[nodiscard]] schema::CameraInfoT to_message(const CameraInfo& info);
[[nodiscard]] schema::StereoCameraInfoT to_message(const StereoCameraInfo& info);

// A binning of 0, which ROS uses for 1, reads as 1; an absent header as an
// empty one.
// @throws std::invalid_argument for a message without its cameras or
// extrinsics, or one validate() refuses.
[[nodiscard]] CameraInfo from_message(const schema::CameraInfoT& message);
[[nodiscard]] StereoCameraInfo from_message(const schema::StereoCameraInfoT& message);

[[nodiscard]] holoscan::schema::HeaderT to_message(const Header& header);
[[nodiscard]] Header from_message(const holoscan::schema::HeaderT& header);

}  // namespace holoscan::messages
