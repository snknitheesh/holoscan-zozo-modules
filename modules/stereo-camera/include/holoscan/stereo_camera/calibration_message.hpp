// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The calibration as StereoCameraOp sends it: the message generated from
// schemas/stereo_calibration.fbs, and the conversions to and from
// StereoCalibration.

#pragma once

#include <holoscan/stereo_camera/calibration.hpp>
#include <holoscan/stereo_camera/stereo_calibration_schema_traits.hpp>

namespace holoscan::stereo_camera {

// The message for `calibration`; frame, timestamp_ns, device_sequence and
// backend are left for the caller. The rgb and depth cameras are included when
// the calibration has them.
[[nodiscard]] schema::StereoCalibrationT to_message(const StereoCalibration& calibration);

// The calibration a message holds, validated; camera settings are not in the
// message and stay empty.
// @throws std::invalid_argument naming the first unusable field.
[[nodiscard]] StereoCalibration from_message(const schema::StereoCalibrationT& message);

}  // namespace holoscan::stereo_camera
