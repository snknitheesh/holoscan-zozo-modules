// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StereoGeometry as StereoDisparityOp's `geometry` input takes it: the message
// generated from schemas/stereo_geometry.fbs, and the conversions to and from
// StereoGeometry.

#pragma once

#include <holoscan/stereo_disparity/stereo_geometry.hpp>
#include <holoscan/stereo_disparity/stereo_geometry_schema_traits.hpp>

namespace holoscan::stereo_disparity {

[[nodiscard]] schema::StereoGeometryT to_message(const StereoGeometry& geometry);

// The geometry a message holds, validated.
// @throws std::invalid_argument naming the first unusable field.
[[nodiscard]] StereoGeometry from_message(const schema::StereoGeometryT& message);

}  // namespace holoscan::stereo_disparity
