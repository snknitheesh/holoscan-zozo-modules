// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Each backend's factory, in its own source file; create_source() validates the
// configuration first. The RealSense one throws in a build without librealsense.

#pragma once

#include <memory>

#include <holoscan/stereo_camera/source.hpp>

namespace holoscan::stereo_camera::detail {

[[nodiscard]] std::unique_ptr<StereoSource> create_playback_source(const SourceConfig& config);
[[nodiscard]] std::unique_ptr<StereoSource> create_v4l2_source(const SourceConfig& config);
[[nodiscard]] std::unique_ptr<StereoSource> create_realsense_source(const SourceConfig& config);

// The layout a RealSense gives for `config`, known without the camera.
[[nodiscard]] SourceLayout realsense_layout(const SourceConfig& config);

}  // namespace holoscan::stereo_camera::detail
