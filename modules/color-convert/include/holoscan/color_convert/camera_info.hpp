// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The camera of a resized image: what ResizeOp sends on `resized_camera_info`
// for the images it makes from a camera's. No SDK, CUDA or GPU.

#pragma once

#include <optional>
#include <string>

#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/color_convert/pixel_format.hpp>
#include <holoscan/messages/camera_info.hpp>

namespace holoscan::color_convert {

// Why `info` (already valid, messages::validate()) does not describe
// `width` x `height` images, or nullopt: it must be for images of that size.
[[nodiscard]] std::optional<std::string> camera_info_problem(const messages::CameraInfo& info,
                                                             int width, int height);

// The camera of the `width` x `height` `destination` image that a conversion
// with `params` makes of a `source_width` x `source_height` image that `info`
// describes.
//
// compute_mapping() places the source rectangle s (the crop, or the whole
// image) at the content rectangle c of the output, scaled by
// sx = c.width / s.width and sy = c.height / s.height, with pixel centers at
// integer coordinates: source pixel x lands at (x + 0.5 - s.x) * sx - 0.5 + c.x.
// So fx' = fx * sx, skew' = skew * sx, cx' = (cx + 0.5 - s.x) * sx - 0.5 + c.x,
// and likewise fy, cy with sy. Distortion acts before the intrinsics, so the
// model, its coefficients, the state and the rectification stay; the letterbox
// band is outside every projection.
//
// The region of interest records a crop, in full-resolution sensor pixels:
// each source pixel spans roi.width / source_width sensor columns when `info`
// has a region of interest, else binning_x (the images taken as the sensor's
// at its binning), and likewise for rows; the crop's offset and size are
// scaled by that and rounded to whole sensor pixels. Without a crop the region
// of interest stays, and the binning always does: it records how the images
// were read from the sensor, not how they were resampled since.
//
// The header is kept; the sender stamps its capture time.
// @throws std::invalid_argument if camera_info_problem() finds one for the
// source size, the conversion is not possible (compute_mapping()), or the
// result is not valid.
[[nodiscard]] messages::CameraInfo resized_camera_info(const messages::CameraInfo& info,
                                                       int source_width, int source_height,
                                                       int width, int height,
                                                       PixelFormat destination,
                                                       const ConversionParams& params);

}  // namespace holoscan::color_convert
