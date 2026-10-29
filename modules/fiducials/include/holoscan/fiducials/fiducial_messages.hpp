// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Converting between the FiducialDetections message and the module's own types, for the
// detectors and for whoever consumes their messages.

#pragma once

#include <optional>
#include <vector>

#include <holoscan/fiducials/charuco.hpp>
#include <holoscan/fiducials/fiducial_detections_schema_traits.hpp>
#include <holoscan/fiducials/types.hpp>

namespace holoscan::fiducials {

[[nodiscard]] schema::Pose to_message(const Pose& pose);
[[nodiscard]] Pose from_message(const schema::Pose& pose);
[[nodiscard]] schema::Marker to_message(const Marker& marker);
[[nodiscard]] Marker from_message(const schema::Marker& marker);

// A message's markers, poses included where present.
[[nodiscard]] std::vector<Marker> markers_of(const schema::FiducialDetectionsT& message);
// A CharucoDetectorOp message's board view, with the message's markers; nullopt for a message
// without one.
[[nodiscard]] std::optional<CharucoObservation> board_of(
    const schema::FiducialDetectionsT& message);

}  // namespace holoscan::fiducials
