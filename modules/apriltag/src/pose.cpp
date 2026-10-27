// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/apriltag/pose.hpp"

// The AprilTag 3 library's C API.
extern "C" {
#include <apriltag.h>
#include <apriltag_pose.h>

#include <common/matd.h>
}

#include <cmath>
#include <memory>

#include "homography.hpp"

namespace holoscan::apriltag {
namespace {

struct MatdDeleter {
  void operator()(matd_t* m) const noexcept { matd_destroy(m); }
};
using MatdPtr = std::unique_ptr<matd_t, MatdDeleter>;

}  // namespace

std::optional<TagPose> estimate_pose(const Detection& detection, const CameraIntrinsics& intrinsics,
                                     double tag_size) {
  double corners[4][2];
  for (int i = 0; i < 4; ++i) {
    corners[i][0] = detection.corners[i][0];
    corners[i][1] = detection.corners[i][1];
  }
  double homography[9];
  if (!homography_from_corners(corners, homography)) {
    return std::nullopt;
  }
  MatdPtr h{matd_create_data(3, 3, homography)};
  apriltag_detection_t det{};
  det.id = detection.id;
  det.H = h.get();
  det.c[0] = detection.center[0];
  det.c[1] = detection.center[1];
  for (int i = 0; i < 4; ++i) {
    det.p[i][0] = detection.corners[i][0];
    det.p[i][1] = detection.corners[i][1];
  }
  apriltag_detection_info_t info{.det = &det,
                                 .tagsize = tag_size,
                                 .fx = intrinsics.fx,
                                 .fy = intrinsics.fy,
                                 .cx = intrinsics.cx,
                                 .cy = intrinsics.cy};
  apriltag_pose_t pose{};
  const double error = estimate_tag_pose(&info, &pose);
  MatdPtr rotation{pose.R};
  MatdPtr translation{pose.t};
  if (!rotation || !translation || !std::isfinite(error)) {
    return std::nullopt;
  }
  TagPose result;
  for (int i = 0; i < 9; ++i) {
    result.rotation[i] = rotation->data[i];
  }
  for (int i = 0; i < 3; ++i) {
    result.translation[i] = translation->data[i];
  }
  result.error = error;
  return result;
}

}  // namespace holoscan::apriltag
