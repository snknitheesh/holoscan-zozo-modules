// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <cstdint>
#include <utility>

#include <holoscan/icp/registration.hpp>

namespace holoscan::icp {

IcpOdometry::IcpOdometry(IcpOptions options, bool constant_velocity)
    : registration_(std::move(options)), constant_velocity_(constant_velocity) {}

void IcpOdometry::reset() noexcept {
  pose_ = identity_transform();
  motion_ = identity_transform();
  frames_ = 0;
}

cudaError_t IcpOdometry::track(const float* points, std::uint32_t count, cudaStream_t stream,
                               OdometryStep& step, const Matrix4* guess) {
  step = OdometryStep{};
  if (frames_ == 0U) {
    // The first frame defines the trajectory's origin and only becomes the target.
    if (const cudaError_t status = registration_.set_target(points, count, stream);
        status != cudaSuccess) {
      return status;
    }
    step.result.target_points = registration_.target_points();
    step.result.source_points = step.result.target_points;
    step.pose = pose_;
    step.first = true;
    frames_ = 1;
    return cudaSuccess;
  }
  const Matrix4 initial =
      guess != nullptr ? *guess : (constant_velocity_ ? motion_ : identity_transform());
  if (const cudaError_t status = registration_.align(points, count, initial, stream, step.result);
      status != cudaSuccess) {
    return status;
  }
  // The registration maps this frame into the previous one: previous_from_current.
  const bool failed = step.result.status == IcpStatus::kTooFewCorrespondences ||
                      step.result.status == IcpStatus::kDegenerate;
  if (failed) {
    // The guess stands in for the motion, in the result as in the pose.
    step.result.transform = initial;
  }
  motion_ = step.result.transform;
  pose_ = compose(pose_, motion_);
  step.pose = pose_;
  ++frames_;
  return registration_.promote_source_to_target(stream);
}

}  // namespace holoscan::icp
