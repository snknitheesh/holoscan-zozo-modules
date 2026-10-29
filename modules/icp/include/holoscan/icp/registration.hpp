// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Iterative closest point registration on the GPU: IcpRegistration aligns a
// source cloud to a target cloud, and IcpOdometry chains it frame to frame.
// Nothing here depends on the Holoscan SDK; IcpOp wraps these for a graph.
//
// How it works, per call:
//   1. Each cloud's non-finite points are dropped and, with a voxel size, one
//      point of each occupied voxel is kept, picked by a hash of its index.
//   2. The target's points are sorted into a uniform grid of cubic cells a
//      little wider than the correspondence distance, so every point within
//      that distance of a query is in the 27 cells around it. For point-to-plane,
//      each target point gets a normal from the principal axes of its
//      neighbors within the normal radius.
//   3. Each iteration, one kernel moves every source point by the current
//      transform, finds its nearest target point within the correspondence
//      distance, and adds its terms to the 6x6 normal equations; CUB block
//      reductions, then one fixed-order pass, sum them. The host solves the
//      6x6 system and updates the transform.
//   4. Coarse to fine: the correspondence distance starts at the maximum and
//      halves each time the iterations converge, down to the minimum. A wide
//      distance catches a large motion; a narrow one keeps points the other
//      cloud does not see from pulling on the result.
//   5. A last pass at the final transform gives the fitness and RMSE.
//
// Every step sums in a fixed order and breaks ties by point index, so the
// same input on the same GPU gives bit-identical results.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include <holoscan/icp/rigid_transform.hpp>

namespace holoscan::icp {

enum class IcpMethod : std::uint8_t {
  // Minimizes the distance between corresponding points.
  kPointToPoint,
  // Minimizes the distance of each source point to its target point's
  // tangent plane; converges in fewer iterations on surfaces, and needs the
  // target's normals.
  kPointToPlane,
};

// The weight of a correspondence by the size of its residual r: none weighs
// all alike; Huber weighs k / |r| beyond k; Tukey (1 - (r / k)^2)^2 within k
// and 0 beyond. The residual is the point-to-plane distance, or the
// point-to-point distance.
enum class RobustKernel : std::uint8_t { kNone, kHuber, kTukey };

struct IcpOptions {
  IcpMethod method{IcpMethod::kPointToPlane};
  // Source and target points farther apart than this, in the clouds' units
  // (meters for the module's operators), do not correspond: at first, ...
  float max_correspondence_distance{0.05F};
  // ...and then half as far each time the iterations converge, down to this
  // distance, at which they converge for good. 0, or the maximum, keeps one
  // distance throughout; the maximum may be at most 128 times this.
  float min_correspondence_distance{0.0F};
  // Iterations in all, each one Gauss-Newton update, 1..1000.
  int max_iterations{30};
  // Converged (at a distance) when one update moves by less than both of
  // these...
  double translation_epsilon{1e-5};
  double rotation_epsilon{1e-5};  // Radians.
  // ...or when the fitness does not change and the RMSE changes by less than
  // this fraction of itself from one iteration to the next.
  double relative_rmse_epsilon{1e-6};
  // Voxel edge for subsampling both clouds to one point per voxel; 0 keeps
  // every point.
  float voxel_size{0.0F};
  // Neighbors within this distance give a target point's normal
  // (point-to-plane); 0 uses max_correspondence_distance.
  float normal_radius{0.0F};
  RobustKernel robust_kernel{RobustKernel::kNone};
  // The kernel's k, in the clouds' units; used with a robust kernel.
  float robust_scale{0.01F};
  // Fewer correspondences than this end the registration unconverged.
  std::uint32_t min_correspondences{10U};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_options(const IcpOptions& options);

// Why a registration ended.
enum class IcpStatus : std::uint8_t {
  kConverged,              // At the minimum distance, by an epsilon.
  kMaxIterations,          // max_iterations updates without converging.
  kTooFewCorrespondences,  // Fewer than min_correspondences, or no target.
  kDegenerate,             // The normal equations could not be solved.
};

[[nodiscard]] const char* to_string(IcpStatus status);

struct IcpResult {
  // Maps source points into the target's frame.
  Matrix4 transform{identity_transform()};
  // Correspondences at the final transform and the final (minimum)
  // distance, over the source's points (after subsampling): 0..1.
  double fitness{};
  // Root mean square distance between those corresponding points; 0
  // without correspondences.
  double rmse{};
  // Updates applied.
  int iterations{};
  bool converged{};
  IcpStatus status{IcpStatus::kTooFewCorrespondences};
  std::uint32_t correspondences{};
  // Points used, after dropping non-finite ones and subsampling.
  std::uint32_t source_points{};
  std::uint32_t target_points{};
};

// Clouds are `count` packed float32 x, y, z triplets in CUDA device memory;
// non-finite points are ignored. Coordinates must be within 2^20 voxels (or
// grid cells) of the origin; points beyond are ignored. Calls are ordered on
// `stream` and return when their result is on the host. One instance is not
// thread-safe; its device memory grows to the largest clouds it has seen and
// is freed when it is destroyed.
class IcpRegistration {
 public:
  // @throws std::invalid_argument if `options` is not valid.
  explicit IcpRegistration(IcpOptions options);
  ~IcpRegistration();
  IcpRegistration(const IcpRegistration&) = delete;
  IcpRegistration& operator=(const IcpRegistration&) = delete;
  IcpRegistration(IcpRegistration&&) noexcept;
  IcpRegistration& operator=(IcpRegistration&&) noexcept;

  [[nodiscard]] const IcpOptions& options() const noexcept;

  // Replaces the target: subsamples it, builds its grid and, for
  // point-to-plane, estimates its normals.
  [[nodiscard]] cudaError_t set_target(const float* points, std::uint32_t count,
                                       cudaStream_t stream);
  // Makes the source of the last align() the target, without preparing it
  // again: the step from one frame to the next of odometry.
  [[nodiscard]] cudaError_t promote_source_to_target(cudaStream_t stream);
  [[nodiscard]] bool has_target() const noexcept;
  [[nodiscard]] std::uint32_t target_points() const noexcept;

  // Aligns the source to the target from `initial`, which maps source points
  // into the target's frame. Without a target, the result is unconverged
  // with status kTooFewCorrespondences.
  [[nodiscard]] cudaError_t align(const float* points, std::uint32_t count, const Matrix4& initial,
                                  cudaStream_t stream, IcpResult& result);

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

// One frame of IcpOdometry.
struct OdometryStep {
  // The registration of this frame to the one before; on the first frame,
  // the identity with no iterations.
  IcpResult result;
  // This frame's pose: maps its points into the first frame's camera.
  Matrix4 pose{identity_transform()};
  bool first{};
};

// Frame-to-frame odometry: each cloud is aligned to the previous one, and the
// relative transforms accumulate into a pose relative to the first frame.
// The initial guess for each frame is the previous frame's motion (constant
// velocity) or the identity. A frame whose registration fails with too few
// correspondences or degenerate equations keeps the guess as its motion, and
// its result's transform is the guess; one that only runs out of iterations
// keeps its estimate.
class IcpOdometry {
 public:
  // @throws std::invalid_argument if `options` is not valid.
  explicit IcpOdometry(IcpOptions options, bool constant_velocity = true);

  // Registers `points` to the previous call's cloud. `guess`, when given,
  // replaces the motion model's guess of the motion from the previous frame
  // to this one (previous_from_current).
  [[nodiscard]] cudaError_t track(const float* points, std::uint32_t count, cudaStream_t stream,
                                  OdometryStep& step, const Matrix4* guess = nullptr);
  // Forgets every frame: the next one starts a new trajectory at the identity.
  void reset() noexcept;
  [[nodiscard]] const Matrix4& pose() const noexcept { return pose_; }
  [[nodiscard]] std::uint32_t frames() const noexcept { return frames_; }

 private:
  IcpRegistration registration_;
  bool constant_velocity_{};
  Matrix4 pose_{identity_transform()};
  Matrix4 motion_{identity_transform()};
  std::uint32_t frames_{};
};

// Writes `transform` * p for each of `count` packed float32 points from `in`
// to `out` (which may be `in`), on `stream`; non-finite points stay as they
// are. Both are device memory.
[[nodiscard]] cudaError_t launch_transform_points(const float* in, float* out, std::uint64_t count,
                                                  const Matrix4& transform, cudaStream_t stream);

}  // namespace holoscan::icp
