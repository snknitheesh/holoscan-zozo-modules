// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Synthetic scenes of planes, boxes and spheres, ray cast on the GPU into the
// depth image or organized point cloud of a pinhole camera at a known pose,
// with optional depth noise; and a smooth closed camera path through them.
// With the camera's poses known, they give ground truth for registration and
// odometry. For tests, benchmarks and demos; nothing here depends on the
// Holoscan SDK.
//
// Frames: the world's axes are those of the camera at the identity pose, the
// optical frame (x right, y down, z forward). A pose maps camera-frame points
// into the world (world_from_camera).

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include <holoscan/icp/rigid_transform.hpp>

namespace holoscan::icp {

// A pinhole camera; pixel centers are at integer coordinates.
struct PinholeCamera {
  int width{};
  int height{};
  float fx{};
  float fy{};
  float cx{};
  float cy{};
};

// A camera of `width` x `height` pixels with a horizontal field of view of
// about 64 degrees and square pixels, centered.
[[nodiscard]] PinholeCamera synthetic_camera(int width, int height);

// The points x with normal . x = offset; both sides are solid.
struct ScenePlane {
  float normal[3]{};
  float offset{};
};

// An axis-aligned box in the world.
struct SceneBox {
  float min[3]{};
  float max[3]{};
};

struct SceneSphere {
  float center[3]{};
  float radius{};
};

inline constexpr int kMaxScenePrimitives = 8;

struct SyntheticScene {
  ScenePlane planes[kMaxScenePrimitives]{};
  int plane_count{};
  SceneBox boxes[kMaxScenePrimitives]{};
  int box_count{};
  SceneSphere spheres[kMaxScenePrimitives]{};
  int sphere_count{};
};

// The scenes the module's tests use; all are about 1 to 6 m from the
// identity pose, in front of it.
enum class SceneKind : std::uint8_t {
  // A closed room, 6 m wide and 8 m deep, with boxes and spheres on the floor:
  // every ray hits something. synthetic_trajectory() moves inside it.
  kRoom,
  // A floor with boxes and spheres on it and nothing else: rays above the
  // horizon hit nothing.
  kObjects,
  // The corner of two walls and the floor, with spheres in it.
  kCorner,
};

[[nodiscard]] SyntheticScene synthetic_scene(SceneKind kind);

// Depth noise: each pixel's depth z gets Gaussian noise of standard
// deviation sigma + sigma_per_m2 * z^2, drawn from `seed` and the pixel,
// so the same seed gives the same noise.
struct DepthNoise {
  float sigma{};
  float sigma_per_m2{};
  std::uint32_t seed{};
};

// Renders the scene seen by `camera` at pose `world_from_camera` into a
// packed width x height float32 depth image in meters, 0 where nothing lies
// within `max_depth`; device memory, ordered on `stream`.
[[nodiscard]] cudaError_t render_scene_depth(const SyntheticScene& scene,
                                             const Matrix4& world_from_camera,
                                             const PinholeCamera& camera, float max_depth,
                                             const DepthNoise& noise, float* depth,
                                             cudaStream_t stream);

// As render_scene_depth(), as an organized [height, width, 3] float32 point
// cloud in the camera frame, NaN where there is no depth; device memory.
[[nodiscard]] cudaError_t render_scene_cloud(const SyntheticScene& scene,
                                             const Matrix4& world_from_camera,
                                             const PinholeCamera& camera, float max_depth,
                                             const DepthNoise& noise, float* points,
                                             cudaStream_t stream);

// Pose `frame` of `frames` on a smooth closed path through the room scene:
// the camera sways sideways and up and down, moves forward and back, yaws,
// pitches and rolls, and is back at the identity at frame `frames`: about
// 5.5 m in all. Each step moves it at most about 7 / frames meters and turns
// it at most about 120 / frames degrees.
[[nodiscard]] Matrix4 synthetic_trajectory(std::uint32_t frame, std::uint32_t frames);

}  // namespace holoscan::icp
