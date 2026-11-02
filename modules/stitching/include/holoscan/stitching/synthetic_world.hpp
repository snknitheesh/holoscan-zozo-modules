// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic world to test stitching against: an endless ground plane with a
// coloured checkerboard, and a checkered cylindrical wall around the rig (a
// backdrop that moves with the rig, but keeps its orientation, like a
// skybox). Virtual cameras of any rig render it on the GPU; the ground truth
// of an output geometry is rendered directly, for comparison; and the
// checkerboards' corners give a geometric check. For tests and demos.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <holoscan/stitching/camera_rig.hpp>
#include <holoscan/stitching/projection.hpp>
#include <holoscan/stitching/remap.hpp>

namespace holoscan::stitching {

struct SyntheticWorld {
  // The wall: a vertical cylinder of this radius around the rig's origin,
  // from the ground to this height, in metres.
  double wall_radius{6.0};
  double wall_height{4.0};
  // The checkerboards: ground squares of this side; the wall in this many
  // columns around, and rows of this height.
  double ground_square{0.5};
  int wall_columns{72};
  double wall_row_height{0.5};
};

// Where the rig is in the world: its origin's position on the ground and its
// heading (world_from_rig is a turn about z by yaw, then this offset).
struct RigPose {
  double x{};
  double y{};
  double yaw_deg{};
};

// The demo's motion: the rig drives around a circle of 3 m radius, one lap
// every `frames_per_lap` frames, facing along its path.
[[nodiscard]] RigPose moving_pose(std::uint32_t frame, std::uint32_t frames_per_lap = 600);

// Four fisheye cameras (kKannalaBrandt4, about 215 degrees across, 120
// degrees high) on a 4 m x 2 m vehicle, as surround-view systems mount them:
// "front" and "rear" 0.8 m up at the bumpers, tilted down 30 degrees, "left"
// and "right" 1 m up on the sides, tilted down 40 degrees.
[[nodiscard]] CameraRig surround_rig(int width = 1280, int height = 720);

// `count` pinhole cameras (kBrownConrady, mild barrel distortion) of
// `horizontal_fov_deg` across, looking outwards from a ring of `radius` metres
// at 1 m height, evenly around, level; camera 0 looks along +x.
[[nodiscard]] CameraRig ring_rig(int count, int width, int height, double horizontal_fov_deg,
                                 double radius);

// A 360-degree panorama seen from 1 m (`center_height`) above the rig's
// origin, with square pixels, its rows centred on the wall, and the wall's
// radius: every pixel between the wall's bottom and top edges is a point of
// the wall; above and below, the cylinder (or sphere) continues where the
// world has sky and ground.
[[nodiscard]] OutputGeometry wall_panorama(const SyntheticWorld& world, Projection projection,
                                           int width, int height, double center_height = 1.0);

// A bird's-eye view of the ground `half_extent` metres around the rig's
// origin in each direction.
[[nodiscard]] OutputGeometry ground_view(int width, int height, double half_extent);

// Renders what `camera` of a rig at `pose` sees, scaled by `gain` (an
// exposure), into `image`: `camera.width` x `camera.height` in `format`,
// `pitch` bytes per row, in CUDA device memory. Each pixel averages
// `samples` x `samples` rays (1..8); pixels with no valid ray are black.
[[nodiscard]] cudaError_t render_camera(const SyntheticWorld& world, const Camera& camera,
                                        const RigPose& pose, float gain, PixelFormat format,
                                        void* image, std::size_t pitch, int samples,
                                        cudaStream_t stream);

// Renders the ground truth of `geometry` for a rig at `pose` into `rgba`
// (RGBA8, `pitch` bytes per row, CUDA device memory): what the output pixel's
// point really looks like, seen from the panorama's centre. Alpha is 255
// where every ray of the pixel meets the surface the stitcher assumes there,
// so that a stitch can match it: the ground for a bird's-eye view, the wall
// for a panorama; 0 elsewhere.
[[nodiscard]] cudaError_t render_truth(const SyntheticWorld& world, const OutputGeometry& geometry,
                                       const RigPose& pose, void* rgba, std::size_t pitch,
                                       int samples, cudaStream_t stream);

// Where the corners of the checkerboard the output shows (the ground's for a
// bird's-eye view, the wall's for a panorama) appear in it, in pixels, at
// least `margin` pixels inside it.
[[nodiscard]] std::vector<Point2> checker_corners(const SyntheticWorld& world,
                                                  const OutputGeometry& geometry,
                                                  const RigPose& pose, double margin = 8.0);

// Refines a checkerboard corner of a gray image near `guess`, as OpenCV's
// cornerSubPix does: the point that every gradient within `radius` pixels is
// orthogonal to. nullopt when the window holds too little gradient, leaves
// the image, or the corner moves further than `radius` / 2.
[[nodiscard]] std::optional<Point2> refine_corner(std::span<const float> gray, int width,
                                                  int height, Point2 guess, int radius = 5);

// How a stitched image compares with the ground truth, over the pixels the
// stitch covers (alpha 255) and the truth marks comparable, at least `border`
// pixels inside the image.
struct Comparison {
  std::size_t compared_pixels{};
  double coverage{};  // Of the comparable pixels, the share the stitch covers.
  // The ratio of the stitch's mean to the truth's, and the PSNR (8-bit peak)
  // of the stitch against the truth scaled by it, so that an exposure change
  // common to every camera does not count.
  double gain{};
  double psnr_db{};
  // The corners refined in both images, and how far apart, in pixels.
  int corners{};
  double max_corner_error{};
  double mean_corner_error{};
};

// `stitched` is RGBA8 and `truth` as render_truth() writes it, both
// `width` x `height`, `pitch` bytes per row, in host memory. With
// `resolution` (resolution_map() of the stitch's tables), only the corners
// where the cameras resolve at least `min_resolution` camera pixels per
// output pixel are checked: where the output magnifies a camera more, one
// output pixel is a small fraction of what the camera can tell apart.
[[nodiscard]] Comparison compare_with_truth(const std::uint8_t* stitched,
                                            std::size_t stitched_pitch, const std::uint8_t* truth,
                                            std::size_t truth_pitch, int width, int height,
                                            std::span<const Point2> corners, int border = 2,
                                            std::span<const float> resolution = {},
                                            float min_resolution = 0.25F);

}  // namespace holoscan::stitching
