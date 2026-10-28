// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A point cloud made from the synthetic depth camera of synthetic_depth.hpp.
// Its depth image is deprojected into points, moved into a body frame and
// shuffled, with points that must not reach the depth image mixed in:
// duplicates farther along their pixel's ray, points behind the camera and
// outside its field of view, and non-finite points where the camera has no
// depth. The same functions run on the GPU to make clouds and on the host to
// make them for the tests and to know what the depth image should show; as
// for the camera, host code that does must not contract floating-point
// arithmetic (-ffp-contract=off). For tests and demos.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <holoscan/point_cloud/project.hpp>
#include <holoscan/point_cloud/synthetic_depth.hpp>

namespace holoscan::point_cloud {

// The scene's depth in meters at pixel (u, v). With `millimeter_depth` it is a
// whole number of millimeters, so that a uint16 depth image in millimeters
// reproduces it exactly; otherwise it is synthetic_depth().
HOLOSCAN_POINT_CLOUD_HOST_DEVICE inline float synthetic_scene_depth(const SyntheticFrame& f,
                                                                    bool millimeter_depth, int u,
                                                                    int v) {
  return millimeter_depth ? static_cast<float>(synthetic_depth_mm(f, u, v)) * 0.001F
                          : synthetic_depth(f, u, v);
}

// Where the synthetic camera sits on its body: body x forward is camera z,
// body y left is camera -x, body z up is camera -y, and the camera is 12 cm
// forward, 4 cm right and 35 cm up of the body origin. Maps camera-frame
// points into the body frame.
constexpr Extrinsics synthetic_body_from_camera() {
  return Extrinsics{{0.0F, 0.0F, 1.0F, 0.12F,     // Body x: camera z.
                     -1.0F, 0.0F, 0.0F, -0.04F,   // Body y: camera -x.
                     0.0F, -1.0F, 0.0F, 0.35F}};  // Body z: camera -y.
}

// Every kDuplicateEvery-th pixel gets a second point kDuplicateDepthFactor
// times as far along its ray, which must lose the z-buffer. Doubling is exact,
// so a millimeter depth stays a whole number of millimeters.
inline constexpr std::uint32_t kDuplicateEvery = 10U;
inline constexpr float kDuplicateDepthFactor = 2.0F;
// Every kStrayEvery-th pixel gets a point behind the camera, and the pixel
// kOutsideOffset after it a point one image width to the right, outside the
// field of view.
inline constexpr std::uint32_t kStrayEvery = 50U;
inline constexpr std::uint32_t kOutsideOffset = 25U;

// How many points of each kind a synthetic cloud has. Records are numbered
// in this order before the cloud is shuffled.
struct SyntheticCloudLayout {
  std::uint32_t pixels{};      // One per pixel: on the surface, or non-finite in a hole.
  std::uint32_t duplicates{};  // Pixels 0, kDuplicateEvery, 2 * kDuplicateEvery, ...
  std::uint32_t behind{};      // Pixels 0, kStrayEvery, 2 * kStrayEvery, ...
  std::uint32_t outside{};     // Pixels kOutsideOffset, kOutsideOffset + kStrayEvery, ...

  [[nodiscard]] HOLOSCAN_POINT_CLOUD_HOST_DEVICE std::uint32_t total() const {
    return pixels + duplicates + behind + outside;
  }
};

HOLOSCAN_POINT_CLOUD_HOST_DEVICE inline SyntheticCloudLayout synthetic_cloud_layout(int width,
                                                                                    int height) {
  const auto n = static_cast<std::uint32_t>(width) * static_cast<std::uint32_t>(height);
  SyntheticCloudLayout layout;
  layout.pixels = n;
  layout.duplicates = (n + kDuplicateEvery - 1U) / kDuplicateEvery;
  layout.behind = (n + kStrayEvery - 1U) / kStrayEvery;
  layout.outside = n > kOutsideOffset ? (n - kOutsideOffset + kStrayEvery - 1U) / kStrayEvery : 0U;
  return layout;
}

// Whether pixel `index` (v * width + u) has a duplicate point.
HOLOSCAN_POINT_CLOUD_HOST_DEVICE inline bool has_duplicate(std::uint32_t index) {
  return index % kDuplicateEvery == 0U;
}

// The color of a duplicate of a pixel whose color is `rgb`: its inverse, so a
// duplicate that wins shows in the color image as well as in the depth.
HOLOSCAN_POINT_CLOUD_HOST_DEVICE inline void duplicate_color(std::uint8_t rgb[3]) {
  for (int channel = 0; channel < 3; ++channel) {
    rgb[channel] = static_cast<std::uint8_t>(255U - rgb[channel]);
  }
}

// Record `record` of the cloud of frame `f`, moved by `body_from_camera`
// (synthetic_body_from_camera(), passed in as device code cannot call it),
// and its color.
HOLOSCAN_POINT_CLOUD_HOST_DEVICE inline void synthetic_point(
    const SyntheticFrame& f, const SyntheticCloudLayout& layout, bool millimeter_depth,
    const Extrinsics& body_from_camera, std::uint32_t record, float xyz[3], std::uint8_t rgb[3]) {
  enum class Kind { kSurface, kDuplicate, kBehind, kOutside };
  Kind kind = Kind::kSurface;
  std::uint32_t pixel = record;
  if (record >= layout.pixels + layout.duplicates + layout.behind) {
    kind = Kind::kOutside;
    pixel =
        (record - layout.pixels - layout.duplicates - layout.behind) * kStrayEvery + kOutsideOffset;
  } else if (record >= layout.pixels + layout.duplicates) {
    kind = Kind::kBehind;
    pixel = (record - layout.pixels - layout.duplicates) * kStrayEvery;
  } else if (record >= layout.pixels) {
    kind = Kind::kDuplicate;
    pixel = (record - layout.pixels) * kDuplicateEvery;
  }
  const int u = static_cast<int>(pixel % static_cast<std::uint32_t>(f.width));
  const int v = static_cast<int>(pixel / static_cast<std::uint32_t>(f.width));
  const float depth = synthetic_scene_depth(f, millimeter_depth, u, v);
  const bool hole = !(depth > 0.0F);
  float z = hole ? 1.0F : depth;
  if (kind == Kind::kDuplicate) {
    z *= kDuplicateDepthFactor;
  }
  const CameraIntrinsics& k = f.intrinsics;
  const float column =
      kind == Kind::kOutside ? static_cast<float>(u + f.width) : static_cast<float>(u);
  const float camera[3] = {(column - k.cx) * z / k.fx, (static_cast<float>(v) - k.cy) * z / k.fy,
                           kind == Kind::kBehind ? -z : z};
  const float* m = body_from_camera.matrix;
  for (int row = 0; row < 3; ++row) {
    xyz[row] = m[4 * row] * camera[0] + m[4 * row + 1] * camera[1] + m[4 * row + 2] * camera[2] +
               m[4 * row + 3];
  }
  if (hole) {
    // One non-finite coordinate, a different one from pixel to pixel.
    const std::uint32_t which = pixel % 4U;
#if defined(__CUDA_ARCH__)
    const float nan = __int_as_float(0x7FC00000);
    const float infinity = __int_as_float(0x7F800000);
#else
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
#endif
    xyz[which < 3U ? which : 0U] = which < 3U ? nan : infinity;
  }
  if (kind == Kind::kBehind || kind == Kind::kOutside) {
    rgb[0] = 255U;
    rgb[1] = 0U;
    rgb[2] = 255U;
    return;
  }
  synthetic_color(f, u, v, rgb);
  if (kind == Kind::kDuplicate) {
    duplicate_color(rgb);
  }
}

// The deterministic shuffle of a cloud: record r goes to position
// (multiplier * r + offset) % total, a permutation as multiplier and total are
// coprime.
struct CloudShuffle {
  std::uint64_t multiplier{1U};
  std::uint64_t offset{};
};

// The shuffle of frame `frame`'s cloud of `total` points.
[[nodiscard]] CloudShuffle synthetic_shuffle(std::uint32_t total, std::uint32_t frame);

// The cloud of frame `f` as the GPU makes it, made on the host: packed x, y, z
// in `xyz` and RGB in `rgb`, layout.total() points each.
void synthetic_cloud(const SyntheticFrame& f, bool millimeter_depth, std::vector<float>& xyz,
                     std::vector<std::uint8_t>& rgb);

// Makes the cloud of frame `f` in device buffers on `stream`: points as packed
// float32 x, y, z, and colors as RGB8. Either buffer may be null to skip it.
cudaError_t generate_synthetic_cloud(const SyntheticFrame& f, bool millimeter_depth, float* points,
                                     std::uint8_t* colors, cudaStream_t stream);

}  // namespace holoscan::point_cloud
