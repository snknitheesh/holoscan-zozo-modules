// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The synthetic world's renders: each camera pixel, or each output pixel of
// the truth, averages a grid of rays cast into the world (ground, wall, sky).

#include <cuda_runtime.h>

#include <cmath>
#include <numbers>

#include "holoscan/stitching/synthetic_world.hpp"
#include "stitch_math.hpp"

namespace holoscan::stitching {
namespace {

using detail::CameraParams;
using detail::GeometryParams;

constexpr dim3 kBlock{16, 16, 1};

// The world as the kernels read it, with the rig's pose.
struct WorldParams {
  float wall_radius;
  float wall_height;
  float ground_square;
  int wall_columns;
  float wall_row_height;
  float pose_x;
  float pose_y;
  float cos_yaw;
  float sin_yaw;
};

WorldParams world_params(const SyntheticWorld& world, const RigPose& pose) {
  const double yaw = pose.yaw_deg * std::numbers::pi / 180.0;
  return WorldParams{static_cast<float>(world.wall_radius),
                     static_cast<float>(world.wall_height),
                     static_cast<float>(world.ground_square),
                     world.wall_columns,
                     static_cast<float>(world.wall_row_height),
                     static_cast<float>(pose.x),
                     static_cast<float>(pose.y),
                     static_cast<float>(std::cos(yaw)),
                     static_cast<float>(std::sin(yaw))};
}

enum Surface : int { kSky = 0, kGround = 1, kWall = 2 };

// A checker square's colour: light or dark by parity, its hue from a hash.
__device__ float3 checker(int i, int j, int salt) {
  constexpr float kLight[6][3] = {{200, 170, 150}, {150, 200, 170}, {170, 160, 205},
                                  {205, 195, 140}, {140, 190, 205}, {190, 150, 190}};
  constexpr float kDark[6][3] = {{70, 40, 30}, {30, 70, 45}, {45, 40, 80},
                                 {75, 65, 25}, {25, 60, 75}, {65, 30, 60}};
  const unsigned h = (static_cast<unsigned>(i) * 73856093U) ^
                     (static_cast<unsigned>(j) * 19349663U) ^
                     (static_cast<unsigned>(salt) * 83492791U);
  const int hue = static_cast<int>((h >> 7) % 6U);
  const bool light = ((i + j) & 1) == 0;
  const float* c = light ? kLight[hue] : kDark[hue];
  return make_float3(c[0], c[1], c[2]);
}

// Casts a ray of the world frame: its colour, and which surface it meets.
__device__ float3 trace(const WorldParams& w, float3 o, float3 d, int& surface) {
  // The wall is centred on the rig.
  const float ox = o.x - w.pose_x;
  const float oy = o.y - w.pose_y;
  const float a = d.x * d.x + d.y * d.y;
  float t_wall = 3.0e38F;
  if (a > 0.0F) {
    const float b = ox * d.x + oy * d.y;
    const float c = ox * ox + oy * oy - w.wall_radius * w.wall_radius;
    const float disc = b * b - a * c;
    if (disc >= 0.0F) {
      t_wall = (-b + sqrtf(disc)) / a;
    }
  }
  const float t_ground = d.z < 0.0F ? -o.z / d.z : 3.0e38F;
  if (t_ground < t_wall) {
    surface = kGround;
    const float x = o.x + t_ground * d.x;
    const float y = o.y + t_ground * d.y;
    const int i = static_cast<int>(floorf(x / w.ground_square));
    const int j = static_cast<int>(floorf(y / w.ground_square));
    const float3 c = checker(i, j, 1);
    const float shade = 0.9F + 0.1F * cosf(0.86F * x) * cosf(1.23F * y);
    return make_float3(c.x * shade, c.y * shade, c.z * shade);
  }
  const float z = o.z + t_wall * d.z;
  if (t_wall < 3.0e38F && z >= 0.0F && z <= w.wall_height) {
    surface = kWall;
    const float x = ox + t_wall * d.x;
    const float y = oy + t_wall * d.y;
    float azimuth = atan2f(y, x);
    if (azimuth < 0.0F) azimuth += 2.0F * 3.14159265358979F;
    const int i = static_cast<int>(floorf(azimuth / (2.0F * 3.14159265358979F) *
                                          static_cast<float>(w.wall_columns))) %
                  w.wall_columns;
    const int j = static_cast<int>(floorf(z / w.wall_row_height));
    const float3 c = checker(i, j, 2);
    const float shade = 0.9F + 0.1F * cosf(3.0F * azimuth) * cosf(0.9F * z);
    return make_float3(c.x * shade, c.y * shade, c.z * shade);
  }
  surface = kSky;
  return make_float3(110.0F, 150.0F, 200.0F);
}

// A direction or point of the rig frame in the world frame.
__device__ float3 to_world(const WorldParams& w, float3 v, bool point) {
  float3 r = make_float3(w.cos_yaw * v.x - w.sin_yaw * v.y, w.sin_yaw * v.x + w.cos_yaw * v.y, v.z);
  if (point) {
    r.x += w.pose_x;
    r.y += w.pose_y;
  }
  return r;
}

__device__ void store(float3 rgb, float gain, int format, std::uint8_t* out) {
  const float r = rgb.x * gain;
  const float g = rgb.y * gain;
  const float b = rgb.z * gain;
  switch (format) {
    case detail::kMono8:
      out[0] = detail::to_byte(0.299F * r + 0.587F * g + 0.114F * b);
      return;
    case detail::kBgr8:
    case detail::kBgra8:
      out[0] = detail::to_byte(b);
      out[1] = detail::to_byte(g);
      out[2] = detail::to_byte(r);
      break;
    default:
      out[0] = detail::to_byte(r);
      out[1] = detail::to_byte(g);
      out[2] = detail::to_byte(b);
      break;
  }
  if (format == detail::kRgba8 || format == detail::kBgra8) {
    out[3] = 255;
  }
}

__global__ void camera_kernel(WorldParams w, CameraParams c, float3 origin, float gain, int format,
                              int channels, std::uint8_t* image, std::size_t pitch, int samples) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= c.width || y >= c.height) {
    return;
  }
  float3 sum = make_float3(0.0F, 0.0F, 0.0F);
  for (int sy = 0; sy < samples; ++sy) {
    for (int sx = 0; sx < samples; ++sx) {
      const float u = static_cast<float>(x) - 0.5F + (static_cast<float>(sx) + 0.5F) / samples;
      const float v = static_cast<float>(y) - 0.5F + (static_cast<float>(sy) + 0.5F) / samples;
      float ray[3];
      if (!detail::pixel_ray(c, u, v, ray)) {
        continue;
      }
      // rig = R^T camera.
      float3 d_rig;
      d_rig.x = static_cast<float>(c.rotation[0]) * ray[0] +
                static_cast<float>(c.rotation[3]) * ray[1] +
                static_cast<float>(c.rotation[6]) * ray[2];
      d_rig.y = static_cast<float>(c.rotation[1]) * ray[0] +
                static_cast<float>(c.rotation[4]) * ray[1] +
                static_cast<float>(c.rotation[7]) * ray[2];
      d_rig.z = static_cast<float>(c.rotation[2]) * ray[0] +
                static_cast<float>(c.rotation[5]) * ray[1] +
                static_cast<float>(c.rotation[8]) * ray[2];
      int surface;
      const float3 rgb = trace(w, to_world(w, origin, true), to_world(w, d_rig, false), surface);
      sum.x += rgb.x;
      sum.y += rgb.y;
      sum.z += rgb.z;
    }
  }
  const float n = static_cast<float>(samples * samples);
  store(make_float3(sum.x / n, sum.y / n, sum.z / n), gain, format,
        image + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x) * channels);
}

__global__ void truth_kernel(WorldParams w, GeometryParams g, std::uint8_t* rgba, std::size_t pitch,
                             int samples) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= g.width || y >= g.height) {
    return;
  }
  const bool birds_eye = g.projection == detail::kBirdsEye;
  float3 sum = make_float3(0.0F, 0.0F, 0.0F);
  bool on_surface = true;
  for (int sy = 0; sy < samples; ++sy) {
    for (int sx = 0; sx < samples; ++sx) {
      const double px = x - 0.5 + (sx + 0.5) / samples;
      const double py = y - 0.5 + (sy + 0.5) / samples;
      double p[3];
      const bool at_infinity = detail::output_point(g, px, py, p);
      float3 origin;
      float3 direction;
      if (birds_eye) {
        // Straight down onto the ground point.
        origin = make_float3(static_cast<float>(p[0]), static_cast<float>(p[1]), 1.0F);
        direction = make_float3(0.0F, 0.0F, -1.0F);
      } else {
        origin = make_float3(static_cast<float>(g.center[0]), static_cast<float>(g.center[1]),
                             static_cast<float>(g.center[2]));
        direction = at_infinity ? make_float3(static_cast<float>(p[0]), static_cast<float>(p[1]),
                                              static_cast<float>(p[2]))
                                : make_float3(static_cast<float>(p[0] - g.center[0]),
                                              static_cast<float>(p[1] - g.center[1]),
                                              static_cast<float>(p[2] - g.center[2]));
      }
      int surface;
      const float3 rgb =
          trace(w, to_world(w, origin, true), to_world(w, direction, false), surface);
      // Beyond the wall, the cameras see the wall where the view shows ground.
      const bool inside_wall = !birds_eye || p[0] * p[0] + p[1] * p[1] <
                                                 static_cast<double>(w.wall_radius) * w.wall_radius;
      on_surface = on_surface && inside_wall && surface == (birds_eye ? kGround : kWall);
      sum.x += rgb.x;
      sum.y += rgb.y;
      sum.z += rgb.z;
    }
  }
  const float n = static_cast<float>(samples * samples);
  std::uint8_t* out = rgba + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x) * 4;
  store(make_float3(sum.x / n, sum.y / n, sum.z / n), 1.0F, detail::kRgb8, out);
  out[3] = on_surface ? 255 : 0;
}

dim3 grid_for(int width, int height) {
  return dim3((static_cast<unsigned>(width) + kBlock.x - 1) / kBlock.x,
              (static_cast<unsigned>(height) + kBlock.y - 1) / kBlock.y, 1);
}

int format_code(PixelFormat format) {
  switch (format) {
    case PixelFormat::kMono8:
      return detail::kMono8;
    case PixelFormat::kRgb8:
      return detail::kRgb8;
    case PixelFormat::kRgba8:
      return detail::kRgba8;
    case PixelFormat::kBgr8:
      return detail::kBgr8;
    case PixelFormat::kBgra8:
      return detail::kBgra8;
  }
  return detail::kRgb8;
}

}  // namespace

cudaError_t render_camera(const SyntheticWorld& world, const Camera& camera, const RigPose& pose,
                          float gain, PixelFormat format, void* image, std::size_t pitch,
                          int samples, cudaStream_t stream) {
  if (image == nullptr || samples < 1 || samples > 8 ||
      pitch < static_cast<std::size_t>(camera.width) * channels(format)) {
    return cudaErrorInvalidValue;
  }
  const CameraParams c = detail::camera_params(camera);
  // The camera's centre in the rig: -R^T t.
  const RigidTransform rig_from_camera = inverse(camera.camera_from_rig);
  const float3 origin = make_float3(static_cast<float>(rig_from_camera.translation[0]),
                                    static_cast<float>(rig_from_camera.translation[1]),
                                    static_cast<float>(rig_from_camera.translation[2]));
  camera_kernel<<<grid_for(camera.width, camera.height), kBlock, 0, stream>>>(
      world_params(world, pose), c, origin, gain, format_code(format), channels(format),
      static_cast<std::uint8_t*>(image), pitch, samples);
  return cudaGetLastError();
}

cudaError_t render_truth(const SyntheticWorld& world, const OutputGeometry& geometry,
                         const RigPose& pose, void* rgba, std::size_t pitch, int samples,
                         cudaStream_t stream) {
  if (rgba == nullptr || samples < 1 || samples > 8 ||
      pitch < static_cast<std::size_t>(geometry.width) * 4U) {
    return cudaErrorInvalidValue;
  }
  const GeometryParams g = detail::geometry_params(geometry);
  truth_kernel<<<grid_for(geometry.width, geometry.height), kBlock, 0, stream>>>(
      world_params(world, pose), g, static_cast<std::uint8_t*>(rgba), pitch, samples);
  return cudaGetLastError();
}

}  // namespace holoscan::stitching
