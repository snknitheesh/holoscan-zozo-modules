// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// TsdfVolume and MeshExtractor without a graph, on the synthetic scene
// rendered at test time from known poses with depth noise: the zero crossing
// of the integrated distance and the mesh's vertices against the exact
// surfaces, the mesh's stability, a closed sphere's mesh being watertight,
// the ray-cast depth against the exact depth, weight capping and truncation,
// the CPU reference on a small volume bit for bit, determinism, edge cases,
// the capacity, the PLY file and the settings.

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/tsdf/marching_cubes.hpp>
#include <holoscan/tsdf/synthetic_scene.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>

#include "reference.hpp"
#include "scene_support.hpp"
#include "test_support.hpp"

namespace holoscan::tsdf {
namespace {

using test::cuda_check;
using test::Device;
using test::ErrorStats;
using test::Frame;

constexpr int kWidth = 320;
constexpr int kHeight = 240;
constexpr std::uint32_t kFrames = 60;

void print(const std::string& what, const ErrorStats& s, float voxel) {
  std::printf("  %s: n=%zu mean=%.2f mm (%.2f voxel) p95=%.2f mm max=%.2f mm (%.2f voxel)\n",
              what.c_str(), s.count, 1000.0 * s.mean, s.mean / voxel, 1000.0 * s.p95,
              1000.0 * s.max, s.max / voxel);
}

// The points where the distance crosses zero along voxel edges between
// observed voxels.
std::vector<float> zero_crossings(const TsdfVolumeConfig& c, const VolumeData& data) {
  std::vector<float> points;
  const auto index = [&c](int x, int y, int z) {
    return static_cast<std::size_t>(x) +
           static_cast<std::size_t>(c.dims[0]) *
               (static_cast<std::size_t>(y) + static_cast<std::size_t>(c.dims[1]) * z);
  };
  for (int z = 0; z < c.dims[2]; ++z) {
    for (int y = 0; y < c.dims[1]; ++y) {
      for (int x = 0; x < c.dims[0]; ++x) {
        const std::size_t i = index(x, y, z);
        if (!(data.weight[i] > 0.0F)) {
          continue;
        }
        const int voxel[3] = {x, y, z};
        for (int axis = 0; axis < 3; ++axis) {
          if (voxel[axis] + 1 >= c.dims[axis]) {
            continue;
          }
          const std::size_t j = index(x + (axis == 0), y + (axis == 1), z + (axis == 2));
          if (!(data.weight[j] > 0.0F) || (data.tsdf[i] < 0.0F) == (data.tsdf[j] < 0.0F)) {
            continue;
          }
          const float s = data.tsdf[i] / (data.tsdf[i] - data.tsdf[j]);
          for (int r = 0; r < 3; ++r) {
            points.push_back(c.origin[r] + c.voxel_size * (static_cast<float>(voxel[r]) +
                                                           (r == axis ? s : 0.0F)));
          }
        }
      }
    }
  }
  return points;
}

void check_surface_accuracy() {
  TsdfVolume volume(scene_volume(SceneKind::kTabletop, 160, true));
  test::integrate_orbit(volume, SceneKind::kTabletop, 0, kFrames, kFrames, kWidth, kHeight,
                        test::kNoise);
  const float voxel = volume.config().voxel_size;
  const SyntheticScene scene = synthetic_scene(SceneKind::kTabletop);
  const ErrorStats crossings =
      test::surface_error(scene, zero_crossings(volume.config(), volume.download()));
  print("TSDF zero crossings", crossings, voxel);
  VERIFY(crossings.count > 10000);
  VERIFY_MSG(crossings.mean < 0.35 * voxel, "zero crossings: mean error");
  VERIFY_MSG(crossings.p95 < 1.0 * voxel, "zero crossings: p95 error");

  const test::HostMesh mesh = test::extract(volume);
  const ErrorStats vertices = test::surface_error(scene, mesh.vertices);
  print("mesh vertices", vertices, voxel);
  std::printf("  mesh: %u vertices, %u triangles\n", mesh.counts[0], mesh.counts[1]);
  VERIFY(mesh.counts[kCountVertices] == mesh.counts[kCountNeededVertices]);
  VERIFY(mesh.counts[kCountTriangles] == mesh.counts[kCountNeededTriangles]);
  VERIFY(mesh.counts[kCountTriangles] > 20000);
  VERIFY_MSG(vertices.mean < 0.35 * voxel, "mesh: mean vertex error");
  VERIFY_MSG(vertices.p95 < 1.0 * voxel, "mesh: p95 vertex error");
  VERIFY_MSG(vertices.max < 4.0 * voxel, "mesh: max vertex error");

  // The vertex normals agree with the scene's, by its distance's gradient.
  std::size_t agree = 0;
  const std::size_t n = mesh.vertices.size() / 3;
  for (std::size_t i = 0; i < n; ++i) {
    const float* p = &mesh.vertices[3 * i];
    float g[3];
    for (int r = 0; r < 3; ++r) {
      float a[3] = {p[0], p[1], p[2]};
      float b[3] = {p[0], p[1], p[2]};
      a[r] += 1e-3F;
      b[r] -= 1e-3F;
      g[r] = scene_distance(scene, a) - scene_distance(scene, b);
    }
    const float length = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    const float* nn = &mesh.normals[3 * i];
    agree += length > 0.0F && (g[0] * nn[0] + g[1] * nn[1] + g[2] * nn[2]) / length > 0.9F ? 1 : 0;
  }
  std::printf("  normals within 26 degrees of the scene's: %.1f%%\n",
              100.0 * agree / std::max<std::size_t>(n, 1));
  VERIFY_MSG(agree > n * 9 / 10, "mesh normals agree with the surfaces");

  // Triangles wind counterclockwise seen from outside: their normals agree
  // with the vertex normals.
  std::size_t outward = 0;
  const std::size_t triangles = mesh.triangles.size() / 3;
  for (std::size_t t = 0; t < triangles; ++t) {
    const float* a = &mesh.vertices[3 * mesh.triangles[3 * t]];
    const float* b = &mesh.vertices[3 * mesh.triangles[3 * t + 1]];
    const float* c = &mesh.vertices[3 * mesh.triangles[3 * t + 2]];
    const float u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const float v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const float face[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                           u[0] * v[1] - u[1] * v[0]};
    const float* nn = &mesh.normals[3 * mesh.triangles[3 * t]];
    outward += face[0] * nn[0] + face[1] * nn[1] + face[2] * nn[2] > 0.0F ? 1 : 0;
  }
  std::printf("  triangles facing out: %.2f%%\n",
              100.0 * outward / std::max<std::size_t>(triangles, 1));
  VERIFY_MSG(outward > triangles * 98 / 100, "triangles wind counterclockwise seen from outside");

  // The colors: the sphere's vertices are blue-green, the box's red.
  std::size_t sphere = 0;
  std::size_t sphere_colored = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float* p = &mesh.vertices[3 * i];
    const float d[3] = {p[0] - scene.sphere_center[0], p[1] - scene.sphere_center[1],
                        p[2] - scene.sphere_center[2]};
    if (std::abs(std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) - scene.sphere_radius) <
        0.005F) {
      ++sphere;
      const std::uint8_t* rgb = &mesh.colors[3 * i];
      sphere_colored += rgb[2] > rgb[0] + 60 ? 1 : 0;
    }
  }
  VERIFY(sphere > 1000);
  VERIFY_MSG(sphere_colored > sphere * 9 / 10, "the sphere's vertices have its color");
}

void check_mesh_stability() {
  TsdfVolume volume(scene_volume(SceneKind::kTabletop, 128, false));
  test::integrate_orbit(volume, SceneKind::kTabletop, 0, 45, kFrames, kWidth, kHeight,
                        test::kNoise);
  const test::HostMesh first = test::extract(volume);
  const test::HostMesh again = test::extract(volume);
  VERIFY_MSG(first.triangles == again.triangles && first.vertices == again.vertices,
             "extracting twice gives the same mesh");
  test::integrate_orbit(volume, SceneKind::kTabletop, 45, kFrames - 45, kFrames, kWidth, kHeight,
                        test::kNoise);
  const test::HostMesh later = test::extract(volume);
  const double change = std::abs(static_cast<double>(later.counts[1]) - first.counts[1]) /
                        static_cast<double>(first.counts[1]);
  std::printf("  triangles after 45 frames %u, after 60 %u (%.2f%% change)\n", first.counts[1],
              later.counts[1], 100.0 * change);
  VERIFY_MSG(change < 0.05, "the triangle count settles");
}

void check_sphere_watertight() {
  TsdfVolume volume(scene_volume(SceneKind::kSphere, 96, false));
  constexpr std::uint32_t kOrbit = 120;
  test::integrate_orbit(volume, SceneKind::kSphere, 0, kOrbit, kOrbit, kWidth, kHeight,
                        test::kNoise);
  const test::HostMesh mesh = test::extract(volume);
  const std::size_t vertices = mesh.vertices.size() / 3;
  const std::size_t triangles = mesh.triangles.size() / 3;
  std::printf("  sphere: %zu vertices, %zu triangles\n", vertices, triangles);
  VERIFY(triangles > 5000);
  // Every edge in exactly two triangles, once in each direction; every vertex
  // used; no degenerate triangle.
  std::map<std::pair<std::uint32_t, std::uint32_t>, int> directed;
  std::vector<bool> used(vertices, false);
  std::size_t degenerate = 0;
  for (std::size_t t = 0; t < triangles; ++t) {
    const std::uint32_t* tri = &mesh.triangles[3 * t];
    degenerate += tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] ? 1 : 0;
    for (int e = 0; e < 3; ++e) {
      ++directed[{tri[e], tri[(e + 1) % 3]}];
      used[tri[e]] = true;
    }
  }
  std::size_t bad_edges = 0;
  for (const auto& [edge, count] : directed) {
    const auto reverse = directed.find({edge.second, edge.first});
    bad_edges += count != 1 || reverse == directed.end() || reverse->second != 1 ? 1 : 0;
  }
  const std::size_t edges = directed.size() / 2;
  const std::int64_t euler = static_cast<std::int64_t>(vertices) -
                             static_cast<std::int64_t>(edges) +
                             static_cast<std::int64_t>(triangles);
  std::printf("  edges %zu, bad edges %zu, Euler characteristic %" PRId64 ", degenerate %zu\n",
              edges, bad_edges, euler, degenerate);
  VERIFY_MSG(bad_edges == 0, "every edge in two triangles of opposite direction: watertight");
  VERIFY_MSG(euler == 2, "the mesh is one closed surface of genus 0");
  VERIFY_MSG(degenerate == 0, "no triangle repeats a vertex");
  VERIFY_MSG(std::all_of(used.begin(), used.end(), [](bool u) { return u; }),
             "every vertex is used");
  const SyntheticScene scene = synthetic_scene(SceneKind::kSphere);
  const ErrorStats error = test::surface_error(scene, mesh.vertices);
  print("sphere vertices", error, volume.config().voxel_size);
  VERIFY(error.max < 2.0 * volume.config().voxel_size);
}

void check_raycast() {
  TsdfVolume volume(scene_volume(SceneKind::kTabletop, 160, true));
  test::integrate_orbit(volume, SceneKind::kTabletop, 0, kFrames, kFrames, kWidth, kHeight,
                        test::kNoise);
  const SyntheticScene scene = synthetic_scene(SceneKind::kTabletop);
  const TsdfVolumeConfig& c = volume.config();
  // From poses of the orbit and from poses between them.
  for (const float at : {0.0F, 10.5F, 31.25F}) {
    const Matrix4 pose =
        orbit_pose(SceneKind::kTabletop, static_cast<std::uint32_t>(at * 4), kFrames * 4);
    const RaycastParams params{.intrinsics = synthetic_intrinsics(kWidth, kHeight),
                               .width = kWidth,
                               .height = kHeight,
                               .depth_min = 0.1F,
                               .depth_max = 3.0F};
    const std::size_t pixels = static_cast<std::size_t>(kWidth) * kHeight;
    Device<float> depth(pixels);
    Device<float> normals(pixels * 3);
    Device<std::uint8_t> colors(pixels * 3);
    cuda_check(volume.raycast(pose, params, RaycastImages{depth.get(), normals.get(), colors.get()},
                              nullptr),
               "raycast");
    const std::vector<float> got = depth.host();
    const std::vector<float> got_normals = normals.host();
    std::vector<double> errors;
    std::size_t expected_hits = 0;
    std::size_t hits = 0;
    std::size_t spurious = 0;
    std::size_t normals_agree = 0;
    for (int v = 0; v < kHeight; ++v) {
      for (int u = 0; u < kWidth; ++u) {
        const std::size_t i = static_cast<std::size_t>(v) * kWidth + u;
        const float truth =
            scene_depth(scene, pose.data(), params.intrinsics, static_cast<float>(u),
                        static_cast<float>(v), 3.0F, nullptr);
        // Only surfaces inside the volume, away from its faces, can be seen.
        bool inside = false;
        if (truth > 0.0F) {
          const float ray[3] = {(u - params.intrinsics.cx) / params.intrinsics.fx * truth,
                                (v - params.intrinsics.cy) / params.intrinsics.fy * truth, truth};
          inside = true;
          for (int r = 0; r < 3; ++r) {
            const float p = pose[4 * r] * ray[0] + pose[4 * r + 1] * ray[1] +
                            pose[4 * r + 2] * ray[2] + pose[4 * r + 3];
            const float lo = c.origin[r] + 3.0F * c.voxel_size;
            const float hi = c.origin[r] + (c.dims[r] - 4) * c.voxel_size;
            inside = inside && p > lo && p < hi;
          }
        }
        if (inside) {
          ++expected_hits;
          if (got[i] > 0.0F) {
            ++hits;
            errors.push_back(std::abs(got[i] - truth));
            // The camera-frame normal faces the camera.
            const float ray[3] = {(u - params.intrinsics.cx) / params.intrinsics.fx,
                                  (v - params.intrinsics.cy) / params.intrinsics.fy, 1.0F};
            normals_agree += got_normals[3 * i] * ray[0] + got_normals[3 * i + 1] * ray[1] +
                                         got_normals[3 * i + 2] * ray[2] <
                                     0.0F
                                 ? 1
                                 : 0;
          }
        } else if (truth == 0.0F && got[i] > 0.0F) {
          ++spurious;
        }
      }
    }
    const ErrorStats s = test::stats(errors);
    std::printf("  ray cast from orbit position %.2f: coverage %.2f%%, spurious %zu\n", at,
                100.0 * hits / std::max<std::size_t>(expected_hits, 1), spurious);
    print("ray-cast depth", s, c.voxel_size);
    VERIFY_MSG(hits > expected_hits * 97 / 100, "the ray cast sees what the camera would");
    VERIFY_MSG(s.mean < 0.5 * c.voxel_size, "ray-cast depth: mean error");
    VERIFY_MSG(s.p95 < 1.5 * c.voxel_size, "ray-cast depth: p95 error");
    VERIFY_MSG(normals_agree > hits * 99 / 100, "ray-cast normals face the camera");
    VERIFY_MSG(spurious < expected_hits / 1000 + 5, "rays that hit nothing stay empty");
  }
}

// A camera at the identity pose looking down +z at a wall `depth` meters
// away: every pixel `depth`.
void integrate_wall(TsdfVolume& volume, float depth, int times) {
  constexpr int w = 64;
  constexpr int h = 48;
  Device<float> image(w * h);
  image.upload(std::vector<float>(w * h, depth));
  const IntegrateParams params{
      .intrinsics = synthetic_intrinsics(w, h), .depth_min = 0.1F, .depth_max = 3.0F};
  for (int i = 0; i < times; ++i) {
    cuda_check(volume.integrate(DepthImage{image.get(), DepthType::kFloat32, w, h, 1.0F}, nullptr,
                                identity_pose(), params, nullptr),
               "integrate");
  }
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

// The z where the distance on the camera's axis crosses zero.
float axis_crossing(const TsdfVolume& volume) {
  const TsdfVolumeConfig& c = volume.config();
  const VolumeData data = volume.download();
  const int x = c.dims[0] / 2;
  const int y = c.dims[1] / 2;
  for (int z = 0; z + 1 < c.dims[2]; ++z) {
    const std::size_t i = x + c.dims[0] * (y + static_cast<std::size_t>(c.dims[1]) * z);
    const std::size_t j = i + static_cast<std::size_t>(c.dims[0]) * c.dims[1];
    if (data.weight[i] > 0.0F && data.weight[j] > 0.0F && data.tsdf[i] >= 0.0F &&
        data.tsdf[j] < 0.0F) {
      return c.origin[2] + c.voxel_size * (z + data.tsdf[i] / (data.tsdf[i] - data.tsdf[j]));
    }
  }
  return std::numeric_limits<float>::quiet_NaN();
}

void check_weights_and_truncation() {
  TsdfVolumeConfig config;
  config.dims[0] = config.dims[1] = config.dims[2] = 64;
  config.voxel_size = 0.01F;
  config.origin[0] = config.origin[1] = -0.315F;  // Voxel 31.5 on the axis: off-center.
  config.origin[2] = 0.2F;
  config.truncation = 0.05F;
  config.weight_cap = 4.0F;
  TsdfVolume volume(config);
  integrate_wall(volume, 0.5F, 10);
  const VolumeData data = volume.download();
  float max_weight = 0.0F;
  bool in_range = true;
  bool truncated_exact = true;
  bool behind_untouched = true;
  std::size_t truncated = 0;
  for (int z = 0; z < 64; ++z) {
    const float depth = config.origin[2] + config.voxel_size * z;
    const std::size_t i = 32 + 64 * (32 + std::size_t{64} * z);
    max_weight = std::max(max_weight, data.weight[i]);
    in_range = in_range && data.tsdf[i] >= -1.0F && data.tsdf[i] <= 1.0F;
    if (0.5F - depth > config.truncation + 1e-4F) {
      truncated_exact = truncated_exact && data.tsdf[i] == 1.0F && data.weight[i] == 4.0F;
      ++truncated;
    }
    if (depth - 0.5F > config.truncation + 1e-4F) {
      behind_untouched = behind_untouched && data.weight[i] == 0.0F && data.tsdf[i] == 1.0F;
    }
    if (std::abs(depth - 0.5F) < config.truncation - 1e-4F) {
      const float expected = (0.5F - depth) / config.truncation;
      VERIFY_MSG(std::abs(data.tsdf[i] - expected) < 1e-5F && data.weight[i] == 4.0F,
                 "inside the band, the distance over the truncation, at the cap");
    }
  }
  VERIFY_MSG(max_weight == 4.0F, "weights stop at the cap");
  VERIFY_MSG(*std::max_element(data.weight.begin(), data.weight.end()) == 4.0F,
             "no voxel beyond the cap");
  VERIFY_MSG(in_range, "distances stay within [-1, 1]");
  VERIFY(truncated > 10);
  VERIFY_MSG(truncated_exact, "in front of the band, exactly 1");
  VERIFY_MSG(behind_untouched, "behind the band, never written");
  VERIFY(std::abs(axis_crossing(volume) - 0.5F) < 1e-3F);

  // With the weight capped, the surface follows a moved wall quickly; with a
  // high cap it moves only in proportion to the images.
  integrate_wall(volume, 0.53F, 10);
  const float capped = axis_crossing(volume);
  TsdfVolumeConfig high = config;
  high.weight_cap = 1000.0F;
  TsdfVolume slow(high);
  integrate_wall(slow, 0.5F, 10);
  integrate_wall(slow, 0.53F, 10);
  const float averaged = axis_crossing(slow);
  std::printf("  wall moved 0.50 -> 0.53 m: capped at 4 the surface is at %.4f m, at 1000 %.4f m\n",
              capped, averaged);
  VERIFY_MSG(capped > 0.525F && capped < 0.531F, "a capped volume follows the new surface");
  VERIFY_MSG(std::abs(averaged - 0.515F) < 0.002F, "an uncapped volume averages the two");
}

void check_cpu_reference() {
  for (const bool color : {false, true}) {
    TsdfVolumeConfig config = scene_volume(SceneKind::kTabletop, 40, color);
    config.weight_cap = 3.0F;  // So the cap is reached.
    TsdfVolume volume(config);
    reference::Volume host(config);
    Frame f(160, 120);
    for (std::uint32_t i = 0; i < 5; ++i) {
      test::render(f, SceneKind::kTabletop, i * 3, 15, test::kNoise);
      const ColorImage rgb{f.rgb.get(), 3, false};
      cuda_check(
          volume.integrate(test::depth_image(f), &rgb, f.pose, test::integrate_params(f), nullptr),
          "integrate");
      reference::integrate(host, f.host_depth, f.width, f.height, f.host_rgb, f.pose,
                           test::integrate_params(f));
    }
    const VolumeData gpu = volume.download();
    const std::string kind = color ? " (color)" : "";
    VERIFY_MSG(std::memcmp(gpu.tsdf.data(), host.data.tsdf.data(), gpu.tsdf.size() * 4) == 0,
               "integration matches the CPU reference bit for bit" + kind);
    VERIFY_MSG(gpu.weight == host.data.weight, "weights match the CPU reference" + kind);
    VERIFY_MSG(gpu.rgb == host.data.rgb, "colors match the CPU reference" + kind);
    std::size_t observed = 0;
    for (const float w : gpu.weight) {
      observed += w > 0.0F ? 1 : 0;
    }
    VERIFY(observed > 5000);

    const test::HostMesh mesh = test::extract(volume);
    const reference::Mesh want = reference::marching_cubes(host);
    std::printf("  reference%s: %zu observed voxels, %zu vertices, %zu triangles\n", kind.c_str(),
                observed, want.vertices.size() / 3, want.triangles.size() / 3);
    VERIFY(want.triangles.size() > 300);
    VERIFY_MSG(mesh.counts[kCountVertices] * 3 == want.vertices.size() &&
                   mesh.counts[kCountTriangles] * 3 == want.triangles.size(),
               "mesh counts match the CPU reference" + kind);
    VERIFY_MSG(
        mesh.vertices.size() == want.vertices.size() &&
            std::memcmp(mesh.vertices.data(), want.vertices.data(), want.vertices.size() * 4) == 0,
        "vertices match the CPU reference bit for bit" + kind);
    VERIFY_MSG(
        mesh.normals.size() == want.normals.size() &&
            std::memcmp(mesh.normals.data(), want.normals.data(), want.normals.size() * 4) == 0,
        "normals match the CPU reference bit for bit" + kind);
    VERIFY_MSG(mesh.colors == want.colors, "vertex colors match the CPU reference" + kind);
    VERIFY_MSG(mesh.triangles == want.triangles, "triangles match the CPU reference" + kind);
  }
}

void check_determinism() {
  const auto run = [] {
    TsdfVolume volume(scene_volume(SceneKind::kTabletop, 96, true));
    test::integrate_orbit(volume, SceneKind::kTabletop, 0, 20, 20, kWidth, kHeight, test::kNoise);
    const RaycastParams params{
        .intrinsics = synthetic_intrinsics(kWidth, kHeight), .width = kWidth, .height = kHeight};
    const std::size_t pixels = static_cast<std::size_t>(kWidth) * kHeight;
    Device<float> depth(pixels);
    Device<float> normals(pixels * 3);
    cuda_check(volume.raycast(orbit_pose(SceneKind::kTabletop, 3, 20), params,
                              RaycastImages{depth.get(), normals.get(), nullptr}, nullptr),
               "raycast");
    return std::tuple{volume.download(), test::extract(volume), depth.host(), normals.host()};
  };
  const auto [volume_a, mesh_a, depth_a, normals_a] = run();
  const auto [volume_b, mesh_b, depth_b, normals_b] = run();
  VERIFY_MSG(
      std::memcmp(volume_a.tsdf.data(), volume_b.tsdf.data(), volume_a.tsdf.size() * 4) == 0 &&
          volume_a.weight == volume_b.weight && volume_a.rgb == volume_b.rgb,
      "the same images give the same volume");
  VERIFY_MSG(mesh_a.vertices.size() == mesh_b.vertices.size() &&
                 std::memcmp(mesh_a.vertices.data(), mesh_b.vertices.data(),
                             mesh_a.vertices.size() * 4) == 0 &&
                 mesh_a.triangles == mesh_b.triangles && mesh_a.colors == mesh_b.colors,
             "the same volume gives the same mesh");
  VERIFY_MSG(std::memcmp(depth_a.data(), depth_b.data(), depth_a.size() * 4) == 0 &&
                 std::memcmp(normals_a.data(), normals_b.data(), normals_a.size() * 4) == 0,
             "the same volume gives the same ray cast");
}

std::size_t observed_voxels(const TsdfVolume& volume) {
  const VolumeData data = volume.download();
  return static_cast<std::size_t>(
      std::count_if(data.weight.begin(), data.weight.end(), [](float w) { return w > 0.0F; }));
}

void check_edge_cases() {
  const TsdfVolumeConfig config = scene_volume(SceneKind::kTabletop, 64, true);
  constexpr int w = 160;
  constexpr int h = 120;
  const CameraIntrinsics k = synthetic_intrinsics(w, h);
  const IntegrateParams params{.intrinsics = k, .depth_min = 0.1F, .depth_max = 3.0F};
  const Matrix4 pose = orbit_pose(SceneKind::kTabletop, 0, 10);
  Device<float> depth(w * h);
  {
    // No depth at all: nothing changes, no mesh, the ray cast sees nothing.
    TsdfVolume volume(config);
    depth.upload(std::vector<float>(w * h, 0.0F));
    cuda_check(volume.integrate(DepthImage{depth.get(), DepthType::kFloat32, w, h, 1.0F}, nullptr,
                                pose, params, nullptr),
               "integrate");
    VERIFY_MSG(observed_voxels(volume) == 0, "empty depth changes nothing");
    const test::HostMesh mesh = test::extract(volume);
    VERIFY_MSG(
        mesh.counts[0] == 0 && mesh.counts[1] == 0 && mesh.counts[2] == 0 && mesh.counts[3] == 0,
        "an empty volume has an empty mesh");
    Device<float> ray(w * h);
    Device<float> normals(w * h * 3);
    cuda_check(volume.raycast(pose, RaycastParams{.intrinsics = k, .width = w, .height = h},
                              RaycastImages{ray.get(), normals.get(), nullptr}, nullptr),
               "raycast");
    const std::vector<float> rays = ray.host();
    const std::vector<float> n = normals.host();
    VERIFY_MSG(std::all_of(rays.begin(), rays.end(), [](float d) { return d == 0.0F; }) &&
                   std::all_of(n.begin(), n.end(), [](float v) { return std::isnan(v); }),
               "an empty volume ray casts to nothing");
    // A zero-size image is accepted and does nothing.
    cuda_check(volume.integrate(DepthImage{nullptr, DepthType::kFloat32, 0, 0, 1.0F}, nullptr, pose,
                                params, nullptr),
               "integrate empty image");
    VERIFY(observed_voxels(volume) == 0);
  }
  {
    // Every depth out of range: nothing changes. Non-finite depths neither.
    TsdfVolume volume(config);
    std::vector<float> values(w * h, 3.5F);
    for (std::size_t i = 0; i < values.size(); i += 3) {
      values[i] = i % 2 == 0 ? std::numeric_limits<float>::quiet_NaN()
                             : std::numeric_limits<float>::infinity();
    }
    values[1] = 0.05F;  // Nearer than depth_min.
    depth.upload(values);
    cuda_check(volume.integrate(DepthImage{depth.get(), DepthType::kFloat32, w, h, 1.0F}, nullptr,
                                pose, params, nullptr),
               "integrate");
    VERIFY_MSG(observed_voxels(volume) == 0, "depths out of range change nothing");
  }
  {
    // The camera inside the volume, looking at the sphere from 35 cm: the
    // voxels behind it are untouched, and the ray cast from there sees the
    // sphere where it is.
    TsdfVolume volume(config);
    const SyntheticScene scene = synthetic_scene(SceneKind::kTabletop);
    const float eye[3] = {scene.sphere_center[0] - 0.35F, scene.sphere_center[1],
                          scene.sphere_center[2] + 0.05F};
    const Matrix4 inside = look_at(eye, scene.sphere_center);
    cuda_check(render_scene(scene, inside, k, w, h, 3.0F, DepthNoise{}, DepthType::kFloat32,
                            depth.get(), nullptr, nullptr),
               "render");
    for (int i = 0; i < 3; ++i) {
      cuda_check(volume.integrate(DepthImage{depth.get(), DepthType::kFloat32, w, h, 1.0F}, nullptr,
                                  inside, params, nullptr),
                 "integrate");
    }
    const VolumeData data = volume.download();
    const Matrix4 camera_from_world = inverse_pose(inside);
    std::size_t observed = 0;
    std::size_t behind = 0;
    for (int z = 0; z < config.dims[2]; ++z) {
      for (int y = 0; y < config.dims[1]; ++y) {
        for (int x = 0; x < config.dims[0]; ++x) {
          const std::size_t i = x + config.dims[0] * (y + std::size_t{64} * z);
          if (!(data.weight[i] > 0.0F)) {
            continue;
          }
          ++observed;
          const float p[3] = {config.origin[0] + config.voxel_size * x,
                              config.origin[1] + config.voxel_size * y,
                              config.origin[2] + config.voxel_size * z};
          const float zc = camera_from_world[8] * p[0] + camera_from_world[9] * p[1] +
                           camera_from_world[10] * p[2] + camera_from_world[11];
          behind += zc <= 0.0F ? 1 : 0;
        }
      }
    }
    VERIFY(observed > 1000);
    VERIFY_MSG(behind == 0, "nothing behind the camera is written");
    Device<float> ray(w * h);
    cuda_check(volume.raycast(inside, RaycastParams{.intrinsics = k, .width = w, .height = h},
                              RaycastImages{ray.get(), nullptr, nullptr}, nullptr),
               "raycast");
    const std::vector<float> rays = ray.host();
    const std::vector<float> truth = depth.host();
    std::vector<double> errors;
    for (std::size_t i = 0; i < rays.size(); ++i) {
      if (rays[i] > 0.0F && truth[i] > 0.0F) {
        errors.push_back(std::abs(rays[i] - truth[i]));
      }
    }
    const ErrorStats s = test::stats(errors);
    print("ray cast from inside the volume", s, config.voxel_size);
    VERIFY(s.count > 5000);
    VERIFY(s.p95 < 1.5 * config.voxel_size);
  }
  {
    // Unusable arguments are refused.
    TsdfVolume volume(config);
    Matrix4 skewed = pose;
    skewed[0] += 0.5F;
    VERIFY(volume.integrate(DepthImage{depth.get(), DepthType::kFloat32, w, h, 1.0F}, nullptr,
                            skewed, params, nullptr) == cudaErrorInvalidValue);
    VERIFY(volume.integrate(DepthImage{depth.get(), DepthType::kFloat32, w, h, 0.0F}, nullptr, pose,
                            params, nullptr) == cudaErrorInvalidValue);
    Device<std::uint8_t> rgb(w * h * 3);
    VERIFY(volume.raycast(pose, RaycastParams{.intrinsics = k, .width = w, .height = h},
                          RaycastImages{nullptr, nullptr, rgb.get()}, nullptr) == cudaSuccess);
    TsdfVolume plain(scene_volume(SceneKind::kTabletop, 32, false));
    VERIFY_MSG(
        plain.raycast(pose, RaycastParams{.intrinsics = k, .width = w, .height = h},
                      RaycastImages{nullptr, nullptr, rgb.get()}, nullptr) == cudaErrorInvalidValue,
        "colors need a color volume");
  }
}

void check_capacity_and_reset() {
  TsdfVolume volume(scene_volume(SceneKind::kTabletop, 64, false));
  test::integrate_orbit(volume, SceneKind::kTabletop, 0, 5, 20, 160, 120, test::kNoise);
  const test::HostMesh full = test::extract(volume);
  VERIFY(full.counts[1] > 100);
  const test::HostMesh small = test::extract(volume, 1U << 20, 10);
  VERIFY_MSG(small.counts[kCountVertices] == 0 && small.counts[kCountTriangles] == 0,
             "a mesh beyond the capacity is not written");
  VERIFY_MSG(small.counts[kCountNeededTriangles] == full.counts[1] &&
                 small.counts[kCountNeededVertices] == full.counts[0],
             "the counts say what the mesh needs");
  const test::HostMesh few_vertices = test::extract(volume, 10, 1U << 20);
  VERIFY(few_vertices.counts[0] == 0 && few_vertices.counts[1] == 0 &&
         few_vertices.counts[kCountNeededVertices] == full.counts[0]);
  // An extractor reused after a capacity overflow still works.
  VERIFY(test::extract(volume).triangles == full.triangles);

  cuda_check(volume.reset(nullptr), "reset");
  const VolumeData data = volume.download();
  VERIFY_MSG(
      std::all_of(data.weight.begin(), data.weight.end(), [](float w) { return w == 0.0F; }) &&
          std::all_of(data.tsdf.begin(), data.tsdf.end(), [](float t) { return t == 1.0F; }),
      "reset clears every voxel");
  VERIFY(test::extract(volume).counts[1] == 0);
}

void check_ply() {
  TsdfVolume volume(scene_volume(SceneKind::kTabletop, 64, true));
  test::integrate_orbit(volume, SceneKind::kTabletop, 0, 8, 8, 160, 120, test::kNoise);
  const test::HostMesh mesh = test::extract(volume);
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / ("tsdf_test_" + std::to_string(::getpid()) + ".ply");
  volume.export_ply(path.string());
  std::ifstream file(path, std::ios::binary);
  std::string line;
  std::size_t vertices = 0;
  std::size_t faces = 0;
  bool colored = false;
  std::size_t header = 0;
  while (std::getline(file, line)) {
    header += line.size() + 1;
    if (line.rfind("element vertex ", 0) == 0) {
      vertices = std::stoul(line.substr(15));
    } else if (line.rfind("element face ", 0) == 0) {
      faces = std::stoul(line.substr(13));
    } else if (line == "property uchar red") {
      colored = true;
    } else if (line == "end_header") {
      break;
    }
  }
  file.close();
  const auto size = std::filesystem::file_size(path);
  std::filesystem::remove(path);
  VERIFY_MSG(vertices == mesh.counts[0] && faces == mesh.counts[1],
             "the PLY file has the mesh's vertices and triangles");
  VERIFY(colored);
  VERIFY_MSG(size == header + vertices * 27 + faces * 13, "the PLY file has every record");
  VERIFY(test::throws<std::runtime_error>(
      [&volume] { volume.export_ply("/nonexistent-folder/mesh.ply"); }));
}

void check_settings() {
  const auto refused = [](auto change) {
    TsdfVolumeConfig config;
    change(config);
    return test::throws<std::invalid_argument>([&config] { validate_config(config); });
  };
  VERIFY(refused([](TsdfVolumeConfig& c) { c.dims[0] = 1; }));
  VERIFY(refused([](TsdfVolumeConfig& c) { c.dims[2] = 1025; }));
  VERIFY(refused([](TsdfVolumeConfig& c) { c.voxel_size = 0.0F; }));
  VERIFY(refused([](TsdfVolumeConfig& c) { c.truncation = 0.5F * c.voxel_size; }));
  VERIFY(refused([](TsdfVolumeConfig& c) { c.weight_cap = 0.5F; }));
  VERIFY(
      refused([](TsdfVolumeConfig& c) { c.origin[1] = std::numeric_limits<float>::infinity(); }));
  VERIFY(!refused([](TsdfVolumeConfig&) {}));
  TsdfVolumeConfig two_g;
  two_g.dims[0] = two_g.dims[1] = two_g.dims[2] = 1024;
  VERIFY(!test::throws<std::invalid_argument>([&two_g] { validate_config(two_g); }));
  VERIFY(volume_bytes(two_g) == (std::uint64_t{1} << 33));
  VERIFY(test::throws<std::invalid_argument>([] { MeshExtractor(0, 10); }));
  VERIFY(test::throws<std::invalid_argument>([] { MeshExtractor(10, kMaxMeshCapacity + 1); }));
  VERIFY(test::throws<std::invalid_argument>(
      [] { validate_params(IntegrateParams{.intrinsics = {0.0F, 1.0F, 0.0F, 0.0F}}); }));
  VERIFY(test::throws<std::invalid_argument>(
      [] { validate_params(RaycastParams{.intrinsics = {1.0F, 1.0F, 0.0F, 0.0F}, .width = 0}); }));
  VERIFY(valid_pose(orbit_pose(SceneKind::kSphere, 7, 30)));
  const Matrix4 pose = orbit_pose(SceneKind::kTabletop, 3, 30);
  const Matrix4 inverse = inverse_pose(pose);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      double sum = 0.0;
      for (int k = 0; k < 4; ++k) {
        sum += static_cast<double>(pose[4 * r + k]) * inverse[4 * k + c];
      }
      VERIFY(std::abs(sum - (r == c ? 1.0 : 0.0)) < 1e-5);
    }
  }
}

}  // namespace

int run_all() {
  return test::run({
      {"surface accuracy (zero crossings, vertices, normals, winding, colors)",
       check_surface_accuracy},
      {"mesh stability", check_mesh_stability},
      {"closed sphere is watertight", check_sphere_watertight},
      {"ray cast against the exact depth", check_raycast},
      {"weight cap and truncation", check_weights_and_truncation},
      {"CPU reference", check_cpu_reference},
      {"determinism", check_determinism},
      {"edge cases", check_edge_cases},
      {"mesh capacity and reset", check_capacity_and_reset},
      {"PLY export", check_ply},
      {"settings", check_settings},
  });
}

}  // namespace holoscan::tsdf

int main() { return holoscan::tsdf::run_all(); }
