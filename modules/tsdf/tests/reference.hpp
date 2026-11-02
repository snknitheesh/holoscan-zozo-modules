// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CPU references of the integration and the marching cubes, written from
// their definitions and following the kernels' arithmetic step by step, so
// that with no contraction on either side (--fmad=false, -ffp-contract=off)
// they give the same bits. For small volumes.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <holoscan/tsdf/marching_cubes.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>

#include "mc_tables.hpp"

namespace holoscan::tsdf::reference {

// A volume on the host, as TsdfVolume::download() gives it.
struct Volume {
  TsdfVolumeConfig config;
  VolumeData data;

  explicit Volume(const TsdfVolumeConfig& c) : config(c) {
    const std::size_t n = voxels();
    data.tsdf.assign(n, 1.0F);
    data.weight.assign(n, 0.0F);
    if (config.color) {
      data.rgb.assign(3 * n, 0U);
    }
  }
  [[nodiscard]] std::size_t voxels() const {
    return static_cast<std::size_t>(config.dims[0]) * config.dims[1] * config.dims[2];
  }
  [[nodiscard]] std::size_t index(int x, int y, int z) const {
    return static_cast<std::size_t>(x) +
           static_cast<std::size_t>(config.dims[0]) *
               (static_cast<std::size_t>(y) + static_cast<std::size_t>(config.dims[1]) * z);
  }
};

// Integrates a host depth image (and RGB color, may be empty) as
// TsdfVolume::integrate() does, visiting every voxel.
inline void integrate(Volume& volume, const std::vector<float>& depth_m, int width, int height,
                      const std::vector<std::uint8_t>& rgb, const Matrix4& world_from_camera,
                      const IntegrateParams& params) {
  const TsdfVolumeConfig& c = volume.config;
  const Matrix4 m = inverse_pose(world_from_camera);
  const CameraIntrinsics& k = params.intrinsics;
  for (int z = 0; z < c.dims[2]; ++z) {
    for (int y = 0; y < c.dims[1]; ++y) {
      for (int x = 0; x < c.dims[0]; ++x) {
        const float px = c.origin[0] + c.voxel_size * static_cast<float>(x);
        const float py = c.origin[1] + c.voxel_size * static_cast<float>(y);
        const float pz = c.origin[2] + c.voxel_size * static_cast<float>(z);
        const float xc = m[0] * px + m[1] * py + m[2] * pz + m[3];
        const float yc = m[4] * px + m[5] * py + m[6] * pz + m[7];
        const float zc = m[8] * px + m[9] * py + m[10] * pz + m[11];
        if (!(zc > 0.0F)) {
          continue;
        }
        const float u = k.fx * xc / zc + k.cx;
        const float v = k.fy * yc / zc + k.cy;
        const float uf = std::floor(u + 0.5F);
        const float vf = std::floor(v + 0.5F);
        if (!(uf >= 0.0F && uf < static_cast<float>(width) && vf >= 0.0F &&
              vf < static_cast<float>(height))) {
          continue;
        }
        const std::size_t pixel =
            static_cast<std::size_t>(vf) * width + static_cast<std::size_t>(uf);
        const float d = depth_m[pixel] * 1.0F;
        if (!(d > 0.0F && d >= params.depth_min && d <= params.depth_max)) {
          continue;
        }
        const float sdf = d - zc;
        if (sdf < -c.truncation) {
          continue;
        }
        const float t = std::min(1.0F, sdf / c.truncation);
        const std::size_t i = volume.index(x, y, z);
        const float w = volume.data.weight[i];
        const float w1 = w + 1.0F;
        volume.data.tsdf[i] = (volume.data.tsdf[i] * w + t) / w1;
        volume.data.weight[i] = std::min(w1, c.weight_cap);
        if (c.color && !rgb.empty()) {
          for (int ch = 0; ch < 3; ++ch) {
            const float before = static_cast<float>(volume.data.rgb[3 * i + ch]);
            const float mixed = (before * w + static_cast<float>(rgb[3 * pixel + ch])) / w1;
            volume.data.rgb[3 * i + ch] =
                static_cast<std::uint8_t>(std::min(255.0F, std::floor(mixed + 0.5F)));
          }
        }
      }
    }
  }
}

struct Mesh {
  std::vector<float> vertices;
  std::vector<float> normals;
  std::vector<std::uint8_t> colors;
  std::vector<std::uint32_t> triangles;
};

// The mesh MeshExtractor gives, by the same definitions, in the same order.
inline Mesh marching_cubes(const Volume& volume) {
  constexpr int kCorners[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                  {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  constexpr int kEdges[12][4] = {{0, 0, 0, 0}, {1, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 0, 1},
                                 {0, 0, 1, 0}, {1, 0, 1, 1}, {0, 1, 1, 0}, {0, 0, 1, 1},
                                 {0, 0, 0, 2}, {1, 0, 0, 2}, {1, 1, 0, 2}, {0, 1, 0, 2}};
  const TsdfVolumeConfig& c = volume.config;
  const int* n = c.dims;
  const auto& tsdf = volume.data.tsdf;
  const auto& weight = volume.data.weight;
  const auto observed = [&](int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || x >= n[0] - 1 || y >= n[1] - 1 || z >= n[2] - 1) {
      return false;
    }
    for (const auto& corner : kCorners) {
      if (!(weight[volume.index(x + corner[0], y + corner[1], z + corner[2])] > 0.0F)) {
        return false;
      }
    }
    return true;
  };
  const auto at = [&](int x, int y, int z) {
    x = std::clamp(x, 0, n[0] - 1);
    y = std::clamp(y, 0, n[1] - 1);
    z = std::clamp(z, 0, n[2] - 1);
    return tsdf[volume.index(x, y, z)];
  };
  const auto gradient = [&](int x, int y, int z, float g[3]) {
    g[0] = at(x + 1, y, z) - at(x - 1, y, z);
    g[1] = at(x, y + 1, z) - at(x, y - 1, z);
    g[2] = at(x, y, z + 1) - at(x, y, z - 1);
  };
  // Vertex of edge (voxel, axis), in owner order.
  std::vector<std::int64_t> vertex_of(volume.voxels() * 3, -1);
  Mesh mesh;
  for (int z = 0; z < n[2]; ++z) {
    for (int y = 0; y < n[1]; ++y) {
      for (int x = 0; x < n[0]; ++x) {
        const std::size_t i = volume.index(x, y, z);
        if (!(weight[i] > 0.0F)) {
          continue;
        }
        const int voxel[3] = {x, y, z};
        for (int axis = 0; axis < 3; ++axis) {
          if (voxel[axis] + 1 >= n[axis]) {
            continue;
          }
          const int next[3] = {x + (axis == 0), y + (axis == 1), z + (axis == 2)};
          const std::size_t j = volume.index(next[0], next[1], next[2]);
          if (!(weight[j] > 0.0F) || (tsdf[j] < 0.0F) == (tsdf[i] < 0.0F)) {
            continue;
          }
          const int a1 = (axis + 1) % 3;
          const int a2 = (axis + 2) % 3;
          bool used = false;
          for (int k = 0; k < 4 && !used; ++k) {
            int cube[3] = {x, y, z};
            cube[a1] -= k & 1;
            cube[a2] -= k >> 1;
            used = observed(cube[0], cube[1], cube[2]);
          }
          if (!used) {
            continue;
          }
          vertex_of[3 * i + axis] = static_cast<std::int64_t>(mesh.vertices.size() / 3);
          const float t0 = tsdf[i];
          const float t1 = tsdf[j];
          const float s = t0 / (t0 - t1);
          for (int r = 0; r < 3; ++r) {
            const float offset =
                r == axis ? static_cast<float>(voxel[r]) + s : static_cast<float>(voxel[r]);
            mesh.vertices.push_back(c.origin[r] + c.voxel_size * offset);
          }
          float g0[3];
          float g1[3];
          gradient(x, y, z, g0);
          gradient(next[0], next[1], next[2], g1);
          float nn[3];
          for (int r = 0; r < 3; ++r) {
            nn[r] = g0[r] + (g1[r] - g0[r]) * s;
          }
          const float length = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
          for (int r = 0; r < 3; ++r) {
            mesh.normals.push_back(length > 0.0F ? nn[r] / length : 0.0F);
          }
          if (c.color) {
            for (int r = 0; r < 3; ++r) {
              const float ca = volume.data.rgb[3 * i + r];
              const float cb = volume.data.rgb[3 * j + r];
              mesh.colors.push_back(static_cast<std::uint8_t>(
                  std::min(255.0F, std::floor(ca + (cb - ca) * s + 0.5F))));
            }
          }
        }
      }
    }
  }
  for (int z = 0; z + 1 < n[2]; ++z) {
    for (int y = 0; y + 1 < n[1]; ++y) {
      for (int x = 0; x + 1 < n[0]; ++x) {
        if (!observed(x, y, z)) {
          continue;
        }
        int cube_case = 0;
        for (int k = 0; k < 8; ++k) {
          if (tsdf[volume.index(x + kCorners[k][0], y + kCorners[k][1], z + kCorners[k][2])] <
              0.0F) {
            cube_case |= 1 << k;
          }
        }
        const auto& row = detail::kTriangleTable[cube_case];
        for (int t = 0; t < 5 && row[3 * t] >= 0; ++t) {
          std::uint32_t corner[3];
          for (int e = 0; e < 3; ++e) {
            const int* edge = kEdges[row[3 * t + e]];
            const std::size_t owner = volume.index(x + edge[0], y + edge[1], z + edge[2]);
            corner[e] = static_cast<std::uint32_t>(vertex_of[3 * owner + edge[3]]);
          }
          // Counterclockwise seen from outside.
          mesh.triangles.insert(mesh.triangles.end(), {corner[0], corner[2], corner[1]});
        }
      }
    }
  }
  return mesh;
}

}  // namespace holoscan::tsdf::reference
