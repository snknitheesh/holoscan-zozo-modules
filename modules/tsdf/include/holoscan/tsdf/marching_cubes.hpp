// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// MeshExtractor: the zero surface of a TsdfVolume as a triangle mesh, by
// marching cubes on the GPU, into buffers of a fixed capacity, so that
// nothing waits for the GPU to know the mesh's size.
//
// A cube of eight neighboring voxels gives triangles when all eight have been
// observed (weight above 0) and the distance changes sign among them. The
// mesh is indexed: one vertex on each voxel edge the surface crosses, shared
// by the triangles of the cubes around it, so a surface seen all round is a
// closed mesh. Vertices are ordered by their edge's first voxel (x fastest,
// then y, then z), then x, y, z edge; triangles by their cube, then the order
// of the case table. The result is deterministic.
//
// Each vertex has the unit normal of the distance's gradient (pointing out
// of the surface, toward where it was seen from), interpolated along its
// edge, and with a color volume the color interpolated likewise. Triangles
// wind counterclockwise seen from outside.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

#include <holoscan/tsdf/tsdf_volume.hpp>

namespace holoscan::tsdf {

// The largest capacity accepted, in vertices or triangles.
inline constexpr std::uint32_t kMaxMeshCapacity = std::uint32_t{1} << 28;

// The mesh counts, uint32 [4], in this order.
inline constexpr int kCountVertices = 0;   // Vertices written.
inline constexpr int kCountTriangles = 1;  // Triangles written.
// Vertices and triangles the whole surface has. When either exceeds its
// capacity, nothing is written and the two counts above are 0.
inline constexpr int kCountNeededVertices = 2;
inline constexpr int kCountNeededTriangles = 3;
inline constexpr int kMeshCountsSize = 4;

// Device buffers the mesh is written to.
struct MeshBuffers {
  float* vertices{};           // [max_vertices, 3] x, y, z in the world, meters.
  float* normals{};            // [max_vertices, 3], may be null.
  std::uint8_t* colors{};      // [max_vertices, 3] RGB8, may be null; needs a color volume.
  std::uint32_t* triangles{};  // [max_triangles, 3] vertex indices.
  std::uint32_t* counts{};     // [kMeshCountsSize].
};

class MeshExtractor {
 public:
  // @throws std::invalid_argument if a capacity is not 1..kMaxMeshCapacity.
  MeshExtractor(std::uint32_t max_vertices, std::uint32_t max_triangles);
  ~MeshExtractor();
  MeshExtractor(const MeshExtractor&) = delete;
  MeshExtractor& operator=(const MeshExtractor&) = delete;

  [[nodiscard]] std::uint32_t max_vertices() const noexcept { return max_vertices_; }
  [[nodiscard]] std::uint32_t max_triangles() const noexcept { return max_triangles_; }

  // Extracts the mesh of `volume` into `out` on `stream`. Only the first
  // counts[kCountVertices] vertices and counts[kCountTriangles] triangles are
  // written. Scratch is allocated on the first call for a volume size, which
  // waits for the stream once.
  cudaError_t extract(const TsdfVolume& volume, const MeshBuffers& out, cudaStream_t stream);

  // Device scratch held for a volume of `voxels` voxels.
  [[nodiscard]] static std::uint64_t scratch_bytes(std::uint64_t voxels, std::uint32_t max_vertices,
                                                   std::uint32_t max_triangles);

 private:
  struct Impl;

  std::uint32_t max_vertices_;
  std::uint32_t max_triangles_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::tsdf
