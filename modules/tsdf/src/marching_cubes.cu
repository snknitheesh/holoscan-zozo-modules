// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Marching cubes on the GPU, in three passes over the volume and one over
// the surface's cubes, every result at a position computed by prefix sums,
// so nothing depends on scheduling:
//
//   1. count: each voxel is the first corner of one cube and the first voxel
//      of its three edges along +x, +y and +z. Each block of voxels counts
//      its cubes that give triangles, those triangles, its voxels that own a
//      vertex and those vertices. A CUB scan over the blocks gives each
//      block's first position, and the totals say whether the mesh fits.
//   2. scatter: each block repeats the classification and, by a block scan,
//      writes its cubes (with their case and first triangle) and its vertex
//      owners (with their edges and first vertex), and the vertices.
//   3. triangles: each listed cube writes its triangles, finding the vertex
//      of each edge by a binary search of the owners, which are in voxel
//      order.
//
// Compiled without fused multiply-adds, so that the vertices are bit for bit
// those tests/reference.hpp computes on the host.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <cub/block/block_reduce.cuh>  // NOLINT(build/include_order): CUB
#include <cub/block/block_scan.cuh>    // NOLINT(build/include_order): CUB
#include <cub/device/device_scan.cuh>  // NOLINT(build/include_order): CUB
#include <holoscan/tsdf/marching_cubes.hpp>

#include "device_buffer.hpp"
#include "mc_tables.hpp"

namespace holoscan::tsdf {
namespace {

constexpr int kBlock = 256;

// The case table's triangles wind clockwise seen from where the corners are
// outside (the distance positive); reversing them winds them
// counterclockwise seen from outside.
constexpr bool kReverseWinding = true;

__constant__ std::int8_t c_triangles[256][16] = HOLOSCAN_TSDF_TRIANGLE_TABLE;
// Corner k of a cube, as an offset from its first voxel (kCubeCorners).
__constant__ std::uint8_t c_corners[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
};
// Edge e of a cube (kCubeEdges): the offset of its first voxel from the
// cube's, and its axis.
__constant__ std::uint8_t c_edges[12][4] = {
    {0, 0, 0, 0}, {1, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {1, 0, 1, 1},
    {0, 1, 1, 0}, {0, 0, 1, 1}, {0, 0, 0, 2}, {1, 0, 0, 2}, {1, 1, 0, 2}, {0, 1, 0, 2},
};

struct Counts {
  std::uint32_t cubes;
  std::uint32_t triangles;
  std::uint32_t owners;
  std::uint32_t vertices;
};

__host__ __device__ Counts add(const Counts& a, const Counts& b) {
  return Counts{a.cubes + b.cubes, a.triangles + b.triangles, a.owners + b.owners,
                a.vertices + b.vertices};
}

struct AddCounts {
  __host__ __device__ Counts operator()(const Counts& a, const Counts& b) const {
    return add(a, b);
  }
};

struct State {
  Counts total;
  std::uint32_t fits;
};

struct Volume {
  int n[3];
  float origin[3];
  float voxel;
  const float* tsdf;
  const float* weight;
  const uchar4* rgba;

  __device__ std::uint32_t index(int x, int y, int z) const {
    return static_cast<std::uint32_t>(x) +
           static_cast<std::uint32_t>(n[0]) *
               (static_cast<std::uint32_t>(y) +
                static_cast<std::uint32_t>(n[1]) * static_cast<std::uint32_t>(z));
  }
};

__device__ int triangle_count(int cube_case) {
  int count = 0;
  while (count < 5 && c_triangles[cube_case][3 * count] >= 0) {
    ++count;
  }
  return count;
}

// Whether the cube whose first voxel is (x, y, z) lies in the volume and has
// all eight corners observed.
__device__ bool cube_observed(const Volume& v, int x, int y, int z) {
  if (x < 0 || y < 0 || z < 0 || x >= v.n[0] - 1 || y >= v.n[1] - 1 || z >= v.n[2] - 1) {
    return false;
  }
  for (int k = 0; k < 8; ++k) {
    if (!(v.weight[v.index(x + c_corners[k][0], y + c_corners[k][1], z + c_corners[k][2])] >
          0.0F)) {
      return false;
    }
  }
  return true;
}

struct Voxel {
  int x;
  int y;
  int z;
  std::uint8_t cube_case;  // With `cube`.
  std::uint8_t triangles;  // With `cube`.
  std::uint8_t edges;      // Bit a: the edge along axis a carries a vertex.
  bool cube;               // The cube gives triangles.

  __device__ Counts counts() const {
    return Counts{cube ? 1U : 0U, cube ? triangles : 0U, edges != 0U ? 1U : 0U,
                  static_cast<std::uint32_t>(__popc(edges))};
  }
};

__device__ Voxel classify(const Volume& v, std::uint32_t i, std::uint32_t voxels) {
  Voxel c{};
  if (i >= voxels) {
    return c;
  }
  const auto nx = static_cast<std::uint32_t>(v.n[0]);
  const auto ny = static_cast<std::uint32_t>(v.n[1]);
  c.x = static_cast<int>(i % nx);
  c.y = static_cast<int>((i / nx) % ny);
  c.z = static_cast<int>(i / (nx * ny));
  if (!(v.weight[i] > 0.0F)) {
    return c;  // The first corner of the cube and of the three edges.
  }
  bool own_cube = false;
  if (c.x < v.n[0] - 1 && c.y < v.n[1] - 1 && c.z < v.n[2] - 1) {
    own_cube = true;
    int cube_case = 0;
    for (int k = 0; k < 8; ++k) {
      const std::uint32_t j =
          v.index(c.x + c_corners[k][0], c.y + c_corners[k][1], c.z + c_corners[k][2]);
      if (!(v.weight[j] > 0.0F)) {
        own_cube = false;
        break;
      }
      cube_case |= v.tsdf[j] < 0.0F ? 1 << k : 0;
    }
    if (own_cube && cube_case != 0 && cube_case != 255) {
      c.cube = true;
      c.cube_case = static_cast<std::uint8_t>(cube_case);
      c.triangles = static_cast<std::uint8_t>(triangle_count(cube_case));
    }
  }
  const bool inside = v.tsdf[i] < 0.0F;
  const int at[3] = {c.x, c.y, c.z};
  const std::uint32_t stride[3] = {1U, nx, nx * ny};
  for (int axis = 0; axis < 3; ++axis) {
    if (at[axis] + 1 >= v.n[axis]) {
      continue;
    }
    const std::uint32_t j = i + stride[axis];
    if (!(v.weight[j] > 0.0F) || (v.tsdf[j] < 0.0F) == inside) {
      continue;
    }
    // A vertex where one of the four cubes around the edge is observed:
    // those are the cubes that use it.
    const int a1 = (axis + 1) % 3;
    const int a2 = (axis + 2) % 3;
    bool used = false;
    for (int k = 0; k < 4 && !used; ++k) {
      int cube[3] = {c.x, c.y, c.z};
      cube[a1] -= k & 1;
      cube[a2] -= k >> 1;
      used = k == 0 && own_cube ? true : cube_observed(v, cube[0], cube[1], cube[2]);
    }
    if (used) {
      c.edges |= static_cast<std::uint8_t>(1U << axis);
    }
  }
  return c;
}

__device__ float tsdf_at(const Volume& v, int x, int y, int z) {
  x = min(max(x, 0), v.n[0] - 1);
  y = min(max(y, 0), v.n[1] - 1);
  z = min(max(z, 0), v.n[2] - 1);
  return v.tsdf[v.index(x, y, z)];
}

// The distance's gradient at a voxel by central differences, one-sided at
// the volume's faces, not scaled.
__device__ void gradient(const Volume& v, int x, int y, int z, float g[3]) {
  g[0] = tsdf_at(v, x + 1, y, z) - tsdf_at(v, x - 1, y, z);
  g[1] = tsdf_at(v, x, y + 1, z) - tsdf_at(v, x, y - 1, z);
  g[2] = tsdf_at(v, x, y, z + 1) - tsdf_at(v, x, y, z - 1);
}

__device__ void write_vertex(const Volume& v, const Voxel& c, std::uint32_t i, int axis,
                             std::uint32_t at, const MeshBuffers& out) {
  const auto nx = static_cast<std::uint32_t>(v.n[0]);
  const std::uint32_t stride[3] = {1U, nx, nx * static_cast<std::uint32_t>(v.n[1])};
  const std::uint32_t j = i + stride[axis];
  const float t0 = v.tsdf[i];
  const float t1 = v.tsdf[j];
  const float s = t0 / (t0 - t1);
  const int voxel[3] = {c.x, c.y, c.z};
  for (int r = 0; r < 3; ++r) {
    const float offset =
        r == axis ? static_cast<float>(voxel[r]) + s : static_cast<float>(voxel[r]);
    out.vertices[3 * at + r] = v.origin[r] + v.voxel * offset;
  }
  if (out.normals != nullptr) {
    float g0[3];
    float g1[3];
    gradient(v, c.x, c.y, c.z, g0);
    gradient(v, c.x + (axis == 0 ? 1 : 0), c.y + (axis == 1 ? 1 : 0), c.z + (axis == 2 ? 1 : 0),
             g1);
    float n[3];
    for (int r = 0; r < 3; ++r) {
      n[r] = g0[r] + (g1[r] - g0[r]) * s;
    }
    const float length = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    for (int r = 0; r < 3; ++r) {
      out.normals[3 * at + r] = length > 0.0F ? n[r] / length : 0.0F;
    }
  }
  if (out.colors != nullptr && v.rgba != nullptr) {
    const uchar4 a = v.rgba[i];
    const uchar4 b = v.rgba[j];
    const float ca[3] = {static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z)};
    const float cb[3] = {static_cast<float>(b.x), static_cast<float>(b.y), static_cast<float>(b.z)};
    for (int r = 0; r < 3; ++r) {
      out.colors[3 * at + r] =
          static_cast<std::uint8_t>(fminf(255.0F, floorf(ca[r] + (cb[r] - ca[r]) * s + 0.5F)));
    }
  }
}

__global__ void count_kernel(Volume v, std::uint32_t voxels, Counts* __restrict__ block_counts) {
  using Reduce = cub::BlockReduce<Counts, kBlock>;
  __shared__ typename Reduce::TempStorage storage;
  const std::uint32_t i = blockIdx.x * kBlock + threadIdx.x;
  const Counts mine = classify(v, i, voxels).counts();
  const Counts total = Reduce(storage).Reduce(mine, AddCounts{});
  if (threadIdx.x == 0) {
    block_counts[blockIdx.x] = total;
  }
}

__global__ void totals_kernel(const Counts* __restrict__ offsets,
                              const Counts* __restrict__ block_counts, std::uint32_t blocks,
                              std::uint32_t max_vertices, std::uint32_t max_triangles,
                              State* __restrict__ state, std::uint32_t* __restrict__ counts) {
  const Counts total = add(offsets[blocks - 1], block_counts[blocks - 1]);
  const bool fits = total.vertices <= max_vertices && total.triangles <= max_triangles;
  state->total = total;
  state->fits = fits ? 1U : 0U;
  counts[kCountVertices] = fits ? total.vertices : 0U;
  counts[kCountTriangles] = fits ? total.triangles : 0U;
  counts[kCountNeededVertices] = total.vertices;
  counts[kCountNeededTriangles] = total.triangles;
}

struct Lists {
  std::uint32_t* cubes;
  std::uint8_t* cube_cases;
  std::uint32_t* cube_first_triangle;
  std::uint32_t* owners;
  std::uint8_t* owner_edges;
  std::uint32_t* owner_first_vertex;
};

__global__ void scatter_kernel(Volume v, std::uint32_t voxels, const Counts* __restrict__ offsets,
                               const State* __restrict__ state, Lists lists, MeshBuffers out) {
  if (state->fits == 0U) {
    return;  // The same for every thread of the block.
  }
  using Scan = cub::BlockScan<Counts, kBlock>;
  __shared__ typename Scan::TempStorage storage;
  const std::uint32_t i = blockIdx.x * kBlock + threadIdx.x;
  const Voxel c = classify(v, i, voxels);
  Counts at{};
  Scan(storage).ExclusiveScan(c.counts(), at, Counts{0U, 0U, 0U, 0U}, AddCounts{});
  at = add(at, offsets[blockIdx.x]);
  if (c.cube) {
    lists.cubes[at.cubes] = i;
    lists.cube_cases[at.cubes] = c.cube_case;
    lists.cube_first_triangle[at.cubes] = at.triangles;
  }
  if (c.edges != 0U) {
    lists.owners[at.owners] = i;
    lists.owner_edges[at.owners] = c.edges;
    lists.owner_first_vertex[at.owners] = at.vertices;
    std::uint32_t vertex = at.vertices;
    for (int axis = 0; axis < 3; ++axis) {
      if ((c.edges & (1U << axis)) != 0U) {
        write_vertex(v, c, i, axis, vertex++, out);
      }
    }
  }
}

__global__ void triangles_kernel(Volume v, const State* __restrict__ state, Lists lists,
                                 std::uint32_t* __restrict__ triangles) {
  const std::uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (state->fits == 0U || k >= state->total.cubes) {
    return;
  }
  const std::uint32_t owners = state->total.owners;
  const std::uint32_t cube = lists.cubes[k];
  const int cube_case = lists.cube_cases[k];
  const std::uint32_t first = lists.cube_first_triangle[k];
  const auto nx = static_cast<std::uint32_t>(v.n[0]);
  const auto ny = static_cast<std::uint32_t>(v.n[1]);
  for (int t = 0; t < 5 && c_triangles[cube_case][3 * t] >= 0; ++t) {
    std::uint32_t corner[3];
    for (int e = 0; e < 3; ++e) {
      const std::uint8_t* edge = c_edges[c_triangles[cube_case][3 * t + e]];
      const std::uint32_t owner = cube + edge[0] + nx * (edge[1] + ny * edge[2]);
      // The owner's position in the sorted owner list.
      std::uint32_t lo = 0;
      std::uint32_t hi = owners;
      while (lo < hi) {
        const std::uint32_t mid = lo + (hi - lo) / 2U;
        if (lists.owners[mid] < owner) {
          lo = mid + 1U;
        } else {
          hi = mid;
        }
      }
      const std::uint32_t below = lists.owner_edges[lo] & ((1U << edge[3]) - 1U);
      corner[e] = lists.owner_first_vertex[lo] + static_cast<std::uint32_t>(__popc(below));
    }
    std::uint32_t* out = triangles + 3U * (first + static_cast<std::uint32_t>(t));
    out[0] = corner[0];
    out[1] = kReverseWinding ? corner[2] : corner[1];
    out[2] = kReverseWinding ? corner[1] : corner[2];
  }
}

std::uint32_t blocks_for(std::uint64_t voxels) {
  return static_cast<std::uint32_t>((voxels + kBlock - 1) / kBlock);
}

// Holds the volume for the extraction enqueued on a stream.
class VolumeAccess {
 public:
  VolumeAccess(cudaError_t begun, cudaError_t (*end)(const TsdfVolume&, cudaStream_t),
               const TsdfVolume& volume, cudaStream_t stream)
      : status_(begun), end_(end), volume_(volume), stream_(stream) {}
  ~VolumeAccess() {
    if (status_ == cudaSuccess) {
      end_(volume_, stream_);
    }
  }
  VolumeAccess(const VolumeAccess&) = delete;
  VolumeAccess& operator=(const VolumeAccess&) = delete;
  [[nodiscard]] cudaError_t status() const noexcept { return status_; }

 private:
  cudaError_t status_;
  cudaError_t (*end_)(const TsdfVolume&, cudaStream_t);
  const TsdfVolume& volume_;
  cudaStream_t stream_;
};

}  // namespace

struct MeshExtractor::Impl {
  detail::DeviceBuffer block_counts;
  detail::DeviceBuffer block_offsets;
  detail::DeviceBuffer scan_scratch;
  detail::DeviceBuffer state;
  detail::DeviceBuffer cubes;
  detail::DeviceBuffer cube_cases;
  detail::DeviceBuffer cube_first_triangle;
  detail::DeviceBuffer owners;
  detail::DeviceBuffer owner_edges;
  detail::DeviceBuffer owner_first_vertex;
  std::size_t scan_bytes{};
  std::uint64_t voxels{};  // The volume size the block scratch is for.
};

MeshExtractor::MeshExtractor(std::uint32_t max_vertices, std::uint32_t max_triangles)
    : max_vertices_(max_vertices), max_triangles_(max_triangles), impl_(std::make_unique<Impl>()) {
  if (max_vertices_ < 1U || max_vertices_ > kMaxMeshCapacity || max_triangles_ < 1U ||
      max_triangles_ > kMaxMeshCapacity) {
    throw std::invalid_argument("mesh capacities must be 1.." + std::to_string(kMaxMeshCapacity));
  }
}

MeshExtractor::~MeshExtractor() = default;

std::uint64_t MeshExtractor::scratch_bytes(std::uint64_t voxels, std::uint32_t max_vertices,
                                           std::uint32_t max_triangles) {
  const std::uint64_t blocks = blocks_for(voxels);
  std::size_t scan = 0;
  cub::DeviceScan::ExclusiveScan(nullptr, scan, static_cast<const Counts*>(nullptr),
                                 static_cast<Counts*>(nullptr), AddCounts{}, Counts{0U, 0U, 0U, 0U},
                                 static_cast<int>(blocks));
  return 2U * blocks * sizeof(Counts) + scan + sizeof(State) +
         9U * (std::uint64_t{max_vertices} + max_triangles);
}

cudaError_t MeshExtractor::extract(const TsdfVolume& volume, const MeshBuffers& out,
                                   cudaStream_t stream) {
  if (out.vertices == nullptr || out.triangles == nullptr || out.counts == nullptr) {
    return cudaErrorInvalidValue;
  }
  const TsdfVolumeConfig& config = volume.config();
  const std::uint64_t voxels = volume.voxel_count();
  const std::uint32_t blocks = blocks_for(voxels);
  Impl& s = *impl_;
  // Scratch: the lists at their capacities, and per-block counts for this
  // volume size.
  if (s.voxels != voxels) {
    std::size_t scan = 0;
    if (const cudaError_t status = cub::DeviceScan::ExclusiveScan(
            nullptr, scan, static_cast<const Counts*>(nullptr), static_cast<Counts*>(nullptr),
            AddCounts{}, Counts{0U, 0U, 0U, 0U}, static_cast<int>(blocks), stream);
        status != cudaSuccess) {
      return status;
    }
    for (auto [buffer, bytes] :
         {std::pair{&s.block_counts, std::size_t{blocks} * sizeof(Counts)},
          std::pair{&s.block_offsets, std::size_t{blocks} * sizeof(Counts)},
          std::pair{&s.scan_scratch, scan}, std::pair{&s.state, sizeof(State)},
          std::pair{&s.cubes, std::size_t{max_triangles_} * 4U},
          std::pair{&s.cube_cases, std::size_t{max_triangles_}},
          std::pair{&s.cube_first_triangle, std::size_t{max_triangles_} * 4U},
          std::pair{&s.owners, std::size_t{max_vertices_} * 4U},
          std::pair{&s.owner_edges, std::size_t{max_vertices_}},
          std::pair{&s.owner_first_vertex, std::size_t{max_vertices_} * 4U}}) {
      if (const cudaError_t status = buffer->reserve(bytes, stream); status != cudaSuccess) {
        return status;
      }
    }
    s.scan_bytes = scan;
    s.voxels = voxels;
  }

  const TsdfVolume::DeviceView view = volume.view();
  Volume v{};
  for (int r = 0; r < 3; ++r) {
    v.n[r] = config.dims[r];
    v.origin[r] = config.origin[r];
  }
  v.voxel = config.voxel_size;
  v.tsdf = view.tsdf;
  v.weight = view.weight;
  v.rgba = reinterpret_cast<const uchar4*>(view.rgba);
  MeshBuffers buffers = out;
  if (v.rgba == nullptr) {
    buffers.colors = nullptr;
  }
  const Lists lists{s.cubes.as<std::uint32_t>(),
                    s.cube_cases.as<std::uint8_t>(),
                    s.cube_first_triangle.as<std::uint32_t>(),
                    s.owners.as<std::uint32_t>(),
                    s.owner_edges.as<std::uint8_t>(),
                    s.owner_first_vertex.as<std::uint32_t>()};
  auto* state = s.state.as<State>();
  const auto count = static_cast<std::uint32_t>(voxels);

  const VolumeAccess access(
      volume.begin_access(stream),
      [](const TsdfVolume& owner, cudaStream_t on) { return owner.end_access(on); }, volume,
      stream);
  if (access.status() != cudaSuccess) {
    return access.status();
  }
  count_kernel<<<blocks, kBlock, 0, stream>>>(v, count, s.block_counts.as<Counts>());
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  std::size_t scan_bytes = s.scan_bytes;
  if (const cudaError_t status = cub::DeviceScan::ExclusiveScan(
          s.scan_scratch.as<void>(), scan_bytes, s.block_counts.as<Counts>(),
          s.block_offsets.as<Counts>(), AddCounts{}, Counts{0U, 0U, 0U, 0U},
          static_cast<int>(blocks), stream);
      status != cudaSuccess) {
    return status;
  }
  totals_kernel<<<1, 1, 0, stream>>>(s.block_offsets.as<Counts>(), s.block_counts.as<Counts>(),
                                     blocks, max_vertices_, max_triangles_, state, out.counts);
  scatter_kernel<<<blocks, kBlock, 0, stream>>>(v, count, s.block_offsets.as<Counts>(), state,
                                                lists, buffers);
  triangles_kernel<<<(max_triangles_ + kBlock - 1) / kBlock, kBlock, 0, stream>>>(v, state, lists,
                                                                                  out.triangles);
  return cudaGetLastError();
}

}  // namespace holoscan::tsdf
