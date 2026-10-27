// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <cstring>
#include <memory>
#include <stdexcept>

// CCCL's headers are C++; cpplint takes any angle-bracketed non-.hpp header for
// a C system header.
#include <cub/device/device_radix_sort.cuh>         // NOLINT(build/include_order)
#include <cub/device/device_run_length_encode.cuh>  // NOLINT(build/include_order)
#include <cub/device/device_scan.cuh>               // NOLINT(build/include_order)
#include <cub/device/device_select.cuh>             // NOLINT(build/include_order)
#include <thrust/iterator/counting_iterator.h>      // NOLINT(build/include_order)

#include "gpu_segmenter.hpp"

namespace holoscan::apriltag {
namespace {

// Threshold tiles are this many decimated pixels on a side.
constexpr int kTileSize = 4;
// Pixels marked neither black nor white.
constexpr std::uint8_t kUnknown = 127;
// Components smaller than this cannot border a tag and are skipped.
constexpr std::uint32_t kMinComponentSize = 25;
// Clusters with fewer raw points cannot be a tag border (AprilTag 3's limit).
constexpr std::uint32_t kMinClusterPoints = 24;
// Sorts after every real key in any range of low bits the sorts look at, so
// the unused tail of a buffer sorts last.
constexpr std::uint64_t kPadKey = ~std::uint64_t{0};
// The point each cluster's points are ordered about: its bounding box's center,
// nudged so that no point lies exactly on it. In doubled decimated pixels.
constexpr float kCenterNudgeX = 0.05118F;
constexpr float kCenterNudgeY = -0.028581F;
constexpr int kBlock = 256;
// The CUB calls in GpuSegmenter::run_cub.
constexpr int kCubStages = 9;

// One boundary point between a black and a white pixel of the decimated image,
// at twice the decimated resolution: (x, y) is the midpoint of the two pixel
// centers times two. (gx, gy) points from the dark pixel to the bright one, and
// `weight` is the gray image's gradient magnitude there, plus one.
struct EdgePoint {
  std::int16_t x;
  std::int16_t y;
  std::int8_t gx;
  std::int8_t gy;
  std::uint16_t weight;
};
static_assert(sizeof(EdgePoint) == sizeof(std::uint64_t));

unsigned int blocks_for(std::size_t items) {
  return static_cast<unsigned int>((items + kBlock - 1) / kBlock);
}

__device__ std::uint32_t found_points(const SegmentCounts* counts, std::uint32_t capacity) {
  return min(counts->points, capacity);
}

// Box-filters `decimate` x `decimate` blocks of the input into one pixel.
__global__ void decimate_kernel(const std::uint8_t* input, std::size_t pitch, int decimate,
                                std::uint8_t* output, int out_width, int out_height) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= out_width * out_height) {
    return;
  }
  const int x = index % out_width;
  const int y = index / out_width;
  unsigned int sum = 0;
  for (int dy = 0; dy < decimate; ++dy) {
    const std::uint8_t* row = input + static_cast<std::size_t>(y * decimate + dy) * pitch;
    for (int dx = 0; dx < decimate; ++dx) {
      sum += row[x * decimate + dx];
    }
  }
  const unsigned int area = static_cast<unsigned int>(decimate * decimate);
  output[index] = static_cast<std::uint8_t>((sum + area / 2) / area);
}

__global__ void tile_minmax_kernel(const std::uint8_t* image, int width, int height, int tiles_x,
                                   int tiles_y, std::uint8_t* tile_min, std::uint8_t* tile_max) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tiles_x * tiles_y) {
    return;
  }
  const int tx = index % tiles_x;
  const int ty = index / tiles_x;
  std::uint8_t lo = 255;
  std::uint8_t hi = 0;
  for (int dy = 0; dy < kTileSize; ++dy) {
    const int y = ty * kTileSize + dy;
    if (y >= height) {
      break;
    }
    for (int dx = 0; dx < kTileSize; ++dx) {
      const int x = tx * kTileSize + dx;
      if (x >= width) {
        break;
      }
      const std::uint8_t v = image[y * width + x];
      lo = min(lo, v);
      hi = max(hi, v);
    }
  }
  tile_min[index] = lo;
  tile_max[index] = hi;
}

// Widens each tile's extrema to its 3x3 tile neighborhood, so a tag edge on a
// tile boundary sees both of its sides.
__global__ void tile_dilate_kernel(const std::uint8_t* tile_min, const std::uint8_t* tile_max,
                                   int tiles_x, int tiles_y, std::uint8_t* out_min,
                                   std::uint8_t* out_max) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tiles_x * tiles_y) {
    return;
  }
  const int tx = index % tiles_x;
  const int ty = index / tiles_x;
  std::uint8_t lo = 255;
  std::uint8_t hi = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    const int y = ty + dy;
    if (y < 0 || y >= tiles_y) {
      continue;
    }
    for (int dx = -1; dx <= 1; ++dx) {
      const int x = tx + dx;
      if (x < 0 || x >= tiles_x) {
        continue;
      }
      lo = min(lo, tile_min[y * tiles_x + x]);
      hi = max(hi, tile_max[y * tiles_x + x]);
    }
  }
  out_min[index] = lo;
  out_max[index] = hi;
}

// Marks each pixel black (0), white (255), or unknown where its tile
// neighborhood has too little contrast. Labels every known pixel as its own
// component.
__global__ void threshold_kernel(const std::uint8_t* image, int width, int height,
                                 const std::uint8_t* tile_min, const std::uint8_t* tile_max,
                                 int tiles_x, int min_white_black_diff, std::uint8_t* output,
                                 std::int32_t* labels) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= width * height) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const int tile = (y / kTileSize) * tiles_x + x / kTileSize;
  const int lo = tile_min[tile];
  const int hi = tile_max[tile];
  std::uint8_t value = kUnknown;
  if (hi - lo >= min_white_black_diff) {
    value = image[index] > lo + (hi - lo) / 2 ? 255 : 0;
  }
  output[index] = value;
  labels[index] = value == kUnknown ? -1 : index;
}

__device__ int find_root(const std::int32_t* labels, int node) {
  const volatile std::int32_t* parents = labels;
  int parent = parents[node];
  while (parent != node) {
    node = parent;
    parent = parents[node];
  }
  return node;
}

// Lock-free union: the larger root is pointed at the smaller with atomicMin,
// retrying from the new root if another thread moved it first.
__device__ void unite(std::int32_t* labels, int a, int b) {
  while (true) {
    a = find_root(labels, a);
    b = find_root(labels, b);
    if (a == b) {
      return;
    }
    if (a > b) {
      const int swap = a;
      a = b;
      b = swap;
    }
    const int previous = atomicMin(&labels[b], a);
    if (previous == b) {
      return;
    }
    b = previous;
  }
}

// Black pixels join their 4-neighbors and white pixels their 8-neighbors, so a
// thin black border is not split by a diagonal white bridge.
__global__ void merge_kernel(const std::uint8_t* image, int width, int height,
                             std::int32_t* labels) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= width * height) {
    return;
  }
  const std::uint8_t value = image[index];
  if (value == kUnknown) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  if (x > 0 && image[index - 1] == value) {
    unite(labels, index, index - 1);
  }
  if (y > 0) {
    const int up = index - width;
    if (image[up] == value) {
      unite(labels, index, up);
    }
    if (value == 255) {
      if (x > 0 && image[up - 1] == value) {
        unite(labels, index, up - 1);
      }
      if (x + 1 < width && image[up + 1] == value) {
        unite(labels, index, up + 1);
      }
    }
  }
}

__global__ void flatten_kernel(int count, std::int32_t* labels, std::uint32_t* sizes) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count || labels[index] < 0) {
    return;
  }
  const int root = find_root(labels, index);
  labels[index] = root;
  atomicAdd(&sizes[root], 1U);
}

__device__ std::uint64_t pack_point(int x, int y, int gx, int gy, int weight) {
  const EdgePoint point{static_cast<std::int16_t>(x), static_cast<std::int16_t>(y),
                        static_cast<std::int8_t>(gx), static_cast<std::int8_t>(gy),
                        static_cast<std::uint16_t>(weight)};
  std::uint64_t packed;
  memcpy(&packed, &point, sizeof(packed));
  return packed;
}

// Emits a point between each pixel and those of its right, lower and two lower
// diagonal neighbors with the opposite color, keyed by the pair of components:
// the lower root, then the higher, in `component_bits` bits each.
__global__ void edge_points_kernel(const std::uint8_t* image, const std::uint8_t* gray, int width,
                                   int height, const std::int32_t* labels,
                                   const std::uint32_t* sizes, int component_bits,
                                   std::uint64_t* keys, std::uint64_t* points,
                                   std::uint32_t* counter, std::uint32_t capacity) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= width * height) {
    return;
  }
  const std::uint8_t v0 = image[index];
  if (v0 == kUnknown) {
    return;
  }
  const int root0 = labels[index];
  if (sizes[root0] < kMinComponentSize) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  constexpr int kOffsets[4][2] = {{1, 0}, {0, 1}, {-1, 1}, {1, 1}};
  for (const auto& offset : kOffsets) {
    const int x1 = x + offset[0];
    const int y1 = y + offset[1];
    if (x1 < 0 || x1 >= width || y1 >= height) {
      continue;
    }
    const int neighbor = y1 * width + x1;
    const std::uint8_t v1 = image[neighbor];
    if (static_cast<int>(v0) + static_cast<int>(v1) != 255) {
      continue;
    }
    const int root1 = labels[neighbor];
    if (sizes[root1] < kMinComponentSize) {
      continue;
    }
    // Past the buffer's end the point is only counted, so the frame can report
    // how many were dropped.
    const std::uint32_t slot = atomicAdd(counter, 1U);
    if (slot >= capacity) {
      continue;
    }
    const auto lo = static_cast<std::uint64_t>(min(root0, root1));
    const auto hi = static_cast<std::uint64_t>(max(root0, root1));
    keys[slot] = (lo << component_bits) | hi;
    const int sign = v1 > v0 ? 1 : -1;
    // Weight by the gray gradient at the pixel the point rounds to, so points
    // on strong edges dominate the line fits.
    const int px = 2 * x + offset[0];
    const int py = 2 * y + offset[1];
    const int ix = (px + 1) / 2;
    const int iy = (py + 1) / 2;
    int weight = 1;
    if (ix > 0 && ix + 1 < width && iy > 0 && iy + 1 < height) {
      const int dx = gray[iy * width + ix + 1] - gray[iy * width + ix - 1];
      const int dy = gray[(iy + 1) * width + ix] - gray[(iy - 1) * width + ix];
      weight = static_cast<int>(sqrtf(static_cast<float>(dx * dx + dy * dy))) + 1;
    }
    points[slot] = pack_point(px, py, offset[0] * sign, offset[1] * sign, weight);
  }
}

// Marks the unused tail of the point buffer as padding, which sorts last.
__global__ void pad_kernel(const SegmentCounts* counts, std::uint32_t capacity,
                           std::uint64_t* keys) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < capacity && index >= found_points(counts, capacity)) {
    keys[index] = kPadKey;
  }
}

__global__ void init_clusters_kernel(std::uint32_t capacity, std::int32_t* box, double* polarity,
                                     std::uint32_t* unique_counts, std::uint32_t* maxima_counts) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= capacity) {
    return;
  }
  box[4 * index + 0] = INT32_MAX;
  box[4 * index + 1] = INT32_MIN;
  box[4 * index + 2] = INT32_MAX;
  box[4 * index + 3] = INT32_MIN;
  polarity[index] = 0.0;
  unique_counts[index] = 0;
  maxima_counts[index] = 0;
}

__device__ EdgePoint unpack_point(std::uint64_t packed) {
  EdgePoint point;
  memcpy(&point, &packed, sizeof(point));
  return point;
}

__device__ float2 cluster_center(const std::int32_t* box, std::uint32_t cluster) {
  return {0.5F * (box[4 * cluster + 0] + box[4 * cluster + 1]) + kCenterNudgeX,
          0.5F * (box[4 * cluster + 2] + box[4 * cluster + 3]) + kCenterNudgeY};
}

// Finds each point's cluster (by binary search over the run offsets) and grows
// that cluster's bounding box.
__global__ void cluster_boxes_kernel(const std::uint64_t* points, ClusterData data,
                                     const std::uint32_t* run_offsets, std::uint32_t* point_cluster,
                                     std::int32_t* box) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= found_points(data.counts, data.capacity)) {
    return;
  }
  int lo = 0;
  int hi = static_cast<int>(data.clusters()) - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (run_offsets[mid] <= index) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  point_cluster[index] = static_cast<std::uint32_t>(lo);
  const EdgePoint point = unpack_point(points[index]);
  atomicMin(&box[4 * lo + 0], point.x);
  atomicMax(&box[4 * lo + 1], point.x);
  atomicMin(&box[4 * lo + 2], point.y);
  atomicMax(&box[4 * lo + 3], point.y);
}

// Keys each point by its cluster, then by its angle about the cluster's center,
// so one sort leaves every cluster in angular order.
__global__ void angle_keys_kernel(const std::uint64_t* points, const SegmentCounts* counts,
                                  std::uint32_t capacity, const std::uint32_t* point_cluster,
                                  const std::int32_t* box, std::uint64_t* keys) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= capacity) {
    return;
  }
  if (index >= found_points(counts, capacity)) {
    keys[index] = kPadKey;
    return;
  }
  const std::uint32_t cluster = point_cluster[index];
  const EdgePoint point = unpack_point(points[index]);
  const float2 center = cluster_center(box, cluster);
  const float angle = atan2f(point.y - center.y, point.x - center.x);
  // Map the float's bits to an unsigned value that sorts in the same order.
  std::uint32_t bits = __float_as_uint(angle);
  bits = (bits & 0x80000000U) != 0U ? ~bits : bits | 0x80000000U;
  keys[index] = (static_cast<std::uint64_t>(cluster) << 32) | bits;
}

// Marks the first of each run of equal points within a cluster (a boundary
// point two neighbors both reported), and counts each cluster's survivors.
__global__ void unique_points_kernel(const std::uint64_t* keys, const std::uint64_t* points,
                                     const SegmentCounts* counts, std::uint32_t capacity,
                                     std::uint8_t* keep, std::uint32_t* unique_counts) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= capacity) {
    return;
  }
  if (index >= found_points(counts, capacity)) {
    keep[index] = 0U;
    return;
  }
  const auto cluster = static_cast<std::uint32_t>(keys[index] >> 32);
  // The low 32 bits of a packed EdgePoint are its (x, y).
  const bool first =
      index == 0 || static_cast<std::uint32_t>(keys[index - 1] >> 32) != cluster ||
      static_cast<std::uint32_t>(points[index - 1]) != static_cast<std::uint32_t>(points[index]);
  keep[index] = first ? 1U : 0U;
  if (first) {
    atomicAdd(&unique_counts[cluster], 1U);
  }
}

// Each kept point's cluster and line-fit moments, and its share of its
// cluster's polarity.
__global__ void moments_kernel(const std::uint64_t* keys, const std::uint64_t* points,
                               const std::uint32_t* unique_index, const SegmentCounts* counts,
                               const std::int32_t* box, std::uint32_t* cluster_of,
                               LineMoments* moments, double* polarity) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= counts->unique) {
    return;
  }
  const std::uint32_t from = unique_index[index];
  const EdgePoint point = unpack_point(points[from]);
  const auto cluster = static_cast<std::uint32_t>(keys[from] >> 32);
  cluster_of[index] = cluster;
  const double x = 0.5 * point.x + 0.5;
  const double y = 0.5 * point.y + 0.5;
  const double w = point.weight;
  moments[index] = {w * x, w * y, w * x * x, w * x * y, w * y * y, w};
  const float2 center = cluster_center(box, cluster);
  atomicAdd(&polarity[cluster],
            static_cast<double>((point.x - center.x) * point.gx + (point.y - center.y) * point.gy));
}

// A point's cluster, the cluster's first point and size, and the point's index in it.
struct ClusterPoint {
  std::uint32_t cluster;
  int offset;
  int n;
  int i;
};

__device__ ClusterPoint locate(std::uint32_t index, const std::uint32_t* cluster_of,
                               const std::uint32_t* offsets, const std::uint32_t* counts) {
  const std::uint32_t cluster = cluster_of[index];
  const int offset = static_cast<int>(offsets[cluster]);
  return {cluster, offset, static_cast<int>(counts[cluster]), static_cast<int>(index) - offset};
}

// How badly a line fits each point's neighborhood: high at corners.
__global__ void window_error_kernel(const LineMoments* prefix, const std::uint32_t* cluster_of,
                                    const std::uint32_t* offsets, const std::uint32_t* counts,
                                    const SegmentCounts* totals, double* error) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= totals->unique) {
    return;
  }
  const auto [cluster, offset, n, i] = locate(index, cluster_of, offsets, counts);
  const int window = min(20, n / 12);
  if (window < 2) {
    error[index] = 0.0;
    return;
  }
  const LineMoments* mine = prefix + offset;
  const int first = (i + n - window) % n;
  const int last = (i + window) % n;
  const LineMoments m = range_moments(first == 0 ? LineMoments{} : mine[first - 1], mine[last],
                                      mine[n - 1], first > last);
  error[index] = (2 * window + 1) * Covariance(m).smallest();
}

__global__ void smooth_kernel(const double* error, const std::uint32_t* cluster_of,
                              const std::uint32_t* offsets, const std::uint32_t* counts,
                              const SegmentCounts* totals, double* smoothed) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= totals->unique) {
    return;
  }
  const auto [cluster, offset, n, i] = locate(index, cluster_of, offsets, counts);
  // AprilTag 3's unit-sigma Gaussian, seven taps, unnormalized.
  double sum = 0.0;
  for (int k = -3; k <= 3; ++k) {
    sum += exp(-0.5 * k * k) * error[offset + ((i + k) % n + n) % n];
  }
  smoothed[index] = sum;
}

// Flags the local maxima of each fittable cluster's smoothed error, and counts
// them per cluster.
__global__ void maxima_kernel(const double* smoothed, const std::uint32_t* cluster_of,
                              const std::uint32_t* offsets, const std::uint32_t* counts,
                              const std::uint32_t* raw_counts, std::uint32_t max_raw_points,
                              const SegmentCounts* totals, std::uint32_t capacity,
                              std::uint8_t* is_maximum, std::uint32_t* maxima_counts) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= capacity) {
    return;
  }
  if (index >= totals->unique) {
    is_maximum[index] = 0U;
    return;
  }
  const auto [cluster, offset, n, i] = locate(index, cluster_of, offsets, counts);
  const std::uint32_t raw = raw_counts[cluster];
  const double here = smoothed[index];
  const bool maximum = raw >= kMinClusterPoints && raw <= max_raw_points && n / 12 >= 2 &&
                       here > smoothed[offset + (i + n - 1) % n] &&
                       here > smoothed[offset + (i + 1) % n];
  is_maximum[index] = maximum ? 1U : 0U;
  if (maximum) {
    atomicAdd(&maxima_counts[cluster], 1U);
  }
}

struct AddMoments {
  __host__ __device__ LineMoments operator()(const LineMoments& a, const LineMoments& b) const {
    return a + b;
  }
};

}  // namespace

struct GpuSegmenter::Buffers {
  Buffers(std::size_t pixel_count, std::size_t tile_count, std::size_t point_capacity)
      : pixels(pixel_count), tiles(tile_count), capacity(point_capacity) {}

  std::size_t pixels;
  std::size_t tiles;
  std::size_t capacity;
  // Per decimated pixel or tile.
  DeviceBuffer<std::uint8_t> decimated{pixels};
  DeviceBuffer<std::uint8_t> tile_min{tiles};
  DeviceBuffer<std::uint8_t> tile_max{tiles};
  DeviceBuffer<std::uint8_t> tile_min_dilated{tiles};
  DeviceBuffer<std::uint8_t> tile_max_dilated{tiles};
  DeviceBuffer<std::uint8_t> threshold{pixels};
  DeviceBuffer<std::int32_t> labels{pixels};
  DeviceBuffer<std::uint32_t> sizes{pixels};
  // Per boundary point.
  DeviceBuffer<std::uint64_t> keys{capacity};
  DeviceBuffer<std::uint64_t> sorted_keys{capacity};
  DeviceBuffer<std::uint64_t> points{capacity};
  DeviceBuffer<std::uint64_t> sorted_points{capacity};
  DeviceBuffer<std::uint32_t> point_cluster{capacity};
  DeviceBuffer<std::uint8_t> keep{capacity};
  DeviceBuffer<std::uint32_t> unique_index{capacity};
  DeviceBuffer<std::uint32_t> cluster_of{capacity};
  DeviceBuffer<LineMoments> moments{capacity};
  DeviceBuffer<LineMoments> prefix{capacity};
  DeviceBuffer<double> error{capacity};
  DeviceBuffer<double> smoothed{capacity};
  DeviceBuffer<std::uint8_t> is_maximum{capacity};
  DeviceBuffer<std::uint32_t> maxima{capacity};
  // Per cluster.
  DeviceBuffer<std::uint64_t> run_keys{capacity};
  DeviceBuffer<std::uint32_t> run_counts{capacity};
  DeviceBuffer<std::uint32_t> run_offsets{capacity};
  DeviceBuffer<std::uint32_t> unique_counts{capacity};
  DeviceBuffer<std::uint32_t> unique_offsets{capacity};
  DeviceBuffer<std::uint32_t> maxima_counts{capacity};
  DeviceBuffer<std::uint32_t> maxima_offsets{capacity};
  DeviceBuffer<double> polarity{capacity};
  DeviceBuffer<std::int32_t> box{4 * capacity};
  DeviceBuffer<SegmentCounts> counts{1};
  std::unique_ptr<DeviceBuffer<std::uint8_t>> scratch;
  std::size_t scratch_bytes{};
};

GpuSegmenter::GpuSegmenter(int width, int height, int decimate, int min_white_black_diff)
    : decimate_(decimate), min_white_black_diff_(min_white_black_diff) {
  if (decimate < 1 || width / decimate < 2 * kTileSize || height / decimate < 2 * kTileSize) {
    throw std::invalid_argument("CUDA backend: the decimated image is too small");
  }
  if (width / decimate > 16383 || height / decimate > 16383) {
    // Edge points store twice the decimated coordinate in 16 bits.
    throw std::invalid_argument("CUDA backend: the decimated image is too large");
  }
  dec_width_ = width / decimate;
  dec_height_ = height / decimate;
  tiles_x_ = (dec_width_ + kTileSize - 1) / kTileSize;
  tiles_y_ = (dec_height_ + kTileSize - 1) / kTileSize;
  const std::size_t pixels = static_cast<std::size_t>(dec_width_) * dec_height_;
  // Two boundary points per decimated pixel: a noisy scene needs about one.
  // Points past it are dropped.
  capacity_ = static_cast<std::uint32_t>(2 * pixels);
  // The sorts look only at the key bits in use: two component roots (pixel
  // indices), or a cluster index above a 32-bit angle. Each bit fewer saves
  // radix passes; the order, and so every result, is the same.
  component_bits_ = static_cast<int>(std::bit_width(pixels - 1));
  cluster_bits_ = static_cast<int>(std::bit_width(capacity_));
  buffers_ =
      std::make_unique<Buffers>(pixels, static_cast<std::size_t>(tiles_x_) * tiles_y_, capacity_);
  // CUB's scratch, sized once for its largest call.
  std::size_t bytes = 0;
  for (int stage = 0; stage < kCubStages; ++stage) {
    std::size_t need = 0;
    run_cub(stage, nullptr, need, nullptr);
    bytes = std::max(bytes, need);
  }
  buffers_->scratch = std::make_unique<DeviceBuffer<std::uint8_t>>(bytes);
  buffers_->scratch_bytes = bytes;
}

GpuSegmenter::~GpuSegmenter() = default;

void GpuSegmenter::run_cub(int stage, void* temp, std::size_t& bytes, cudaStream_t stream) {
  Buffers& b = *buffers_;
  const auto n = static_cast<int>(capacity_);
  const thrust::counting_iterator<std::uint32_t> index{0U};
  cudaError_t status = cudaSuccess;
  switch (stage) {
    case 0:  // Group the points by component pair.
      status = cub::DeviceRadixSort::SortPairs(temp, bytes, b.keys.get(), b.sorted_keys.get(),
                                               b.points.get(), b.sorted_points.get(), n, 0,
                                               2 * component_bits_, stream);
      break;
    case 1:
      status =
          cub::DeviceRunLengthEncode::Encode(temp, bytes, b.sorted_keys.get(), b.run_keys.get(),
                                             b.run_counts.get(), &b.counts.get()->runs, n, stream);
      break;
    case 2:
      status = cub::DeviceScan::ExclusiveSum(temp, bytes, b.run_counts.get(), b.run_offsets.get(),
                                             n, stream);
      break;
    case 3:  // Every cluster in angular order.
      status = cub::DeviceRadixSort::SortPairs(temp, bytes, b.keys.get(), b.sorted_keys.get(),
                                               b.sorted_points.get(), b.points.get(), n, 0,
                                               32 + cluster_bits_, stream);
      break;
    case 4:  // Drop repeated points.
      status = cub::DeviceSelect::Flagged(temp, bytes, index, b.keep.get(), b.unique_index.get(),
                                          &b.counts.get()->unique, n, stream);
      break;
    case 5:
      status = cub::DeviceScan::ExclusiveSum(temp, bytes, b.unique_counts.get(),
                                             b.unique_offsets.get(), n, stream);
      break;
    case 6:
      status = cub::DeviceScan::InclusiveScanByKey(temp, bytes, b.cluster_of.get(), b.moments.get(),
                                                   b.prefix.get(), AddMoments{}, n,
                                                   ::cuda::std::equal_to<>{}, stream);
      break;
    case 7:  // The corner candidates.
      status = cub::DeviceSelect::Flagged(temp, bytes, index, b.is_maximum.get(), b.maxima.get(),
                                          &b.counts.get()->maxima, n, stream);
      break;
    default:
      status = cub::DeviceScan::ExclusiveSum(temp, bytes, b.maxima_counts.get(),
                                             b.maxima_offsets.get(), n, stream);
      break;
  }
  cuda_check(status, "CUB");
}

ClusterData GpuSegmenter::data() const noexcept {
  const Buffers& b = *buffers_;
  return ClusterData{.counts = b.counts.get(),
                     .capacity = capacity_,
                     .raw_points = b.run_counts.get(),
                     .offsets = b.unique_offsets.get(),
                     .point_counts = b.unique_counts.get(),
                     .box = b.box.get(),
                     .polarity = b.polarity.get(),
                     .prefix = b.prefix.get(),
                     .smoothed = b.smoothed.get(),
                     .maxima = b.maxima.get(),
                     .maxima_offsets = b.maxima_offsets.get(),
                     .maxima_counts = b.maxima_counts.get()};
}

void GpuSegmenter::segment(const GrayImageView& image, cudaStream_t stream) {
  Buffers& b = *buffers_;
  const std::size_t pixels = static_cast<std::size_t>(dec_width_) * dec_height_;
  const std::size_t tiles = static_cast<std::size_t>(tiles_x_) * tiles_y_;
  const unsigned int point_blocks = blocks_for(capacity_);
  const auto cub = [&](int stage) {
    std::size_t bytes = b.scratch_bytes;
    run_cub(stage, b.scratch->get(), bytes, stream);
  };

  decimate_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(
      image.data, image.pitch, decimate_, b.decimated.get(), dec_width_, dec_height_);
  tile_minmax_kernel<<<blocks_for(tiles), kBlock, 0, stream>>>(b.decimated.get(), dec_width_,
                                                               dec_height_, tiles_x_, tiles_y_,
                                                               b.tile_min.get(), b.tile_max.get());
  tile_dilate_kernel<<<blocks_for(tiles), kBlock, 0, stream>>>(
      b.tile_min.get(), b.tile_max.get(), tiles_x_, tiles_y_, b.tile_min_dilated.get(),
      b.tile_max_dilated.get());
  threshold_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(
      b.decimated.get(), dec_width_, dec_height_, b.tile_min_dilated.get(),
      b.tile_max_dilated.get(), tiles_x_, min_white_black_diff_, b.threshold.get(), b.labels.get());
  merge_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(b.threshold.get(), dec_width_,
                                                          dec_height_, b.labels.get());
  cuda_check(cudaMemsetAsync(b.sizes.get(), 0, pixels * sizeof(std::uint32_t), stream),
             "cudaMemsetAsync");
  flatten_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(static_cast<int>(pixels),
                                                            b.labels.get(), b.sizes.get());

  // Gather the boundary points, pad the rest, and group them by component pair.
  cuda_check(cudaMemsetAsync(b.counts.get(), 0, sizeof(SegmentCounts), stream), "cudaMemsetAsync");
  edge_points_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(
      b.threshold.get(), b.decimated.get(), dec_width_, dec_height_, b.labels.get(), b.sizes.get(),
      component_bits_, b.keys.get(), b.points.get(), &b.counts.get()->points, capacity_);
  pad_kernel<<<point_blocks, kBlock, 0, stream>>>(b.counts.get(), capacity_, b.keys.get());
  cuda_check(cudaGetLastError(), "segmentation kernels");
  cub(0);
  cub(1);
  cub(2);

  // Order every cluster's points by angle about its center.
  const ClusterData clusters = data();
  init_clusters_kernel<<<point_blocks, kBlock, 0, stream>>>(
      capacity_, b.box.get(), b.polarity.get(), b.unique_counts.get(), b.maxima_counts.get());
  cluster_boxes_kernel<<<point_blocks, kBlock, 0, stream>>>(
      b.sorted_points.get(), clusters, b.run_offsets.get(), b.point_cluster.get(), b.box.get());
  angle_keys_kernel<<<point_blocks, kBlock, 0, stream>>>(b.sorted_points.get(), b.counts.get(),
                                                         capacity_, b.point_cluster.get(),
                                                         b.box.get(), b.keys.get());
  cuda_check(cudaGetLastError(), "angle ordering kernels");
  cub(3);

  // Drop repeated points, then accumulate each cluster's line-fit moments.
  unique_points_kernel<<<point_blocks, kBlock, 0, stream>>>(b.sorted_keys.get(), b.points.get(),
                                                            b.counts.get(), capacity_, b.keep.get(),
                                                            b.unique_counts.get());
  cuda_check(cudaGetLastError(), "unique_points_kernel");
  cub(4);
  cub(5);
  moments_kernel<<<point_blocks, kBlock, 0, stream>>>(
      b.sorted_keys.get(), b.points.get(), b.unique_index.get(), b.counts.get(), b.box.get(),
      b.cluster_of.get(), b.moments.get(), b.polarity.get());
  cuda_check(cudaGetLastError(), "moments_kernel");
  cub(6);

  // Score each point, smooth the scores, and flag the corner candidates.
  window_error_kernel<<<point_blocks, kBlock, 0, stream>>>(
      b.prefix.get(), b.cluster_of.get(), b.unique_offsets.get(), b.unique_counts.get(),
      b.counts.get(), b.error.get());
  smooth_kernel<<<point_blocks, kBlock, 0, stream>>>(b.error.get(), b.cluster_of.get(),
                                                     b.unique_offsets.get(), b.unique_counts.get(),
                                                     b.counts.get(), b.smoothed.get());
  maxima_kernel<<<point_blocks, kBlock, 0, stream>>>(
      b.smoothed.get(), b.cluster_of.get(), b.unique_offsets.get(), b.unique_counts.get(),
      b.run_counts.get(), max_cluster_points(), b.counts.get(), capacity_, b.is_maximum.get(),
      b.maxima_counts.get());
  cuda_check(cudaGetLastError(), "corner candidate kernels");
  cub(7);
  cub(8);
}

}  // namespace holoscan::apriltag
