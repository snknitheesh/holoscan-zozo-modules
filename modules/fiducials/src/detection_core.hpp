// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The detector's per-pixel and per-candidate steps, compiled for the host and the device, so
// the cpu and cuda backends compute the same thing; they differ only in how they loop.
//
//  1. Threshold: a pixel is dark when it is below the middle of the darkest and brightest
//     values in the 3 x 3 tiles around it, and those differ by at least min_contrast.
//  2. Dark pixels are joined into 8-connected components (in the backends).
//  3. Each large enough component is a candidate. Four passes over its pixels find its quad:
//     the pixel farthest from its centroid (c0), the one farthest from c0 (c2), and the ones
//     farthest from the line c0-c2 on either side (c1, c3). A marker's border is convex, so
//     these are its corners, whatever else of the marker joined the component.
//  4. The quad's edges are refined to sub-pixel accuracy: along each edge, the gray image's
//     steepest rise across it, fitted with a line; the corners are where the lines meet.
//  5. The cells are sampled through the quad's homography, thresholded at the middle of the
//     darkest and brightest cell, the border checked black, and the inner cells looked up in
//     the dictionary in all four rotations.
//
// Chessboard corners (ChArUco) are refined by refine_saddle(), the gradient method of
// OpenCV's cornerSubPix.
//
// Steps 4 and 5 and the chessboard corners run on 32 lanes: a CUDA warp per candidate or corner
// on the GPU, a loop over the lanes on the CPU (SerialLanes). Each lane takes its share of the
// edge samples, cells, dictionary markers or window pixels, and the shares are added in a fixed
// butterfly order, the same on both. Everything is plain arithmetic, no transcendental
// functions, and the CUDA file is built without FMA contraction, so both backends round alike
// and give the same results to the bit.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define FIDUCIALS_HD __host__ __device__
// The per-candidate steps stay separate functions on the device: inlined and unrolled into one
// kernel, they take the compiler many minutes and gain nothing.
#define FIDUCIALS_HD_NOINLINE __host__ __device__ __noinline__
#else
#define FIDUCIALS_HD
#define FIDUCIALS_HD_NOINLINE
#endif
#if defined(__CUDA_ARCH__)
#define FIDUCIALS_NO_UNROLL _Pragma("unroll 1")
#else
#define FIDUCIALS_NO_UNROLL
#endif

namespace holoscan::fiducials::core {

// Most candidates a frame examines; components past it are counted, not examined.
inline constexpr std::uint32_t kMaxCandidates = 4096;
// Largest image side: keeps every squared distance in 32 bits.
inline constexpr int kMaxImageSide = 16384;
// Largest marker, in cells per side with the border: 7 x 7 inner cells.
inline constexpr int kMaxCells = 9;
// Lanes that share a candidate's or a chessboard corner's work: a CUDA warp.
inline constexpr int kLanes = 32;
// Largest half window for a chessboard corner.
inline constexpr int kMaxHalfWindow = 12;
// Doubles of scratch the lanes of one candidate or corner share: a marker's cells, or a
// chessboard corner's window and a pixel around it.
inline constexpr int kScratch = (2 * kMaxHalfWindow + 3) * (2 * kMaxHalfWindow + 3);

struct Image {
  const std::uint8_t* data;
  int width;
  int height;
  std::size_t pitch;
};

// Settings every stage reads.
struct Params {
  int width;
  int height;
  int tile_size;
  int tiles_x;
  int tiles_y;
  int min_contrast;
  int min_marker_px;
  std::uint32_t min_component_pixels;
  bool refine;
  int marker_size;  // Inner cells per side.
  int max_hamming;
  int dictionary_count;
};

// A candidate's statistics, gathered by the passes over its pixels. Keys pack a value in the
// high 32 bits and ~pixel index in the low 32, so their maximum is the largest value, the
// lowest pixel index among equals; 0 means none.
struct Candidate {
  std::int32_t root;  // Its component's label: the lowest pixel index in it.
  std::uint32_t pixels;
  unsigned long long sum_x;
  unsigned long long sum_y;
  std::int32_t min_x;
  std::int32_t min_y;
  std::int32_t max_x;
  std::int32_t max_y;
  unsigned long long far0;      // c0: farthest from the centroid.
  unsigned long long far2;      // c2: farthest from c0.
  unsigned long long side_pos;  // c1 or c3: farthest from c0-c2 on the positive side.
  unsigned long long side_neg;  // The other one.
};

// What examining a candidate found.
struct CandidateResult {
  std::int32_t marker;   // 1 when the candidate is a marker of the dictionary, else 0.
  std::int32_t id;
  std::int32_t hamming;  // Bits corrected.
  double corners[4][2];  // In the marker's order: top-left, top-right, bottom-right, bottom-left.
};

FIDUCIALS_HD inline void init_candidate(Candidate& c, std::int32_t root) {
  c.root = root;
  c.pixels = 0;
  c.sum_x = c.sum_y = 0;
  c.min_x = c.min_y = 0x7FFFFFFF;
  c.max_x = c.max_y = -1;
  c.far0 = c.far2 = c.side_pos = c.side_neg = 0;
}

FIDUCIALS_HD inline unsigned long long make_key(std::uint32_t value, std::uint32_t index) {
  return (static_cast<unsigned long long>(value) << 32U) | (0xFFFFFFFFU - index);
}
FIDUCIALS_HD inline std::uint32_t key_index(unsigned long long key) {
  return 0xFFFFFFFFU - static_cast<std::uint32_t>(key & 0xFFFFFFFFU);
}

FIDUCIALS_HD inline int popcount64(std::uint64_t value) {
#if defined(__CUDA_ARCH__)
  return __popcll(value);
#else
  return __builtin_popcountll(value);
#endif
}

// ---- 1. Threshold --------------------------------------------------------------------------

// The darkest and brightest pixel of tile (tx, ty).
FIDUCIALS_HD inline void tile_range(const Image& image, int tile_size, int tx, int ty,
                                    std::uint8_t& lo, std::uint8_t& hi) {
  lo = 255;
  hi = 0;
  const int x_end = (tx + 1) * tile_size < image.width ? (tx + 1) * tile_size : image.width;
  const int y_end = (ty + 1) * tile_size < image.height ? (ty + 1) * tile_size : image.height;
  for (int y = ty * tile_size; y < y_end; ++y) {
    const std::uint8_t* row = image.data + static_cast<std::size_t>(y) * image.pitch;
    for (int x = tx * tile_size; x < x_end; ++x) {
      lo = row[x] < lo ? row[x] : lo;
      hi = row[x] > hi ? row[x] : hi;
    }
  }
}

// Tile (tx, ty)'s range widened to its 3 x 3 tile neighborhood, from the per-tile ranges.
FIDUCIALS_HD inline void neighborhood_range(const std::uint8_t* tile_lo, const std::uint8_t* tile_hi,
                                            int tiles_x, int tiles_y, int tx, int ty,
                                            std::uint8_t& lo, std::uint8_t& hi) {
  lo = 255;
  hi = 0;
  for (int y = ty - 1; y <= ty + 1; ++y) {
    for (int x = tx - 1; x <= tx + 1; ++x) {
      if (x < 0 || y < 0 || x >= tiles_x || y >= tiles_y) continue;
      const int t = y * tiles_x + x;
      lo = tile_lo[t] < lo ? tile_lo[t] : lo;
      hi = tile_hi[t] > hi ? tile_hi[t] : hi;
    }
  }
}

FIDUCIALS_HD inline bool is_dark(int value, int lo, int hi, int min_contrast) {
  return hi - lo >= min_contrast && 2 * value < lo + hi;
}

// ---- 3. Quad passes ------------------------------------------------------------------------

// The keys each pass offers for pixel (x, y), index `index`, of candidate `c`.
FIDUCIALS_HD inline unsigned long long far0_key(const Candidate& c, int x, int y,
                                                std::uint32_t index) {
  const double cx = static_cast<double>(c.sum_x) / c.pixels;
  const double cy = static_cast<double>(c.sum_y) / c.pixels;
  const double dx = x - cx;
  const double dy = y - cy;
  const float d2 = static_cast<float>(dx * dx + dy * dy);
  // A non-negative float orders as its bits.
#if defined(__CUDA_ARCH__)
  const std::uint32_t bits = __float_as_uint(d2);
#else
  std::uint32_t bits;
  static_assert(sizeof(bits) == sizeof(d2));
  std::memcpy(&bits, &d2, sizeof(bits));
#endif
  return make_key(bits, index);
}

FIDUCIALS_HD inline unsigned long long far2_key(const Candidate& c, int width, int x, int y,
                                                std::uint32_t index) {
  const std::uint32_t c0 = key_index(c.far0);
  const std::int64_t dx = x - static_cast<std::int64_t>(c0 % width);
  const std::int64_t dy = y - static_cast<std::int64_t>(c0 / width);
  return make_key(static_cast<std::uint32_t>(dx * dx + dy * dy), index);
}

// The signed area of (c0, c2, p), twice over: which side of c0-c2 the pixel is on, and how far.
FIDUCIALS_HD inline std::int64_t side_of(const Candidate& c, int width, int x, int y) {
  const std::uint32_t c0 = key_index(c.far0);
  const std::uint32_t c2 = key_index(c.far2);
  const std::int64_t x0 = c0 % width;
  const std::int64_t y0 = c0 / width;
  const std::int64_t x2 = c2 % width;
  const std::int64_t y2 = c2 / width;
  return (x2 - x0) * (y - y0) - (y2 - y0) * (x - x0);
}

// ---- 4. Edge refinement --------------------------------------------------------------------

// The image at (x, y), bilinear, pixel centers at integers, clamped to the image.
FIDUCIALS_HD inline double sample(const Image& image, double x, double y) {
  x = x < 0.0 ? 0.0 : (x > image.width - 1.0 ? image.width - 1.0 : x);
  y = y < 0.0 ? 0.0 : (y > image.height - 1.0 ? image.height - 1.0 : y);
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = x0 + 1 < image.width ? x0 + 1 : x0;
  const int y1 = y0 + 1 < image.height ? y0 + 1 : y0;
  const double fx = x - x0;
  const double fy = y - y0;
  const std::uint8_t* r0 = image.data + static_cast<std::size_t>(y0) * image.pitch;
  const std::uint8_t* r1 = image.data + static_cast<std::size_t>(y1) * image.pitch;
  const double top = r0[x0] + fx * (r0[x1] - r0[x0]);
  const double bottom = r1[x0] + fx * (r1[x1] - r1[x0]);
  return top + fy * (bottom - top);
}

// ---- Lanes ---------------------------------------------------------------------------------

// Weighted sums of edge points, relative to an origin.
struct Moments {
  double w, x, y, xx, xy, yy;
  int points;
};
FIDUCIALS_HD inline Moments operator+(const Moments& a, const Moments& b) {
  return {a.w + b.w,   a.x + b.x,   a.y + b.y,           a.xx + b.xx,
          a.xy + b.xy, a.yy + b.yy, a.points + b.points};
}

// The sums of a chessboard corner's equations.
struct SaddleSums {
  double a, b, c, bx, by;
};
FIDUCIALS_HD inline SaddleSums operator+(const SaddleSums& p, const SaddleSums& q) {
  return {p.a + q.a, p.b + q.b, p.c + q.c, p.bx + q.bx, p.by + q.by};
}

// The 32 lanes run one after another on the CPU. sum() adds the lanes' parts as a warp's
// butterfly does: in round k, each lane adds the part of the lane 16 >> k away.
struct SerialLanes {
  template <typename T, typename Part>
  T sum(const Part& part) const {
    T values[kLanes];
    for (int lane = 0; lane < kLanes; ++lane) values[lane] = part(lane);
    for (int offset = kLanes / 2; offset > 0; offset /= 2) {
      T next[kLanes];
      for (int lane = 0; lane < kLanes; ++lane) next[lane] = values[lane] + values[lane ^ offset];
      for (int lane = 0; lane < kLanes; ++lane) values[lane] = next[lane];
    }
    return values[0];
  }
  // The smallest of the lanes' keys.
  template <typename Part>
  unsigned long long min(const Part& part) const {
    unsigned long long best = ~0ULL;
    for (int lane = 0; lane < kLanes; ++lane) {
      const unsigned long long key = part(lane);
      best = key < best ? key : best;
    }
    return best;
  }
  // out[k] = value(k) for k < count, the lanes taking every 32nd k.
  template <typename Value>
  void fill(double* out, int count, const Value& value) const {
    for (int k = 0; k < count; ++k) out[k] = value(k);
  }
};

// ---- 4. Edge refinement --------------------------------------------------------------------

// A line through (px, py) along the unit vector (dx, dy).
struct Line {
  double px, py, dx, dy;
};

// Where two lines meet; false if they are nearly parallel.
FIDUCIALS_HD inline bool intersect(const Line& a, const Line& b, double (&out)[2]) {
  const double cross = a.dx * b.dy - a.dy * b.dx;
  if (fabs(cross) < 1e-6) return false;
  const double t = ((b.px - a.px) * b.dy - (b.py - a.py) * b.dx) / cross;
  out[0] = a.px + t * a.dx;
  out[1] = a.py + t * a.dy;
  return true;
}

// Lane `lane`'s point on the edge from a to b of a dark quad with clockwise corners (image
// coordinates, y down), looking up to `range` pixels either side: where the image rises
// across the edge, relative to a, weighted by how steeply. Nothing for a lane past the edge's
// samples or where the image does not rise.
FIDUCIALS_HD inline Moments edge_point(const Image& image, const double (&a)[2],
                                       const double (&b)[2], double range, int min_contrast,
                                       int lane) {
  constexpr double kStep = 0.25;
  constexpr int kMaxSteps = 16;  // Per side: range up to 4 pixels.
  Moments none{};
  const double ex = b[0] - a[0];
  const double ey = b[1] - a[1];
  const double length = sqrt(ex * ex + ey * ey);
  int samples = static_cast<int>(length / 2.0);
  samples = samples < 4 ? 4 : (samples > kLanes ? kLanes : samples);
  if (lane >= samples) return none;
  const double nx = ey / length;  // Outward: from dark to bright.
  const double ny = -ex / length;
  int steps = static_cast<int>(range / kStep);
  steps = steps < 2 ? 2 : (steps > kMaxSteps ? kMaxSteps : steps);
  // Clear of the corners, where the other edges bend the profile.
  const double t = 0.15 + 0.7 * (lane + 0.5) / samples;
  const double px = t * ex;
  const double py = t * ey;
  double profile[2 * kMaxSteps + 3];
  for (int k = 0; k <= 2 * steps + 2; ++k) {
    const double s = (k - steps - 1) * kStep;
    profile[k] = sample(image, a[0] + px + s * nx, a[1] + py + s * ny);
  }
  int best = 1;
  double best_rise = -1.0;
  for (int k = 1; k <= 2 * steps + 1; ++k) {
    const double rise = profile[k + 1] - profile[k - 1];
    if (rise > best_rise) {
      best_rise = rise;
      best = k;
    }
  }
  // A step of the border's full contrast spread over two pixels rises about this much per step.
  if (best_rise < min_contrast * kStep / 4.0) return none;
  // The rise's centroid over the whole edge around the steepest step: for an edge sampled by
  // pixels and interpolated linearly, exactly where the edge is. (The rise is flat over such
  // an edge, so the steepest step alone could be anywhere along it.)
  const double floor = 0.1 * best_rise;
  int first = best;
  int last = best;
  while (first > 1 && profile[first] - profile[first - 2] > floor) --first;
  while (last < 2 * steps + 1 && profile[last + 2] - profile[last] > floor) ++last;
  double weight = 0.0;
  double offset = 0.0;
  for (int k = first; k <= last; ++k) {
    const double rise = profile[k + 1] - profile[k - 1];
    weight += rise;
    offset += rise * (k - steps - 1) * kStep;
  }
  offset /= weight;
  const double x = px + offset * nx;
  const double y = py + offset * ny;
  const double w = best_rise;
  return {w, w * x, w * y, w * x * x, w * x * y, w * y * y, 1};
}

// Fits the edge from a to b (as for edge_point); false with too few edge points.
template <typename Lanes>
FIDUCIALS_HD_NOINLINE inline bool fit_edge(const Lanes& lanes, const Image& image,
                                           const double (&a)[2], const double (&b)[2],
                                           double range, int min_contrast, Line& line) {
  const double ex = b[0] - a[0];
  const double ey = b[1] - a[1];
  const double length = sqrt(ex * ex + ey * ey);
  if (!(length > 2.0)) return false;
  const Moments m = lanes.template sum<Moments>(
      [&](int lane) { return edge_point(image, a, b, range, min_contrast, lane); });
  if (m.points < 3) return false;
  // The weighted points' principal direction.
  const double mx = m.x / m.w;
  const double my = m.y / m.w;
  const double cxx = m.xx / m.w - mx * mx;
  const double cxy = m.xy / m.w - mx * my;
  const double cyy = m.yy / m.w - my * my;
  const double half = 0.5 * (cxx - cyy);
  const double largest = 0.5 * (cxx + cyy) + sqrt(half * half + cxy * cxy);
  double dx = cxy;
  double dy = largest - cxx;
  if (fabs(cxx - largest) < fabs(cyy - largest)) {
    dx = largest - cyy;
    dy = cxy;
  }
  double norm = sqrt(dx * dx + dy * dy);
  if (!(norm > 0.0)) {
    // All points on one line along an axis: take the edge's own direction.
    dx = ex;
    dy = ey;
    norm = length;
  }
  line = Line{a[0] + mx, a[1] + my, dx / norm, dy / norm};
  return true;
}

// Refines a clockwise quad's corners in place; false, leaving them, if an edge cannot be
// fitted or a corner would move further than the search reached.
template <typename Lanes>
FIDUCIALS_HD_NOINLINE inline bool refine_quad(const Lanes& lanes, const Image& image,
                                              int min_contrast, double (&q)[4][2]) {
  constexpr double kRanges[3] = {4.0, 3.0, 3.0};
  double current[4][2];
  for (int i = 0; i < 4; ++i) {
    current[i][0] = q[i][0];
    current[i][1] = q[i][1];
  }
  FIDUCIALS_NO_UNROLL
  for (const double range : kRanges) {
    Line lines[4];
    FIDUCIALS_NO_UNROLL
    for (int e = 0; e < 4; ++e) {
      if (!fit_edge(lanes, image, current[e], current[(e + 1) % 4], range, min_contrast,
                    lines[e])) {
        return false;
      }
    }
    for (int i = 0; i < 4; ++i) {
      double corner[2];
      if (!intersect(lines[(i + 3) % 4], lines[i], corner)) return false;
      const double mx = corner[0] - current[i][0];
      const double my = corner[1] - current[i][1];
      if (!(mx * mx + my * my <= (range + 1.0) * (range + 1.0))) return false;
      current[i][0] = corner[0];
      current[i][1] = corner[1];
    }
  }
  for (int i = 0; i < 4; ++i) {
    q[i][0] = current[i][0];
    q[i][1] = current[i][1];
  }
  return true;
}

// ---- 5. Decoding ---------------------------------------------------------------------------

// Solves the augmented N x (N + 1) system `m` by Gauss-Jordan elimination with partial
// pivoting into x; false if a pivot is below `tiny`.
template <int N>
FIDUCIALS_HD inline bool solve(double (&m)[N][N + 1], double tiny, double* x) {
  for (int col = 0; col < N; ++col) {
    int pivot = col;
    for (int row = col + 1; row < N; ++row) {
      if (fabs(m[row][col]) > fabs(m[pivot][col])) pivot = row;
    }
    if (fabs(m[pivot][col]) < tiny) return false;
    for (int k = 0; k <= N; ++k) {
      const double swap = m[col][k];
      m[col][k] = m[pivot][k];
      m[pivot][k] = swap;
    }
    for (int row = 0; row < N; ++row) {
      if (row == col) continue;
      const double factor = m[row][col] / m[col][col];
      for (int k = col; k <= N; ++k) m[row][k] -= factor * m[col][k];
    }
  }
  for (int i = 0; i < N; ++i) x[i] = m[i][N] / m[i][i];
  return true;
}

// The homography (row-major, h[8] = 1) taking (0, 0), (s, 0), (s, s), (0, s) to q[0..3].
FIDUCIALS_HD_NOINLINE inline bool square_homography(const double (&q)[4][2], double s,
                                                    double (&h)[9]) {
  const double square[4][2] = {{0.0, 0.0}, {s, 0.0}, {s, s}, {0.0, s}};
  double a[8][9];
  for (int i = 0; i < 4; ++i) {
    const double x = square[i][0];
    const double y = square[i][1];
    const double u = q[i][0];
    const double v = q[i][1];
    const double r0[9] = {x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, u};
    const double r1[9] = {0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, v};
    for (int k = 0; k < 9; ++k) {
      a[2 * i][k] = r0[k];
      a[2 * i + 1][k] = r1[k];
    }
  }
  if (!solve(a, 1e-12, h)) return false;
  h[8] = 1.0;
  return true;
}

FIDUCIALS_HD inline void apply(const double (&h)[9], double x, double y, double& u, double& v) {
  const double w = h[6] * x + h[7] * y + h[8];
  u = (h[0] * x + h[1] * y + h[2]) / w;
  v = (h[3] * x + h[4] * y + h[5]) / w;
}

// The marker's inner cells as a code (Dictionary::code's packing), from the quad q; false if
// the cells lack contrast, the border is not black, or the marker has no white around it.
// `values` is the lanes' shared scratch.
template <typename Lanes>
FIDUCIALS_HD_NOINLINE inline bool read_code(const Lanes& lanes, const Image& image,
                                            const double (&q)[4][2], int marker_size,
                                            int min_contrast, double* values,
                                            std::uint64_t& code) {
  const int cells = marker_size + 2;
  double h[9];
  if (!square_homography(q, static_cast<double>(cells), h)) return false;
  // Each cell's mean over 3 x 3 points inside it, clear of blur.
  lanes.fill(values, cells * cells, [&](int k) {
    constexpr double kOffsets[3] = {0.3, 0.5, 0.7};
    double sum = 0.0;
    for (const double oy : kOffsets) {
      for (const double ox : kOffsets) {
        double u = 0.0;
        double v = 0.0;
        apply(h, k % cells + ox, k / cells + oy, u, v);
        sum += sample(image, u, v);
      }
    }
    return sum / 9.0;
  });
  double lo = 255.0;
  double hi = 0.0;
  for (int k = 0; k < cells * cells; ++k) {
    lo = values[k] < lo ? values[k] : lo;
    hi = values[k] > hi ? values[k] : hi;
  }
  if (hi - lo < min_contrast) return false;
  const double threshold = 0.5 * (lo + hi);
  // A few bright border cells are tolerated (glare, a partly covered edge), as in OpenCV.
  const int border_cells = 4 * (cells - 1);
  int bright_border = 0;
  for (int i = 0; i < cells; ++i) {
    bright_border += values[i] > threshold;                        // Top row.
    bright_border += values[(cells - 1) * cells + i] > threshold;  // Bottom row.
    if (i > 0 && i < cells - 1) {
      bright_border += values[i * cells] > threshold;              // Left column.
      bright_border += values[i * cells + cells - 1] > threshold;  // Right column.
    }
  }
  if (5 * bright_border > border_cells) return false;  // At most 20%.
  // A printed marker has white around it (in a ChArUco board, the square's margin): a dark frame
  // around a pattern in a texture does not. Sampled 0.4 cells out from each border cell, where
  // that is in the image; at least 70% must be bright.
  int outside = 0;
  int bright_outside = 0;
  for (int side = 0; side < 4; ++side) {
    for (int i = 0; i < cells; ++i) {
      const double along = i + 0.5;
      const double x = side == 0 ? along : side == 1 ? cells + 0.4 : side == 2 ? along : -0.4;
      const double y = side == 0 ? -0.4 : side == 1 ? along : side == 2 ? cells + 0.4 : along;
      double u = 0.0;
      double v = 0.0;
      apply(h, x, y, u, v);
      if (!(u >= 0.0 && v >= 0.0 && u <= image.width - 1.0 && v <= image.height - 1.0)) continue;
      ++outside;
      bright_outside += sample(image, u, v) > threshold;
    }
  }
  if (10 * bright_outside < 7 * outside) return false;
  code = 0;
  for (int row = 1; row <= marker_size; ++row) {
    for (int col = 1; col <= marker_size; ++col) {
      code = (code << 1U) | (values[row * cells + col] > threshold ? 1U : 0U);
    }
  }
  return true;
}

// The dictionary marker closest to `code` in any rotation, as a key: distance << 20 | id << 2 |
// turns. `rotated` holds each marker's code turned 0, 1, 2 and 3 quarter turns clockwise. Ties
// go to the lowest id and turn.
template <typename Lanes>
FIDUCIALS_HD_NOINLINE inline unsigned long long identify(const Lanes& lanes, std::uint64_t code,
                                                         const std::uint64_t* rotated, int count) {
  return lanes.min([&](int lane) {
    unsigned long long best = ~0ULL;
    for (int marker = lane; marker < count; marker += kLanes) {
      for (int r = 0; r < 4; ++r) {
        const auto distance =
            static_cast<unsigned long long>(popcount64(code ^ rotated[marker * 4 + r]));
        const unsigned long long key =
            (distance << 20U) | (static_cast<unsigned long long>(marker) << 2U) |
            static_cast<unsigned long long>(r);
        best = key < best ? key : best;
      }
    }
    return best;
  });
}

// ---- 3 to 5 for one candidate --------------------------------------------------------------

FIDUCIALS_HD inline double cross(const double (&o)[2], const double (&a)[2], const double (&b)[2]) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

// A clockwise (image coordinates) convex quad whose sides are all at least `min_side` long.
FIDUCIALS_HD inline bool valid_quad(const double (&q)[4][2], double min_side) {
  for (int i = 0; i < 4; ++i) {
    const double* a = q[i];
    const double* b = q[(i + 1) % 4];
    const double dx = b[0] - a[0];
    const double dy = b[1] - a[1];
    if (!(dx * dx + dy * dy >= min_side * min_side)) return false;
    if (!(cross(q[i], q[(i + 1) % 4], q[(i + 2) % 4]) > 0.0)) return false;
  }
  return true;
}

// Examines a candidate whose passes are done. `rotated` as for identify(); `scratch` holds
// kScratch doubles the lanes share.
template <typename Lanes>
FIDUCIALS_HD inline CandidateResult examine(const Lanes& lanes, const Candidate& c,
                                            const Image& image, const Params& p,
                                            const std::uint64_t* rotated, double* scratch) {
  CandidateResult result{};
  // A component touching the image's edge may be a marker cut by it.
  if (c.min_x <= 0 || c.min_y <= 0 || c.max_x >= p.width - 1 || c.max_y >= p.height - 1) {
    return result;
  }
  if (c.far0 == 0 || c.far2 == 0 || c.side_pos == 0 || c.side_neg == 0) return result;
  const std::uint32_t index[4] = {key_index(c.far0), key_index(c.side_pos), key_index(c.far2),
                                  key_index(c.side_neg)};
  double q[4][2];
  for (int i = 0; i < 4; ++i) {
    q[i][0] = index[i] % p.width;
    q[i][1] = index[i] / p.width;
  }
  // Clockwise in image coordinates: positive area.
  double area = 0.0;
  for (int i = 0; i < 4; ++i) area += q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1];
  if (area < 0.0) {
    for (int k = 0; k < 2; ++k) {
      const double swap = q[1][k];
      q[1][k] = q[3][k];
      q[3][k] = swap;
    }
  }
  if (!valid_quad(q, p.min_marker_px)) return result;
  // The corner pixels' centers lie about half a pixel inside the border's outer corners.
  const double mx = 0.25 * (q[0][0] + q[1][0] + q[2][0] + q[3][0]);
  const double my = 0.25 * (q[0][1] + q[1][1] + q[2][1] + q[3][1]);
  for (int i = 0; i < 4; ++i) {
    const double dx = q[i][0] - mx;
    const double dy = q[i][1] - my;
    const double norm = sqrt(dx * dx + dy * dy);
    q[i][0] += 0.70710678118654752 * dx / norm;
    q[i][1] += 0.70710678118654752 * dy / norm;
  }
  std::uint64_t code = 0;
  // Most candidates are not markers; reading the unrefined quad rejects them before the edges
  // are fitted.
  if (!read_code(lanes, image, q, p.marker_size, p.min_contrast, scratch, code)) return result;
  if (p.refine) {
    refine_quad(lanes, image, p.min_contrast, q);
    if (!valid_quad(q, p.min_marker_px)) return result;
    if (!read_code(lanes, image, q, p.marker_size, p.min_contrast, scratch, code)) return result;
  }
  const unsigned long long best = identify(lanes, code, rotated, p.dictionary_count);
  const int distance = static_cast<int>(best >> 20U);
  if (distance > p.max_hamming) return result;
  const int turns = static_cast<int>(best & 3U);
  result.marker = 1;
  result.id = static_cast<int>((best >> 2U) & 0x3FFFFU);
  result.hamming = distance;
  // The marker turned `turns` quarters clockwise: its top-left corner is the quad's corner
  // `turns`.
  for (int i = 0; i < 4; ++i) {
    result.corners[i][0] = q[(i + turns) % 4][0];
    result.corners[i][1] = q[(i + turns) % 4][1];
  }
  return result;
}

// ---- Chessboard corners --------------------------------------------------------------------

// Whether (x, y) is a chessboard corner: going once around it at `radius`, the image turns
// from dark to bright and back twice, with at least min_contrast between them, and opposite
// points have the same color. Where the edge of something in front of the board crosses a
// chessboard edge, gradients point two ways too, but the image turns only once, or the colors
// are not opposite.
FIDUCIALS_HD inline bool is_saddle(const Image& image, double x, double y, double radius,
                                   int min_contrast) {
  constexpr int kSamples = 16;
  // cos of 2 pi k / 16 for k = 0 .. 4; the rest follow by symmetry.
  constexpr double kCos[5] = {1.0, 0.92387953251128674, 0.70710678118654752,
                              0.38268343236508977, 0.0};
  radius = radius < 1.5 ? 1.5 : radius;
  double values[kSamples];
  double lo = 255.0;
  double hi = 0.0;
  for (int k = 0; k < kSamples; ++k) {
    const int step = k % 4;
    // (cos, sin) of the angle within its quarter, turned `k / 4` quarters.
    double dx = kCos[step];
    double dy = kCos[4 - step];
    for (int q = 0; q < k / 4; ++q) {
      const double t = dx;
      dx = -dy;
      dy = t;
    }
    values[k] = sample(image, x + radius * dx, y + radius * dy);
    lo = values[k] < lo ? values[k] : lo;
    hi = values[k] > hi ? values[k] : hi;
  }
  if (hi - lo < min_contrast) return false;
  const double threshold = 0.5 * (lo + hi);
  int turns = 0;
  int asymmetric = 0;  // Opposite points on different sides of the threshold.
  for (int k = 0; k < kSamples; ++k) {
    turns += (values[k] > threshold) != (values[(k + 1) % kSamples] > threshold);
    if (k < kSamples / 2) {
      asymmetric += (values[k] > threshold) != (values[k + kSamples / 2] > threshold);
    }
  }
  // Opposite sectors of a chessboard corner have one color; points on its two edges may differ.
  return turns == 4 && asymmetric <= 2;
}

// Moves (x, y) to the saddle point of the gray image within `half_window` pixels: the point
// every gradient around it points across, by OpenCV cornerSubPix's iteration (with each
// gradient weighted by its magnitude rather than its square). False if the window holds no
// corner, the point leaves the window, or what it found is not a chessboard corner (see
// is_saddle). `scratch` holds kScratch doubles the lanes share.
template <typename Lanes>
FIDUCIALS_HD_NOINLINE inline bool refine_saddle(const Lanes& lanes, const Image& image,
                                                int half_window, int min_contrast,
                                                double* scratch, double& x, double& y) {
  constexpr int kIterations = 40;
  constexpr double kEpsilon = 0.001;
  if (half_window < 1 || half_window > kMaxHalfWindow) return false;
  const double x0 = x;
  const double y0 = y;
  const int side = 2 * half_window + 3;  // The window and a pixel around it, for gradients.
  const int inner = side - 2;
  const double reach = (half_window + 1.0) * (half_window + 1.0);
  FIDUCIALS_NO_UNROLL
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    lanes.fill(scratch, side * side, [&](int k) {
      return sample(image, x + k % side - half_window - 1, y + k / side - half_window - 1);
    });
    const double* patch = scratch;
    const SaddleSums s = lanes.template sum<SaddleSums>([&](int lane) {
      SaddleSums part{};
      for (int k = lane; k < inner * inner; k += kLanes) {
        const int i = k % inner + 1;
        const int j = k / inner + 1;
        const double dx = i - half_window - 1.0;
        const double dy = j - half_window - 1.0;
        // A smooth, round weight, falling to zero just outside the window.
        double w = 1.0 - (dx * dx + dy * dy) / reach;
        if (w <= 0.0) continue;
        w *= w;
        const double gx = 0.5 * (patch[j * side + i + 1] - patch[j * side + i - 1]);
        const double gy = 0.5 * (patch[(j + 1) * side + i] - patch[(j - 1) * side + i]);
        // Weighted by the gradient's magnitude, not its square as in cornerSubPix: squared, a
        // sharp edge's steepest pixel dominates and pulls the corner toward pixel centers.
        const double magnitude = sqrt(gx * gx + gy * gy);
        if (!(magnitude > 0.0)) continue;
        w /= magnitude;
        const double gxx = w * gx * gx;
        const double gxy = w * gx * gy;
        const double gyy = w * gy * gy;
        part.a += gxx;
        part.b += gxy;
        part.c += gyy;
        part.bx += gxx * dx + gxy * dy;
        part.by += gxy * dx + gyy * dy;
      }
      return part;
    });
    const double det = s.a * s.c - s.b * s.b;
    // Gradients in a single direction (an edge, not a corner) leave the system singular.
    if (!(det > 1e-6 * (s.a + s.c) * (s.a + s.c))) return false;
    const double sx = (s.c * s.bx - s.b * s.by) / det;
    const double sy = (s.a * s.by - s.b * s.bx) / det;
    x += sx;
    y += sy;
    const double mx = x - x0;
    const double my = y - y0;
    if (!(mx * mx + my * my <= static_cast<double>(half_window) * half_window)) return false;
    if (sx * sx + sy * sy < kEpsilon * kEpsilon) break;
  }
  return is_saddle(image, x, y, 0.75 * half_window, min_contrast);
}

}  // namespace holoscan::fiducials::core
