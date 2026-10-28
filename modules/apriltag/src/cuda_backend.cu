// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CUDA backend's second half: fit quads to the segmenter's corner
// candidates, refine and decode them, drop duplicates; one copy to the host.
// Compiled without FMA contraction so it rounds as AprilTag 3 does on a CPU.
// Pixel centers are at +0.5, as in AprilTag 3, until results are reported.

#include <apriltag.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// CCCL's headers are C++; cpplint takes any angle-bracketed non-.hpp header for
// a C system header.
#include <cub/warp/warp_reduce.cuh>  // NOLINT(build/include_order)
#include <holoscan/apriltag/detector_backend.hpp>

#include "gpu_segmenter.hpp"
#include "homography.hpp"
#include "tag_family.hpp"

namespace holoscan::apriltag {
namespace {

// Corner candidates kept per cluster, as in AprilTag 3.
constexpr int kMaxCornerCandidates = 10;
// Most quads one frame can produce; more are counted and dropped.
constexpr std::uint32_t kMaxQuads = 1024;
// Most detections reported per frame; more are counted and dropped.
constexpr auto kMaxResults = static_cast<std::uint32_t>(kMaxDetectionsPerFrame);
// Adjacent edges whose directions' |cos| exceeds this are rejected (10 deg).
constexpr double kMaxEdgeCos = 0.984807753;
// Largest mean squared line-fit error per edge, in decimated pixels (AprilTag 3's
// units; see LineMoments).
constexpr double kMaxLineFitMse = 10.0;
// Bits of the largest family, tagStandard52h13, and cells across its grid.
constexpr int kMaxBits = 64;
constexpr int kMaxGrid = 12;
// Most quad candidates fitted per frame; more are counted and dropped.
constexpr std::uint32_t kMaxJobs = 4096;
// Most samples taken along one edge when refining it: enough for 8192 pixels.
constexpr int kMaxEdgeSamples = 1024;
constexpr int kBlock = 128;
// Threads that rank one frame's decodes.
constexpr int kSelectThreads = 256;

struct Line {
  double point[2];      // The weighted centroid of its samples.
  double direction[2];  // Unit direction.
  double error;         // Mean squared perpendicular distance times the point count.
  double mse;           // Weighted mean squared perpendicular distance.
};

struct Quad {
  double corners[4][2];  // Clockwise in the image (y down), full-resolution pixels.
  std::uint32_t cluster;
  bool reversed;  // The inside is brighter than the outside.
};

struct DecodedTag {
  bool valid;
  std::int32_t id;
  std::int32_t hamming;
  float margin;
  std::uint32_t cluster;
  double center[2];
  double corners[4][2];
  double shortest_side;  // Of the corners' quad, for the duplicate check.
};

// The one block copied to the host per frame. Each count includes what did not
// fit, so the host can tell what the frame lost.
struct FrameResult {
  std::uint32_t count;       // Detections after duplicates are removed.
  std::uint32_t quads;       // Quads fitted.
  std::uint32_t candidates;  // Quad candidates (fit jobs).
  std::uint32_t points;      // Boundary points found.
  DecodedTag tags[kMaxResults];
};

struct TableEntry {
  std::uint64_t code;
  std::int32_t id;  // -1 marks an empty slot.
  std::int32_t hamming;
};

// A tag family and the decode settings, passed to the decode kernel by value.
struct FamilyParams {
  const TableEntry* table;
  std::uint64_t table_mask;
  double sharpening;
  std::int32_t nbits;
  std::int32_t width_at_border;
  std::int32_t total_width;
  bool reversed_border;
  bool refine_edges;
  std::int32_t decimate;
  std::int8_t bit_x[kMaxBits];
  std::int8_t bit_y[kMaxBits];
  std::int8_t rotation[kMaxBits];  // The bit a quarter turn moves each bit to.
};

struct FitParams {
  bool normal_border;
  bool reversed_border;
  double min_tag_width;  // Smallest tag side, in decimated pixels.
  double decimate;
  std::uint32_t max_cluster_points;
};

// A full-resolution device gray image.
struct DeviceGray {
  const std::uint8_t* data;
  int width;
  int height;
  std::size_t pitch;

  // Bilinear sample at (x, y) with pixel centers at +0.5; false outside.
  __device__ bool sample(double x, double y, double& value) const {
    const double fx = x - 0.5;
    const double fy = y - 0.5;
    const double x0 = floor(fx);
    const double y0 = floor(fy);
    // Written so that a NaN coordinate fails too.
    if (!(x0 >= 0 && y0 >= 0 && x0 + 1 < width && y0 + 1 < height)) {
      return false;
    }
    const double ax = fx - x0;
    const double ay = fy - y0;
    const std::uint8_t* row0 = data + static_cast<std::size_t>(y0) * pitch;
    const std::uint8_t* row1 = row0 + pitch;
    const auto ix = static_cast<std::size_t>(x0);
    value = (1 - ax) * (1 - ay) * row0[ix] + ax * (1 - ay) * row0[ix + 1] +
            (1 - ax) * ay * row1[ix] + ax * ay * row1[ix + 1];
    return true;
  }
};

__host__ __device__ std::uint64_t mix(std::uint64_t code) {
  code ^= code >> 33;
  code *= 0xff51afd7ed558ccdULL;
  code ^= code >> 33;
  return code;
}

// The best line through `m`'s `n` points. The direction is the covariance's
// major eigenvector, computed only when the fit is within `max_mse`.
__device__ Line fit_line(const LineMoments& m, double n, double max_mse = HUGE_VAL) {
  Line line{};
  const Covariance c(m);
  const double smallest = c.smallest();
  line.point[0] = c.ex;
  line.point[1] = c.ey;
  line.mse = smallest;
  line.error = smallest * n;
  if (smallest > max_mse) {
    return line;
  }
  // Both vectors lie along the major axis; the longer is better conditioned.
  const double largest = c.half_trace + c.spread;
  const double ax = c.cxy;
  const double ay = largest - c.cxx;
  const double bx = largest - c.cyy;
  const double by = c.cxy;
  const double la = ax * ax + ay * ay;
  const double lb = bx * bx + by * by;
  if (la == 0.0 && lb == 0.0) {
    line.direction[0] = 1.0;  // Isotropic: any direction fits equally.
    line.direction[1] = 0.0;
  } else if (la > lb) {
    const double length = sqrt(la);
    line.direction[0] = ax / length;
    line.direction[1] = ay / length;
  } else {
    const double length = sqrt(lb);
    line.direction[0] = bx / length;
    line.direction[1] = by / length;
  }
  return line;
}

__device__ bool intersect(const Line& a, const Line& b, double (&out)[2]) {
  const double det = a.direction[1] * b.direction[0] - a.direction[0] * b.direction[1];
  if (fabs(det) < 1e-9) {
    return false;
  }
  const double dx = b.point[0] - a.point[0];
  const double dy = b.point[1] - a.point[1];
  const double t = (dy * b.direction[0] - dx * b.direction[1]) / det;
  out[0] = a.point[0] + t * a.direction[0];
  out[1] = a.point[1] + t * a.direction[1];
  return true;
}

__device__ bool corners_acceptable(const double (&corners)[4][2], double min_tag_width) {
  double area = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double* prev = corners[(i + 3) % 4];
    const double* here = corners[i];
    const double* next = corners[(i + 1) % 4];
    // Clockwise in y-down image coordinates means every turn is positive.
    if ((here[0] - prev[0]) * (next[1] - prev[1]) - (here[1] - prev[1]) * (next[0] - prev[0]) <=
        0.0) {
      return false;
    }
    const double ax = prev[0] - here[0];
    const double ay = prev[1] - here[1];
    const double bx = next[0] - here[0];
    const double by = next[1] - here[1];
    const double cosine = (ax * bx + ay * by) / sqrt((ax * ax + ay * ay) * (bx * bx + by * by));
    if (fabs(cosine) > kMaxEdgeCos) {
      return false;
    }
    area += here[0] * next[1] - next[0] * here[1];
  }
  // Corners are in decimated pixels here.
  return 0.5 * fabs(area) >= 0.95 * min_tag_width * min_tag_width;
}

struct Candidate {
  std::uint32_t index;  // Into the cluster's angle-ordered points.
  LineMoments through;  // Moments of the cluster's points 0 .. index.
  LineMoments before;   // Moments of the cluster's points 0 .. index - 1.
};

// A cluster worth fitting: its corner candidates, in index order.
struct FitJob {
  std::uint32_t cluster;
  std::uint32_t points;
  std::int32_t candidates;
  bool reversed;
  LineMoments total;
  Candidate candidate[kMaxCornerCandidates];
};

// The line through the points from candidate `from` to `to`, both included,
// wrapping past the cluster's end.
__device__ Line fit_between(const FitJob& job, const Candidate& from, const Candidate& to,
                            double max_mse = HUGE_VAL) {
  const bool wraps = from.index > to.index;
  const double n = wraps ? static_cast<double>(job.points) - from.index + to.index + 1.0
                         : to.index - from.index + 1.0;
  return fit_line(range_moments(from.before, to.through, job.total, wraps), n, max_mse);
}

// One thread per cluster: its strongest corner candidates (all of them when
// there are at most kMaxCornerCandidates, otherwise those stronger than the
// next one, as AprilTag 3 does), queued for fitting when there are four.
__global__ void candidates_kernel(ClusterData data, FitParams params, FitJob* jobs,
                                  std::uint32_t* job_count) {
  const std::uint32_t cluster = blockIdx.x * blockDim.x + threadIdx.x;
  if (cluster >= data.clusters()) {
    return;
  }
  const int found = static_cast<int>(data.maxima_counts[cluster]);
  if (found < 4 || data.raw_points[cluster] > params.max_cluster_points) {
    return;
  }
  const std::int32_t* box = data.box + 4 * cluster;
  if ((box[1] - box[0]) < 2.0 * params.min_tag_width ||
      (box[3] - box[2]) < 2.0 * params.min_tag_width) {
    return;
  }
  // Gradients point from dark to bright: outward across a normal border.
  const bool reversed = data.polarity[cluster] < 0.0;
  if (reversed ? !params.reversed_border : !params.normal_border) {
    return;
  }
  const std::uint32_t offset = data.offsets[cluster];

  // The kMaxCornerCandidates + 1 strongest maxima, strongest first.
  double value[kMaxCornerCandidates + 1];
  int at[kMaxCornerCandidates + 1];
  const std::uint32_t* mine = data.maxima + data.maxima_offsets[cluster];
  int listed = 0;
  for (int m = 0; m < found; ++m) {
    const auto index = mine[m];
    const double here = data.smoothed[index];
    if (listed == kMaxCornerCandidates + 1 && here <= value[kMaxCornerCandidates]) {
      continue;
    }
    int slot = min(listed, kMaxCornerCandidates);
    while (slot > 0 && value[slot - 1] < here) {
      value[slot] = value[slot - 1];
      at[slot] = at[slot - 1];
      --slot;
    }
    value[slot] = here;
    at[slot] = static_cast<int>(index - offset);
    listed = min(listed + 1, kMaxCornerCandidates + 1);
  }
  const double floor_value = found > kMaxCornerCandidates ? value[kMaxCornerCandidates] : -1.0;
  int chosen[kMaxCornerCandidates];
  int kept = 0;
  for (int k = 0; k < listed && kept < kMaxCornerCandidates; ++k) {
    if (value[k] > floor_value) {
      chosen[kept++] = k;
    }
  }
  if (kept < 4) {
    return;
  }
  const std::uint32_t slot = atomicAdd(job_count, 1U);
  if (slot >= kMaxJobs) {
    return;
  }
  // Index order, so the four edges between candidates follow the boundary.
  for (int a = 1; a < kept; ++a) {
    const int moving = chosen[a];
    int b = a;
    while (b > 0 && at[chosen[b - 1]] > at[moving]) {
      chosen[b] = chosen[b - 1];
      --b;
    }
    chosen[b] = moving;
  }
  FitJob& job = jobs[slot];
  const LineMoments* prefix = data.prefix + offset;
  job.cluster = cluster;
  job.points = data.point_counts[cluster];
  job.candidates = kept;
  job.reversed = reversed;
  job.total = prefix[job.points - 1];
  for (int k = 0; k < kept; ++k) {
    const int i = at[chosen[k]];
    job.candidate[k].index = static_cast<std::uint32_t>(i);
    job.candidate[k].through = prefix[i];
    job.candidate[k].before = i == 0 ? LineMoments{} : prefix[i - 1];
  }
}

// One warp per job: the quad through the four candidates whose four edges fit
// best. Each lane searches its own first two corners; ties go to the first
// combination in order, as in a sequential search.
__global__ void fit_quads_kernel(const FitJob* jobs, const std::uint32_t* job_count,
                                 FitParams params, Quad* quads, std::uint32_t* quad_count) {
  const std::uint32_t index = blockIdx.x;
  const int lane = static_cast<int>(threadIdx.x);
  if (index >= min(*job_count, kMaxJobs)) {
    return;
  }
  const FitJob& job = jobs[index];
  const int kept = job.candidates;
  const Candidate* candidates = job.candidate;
  double best_error = HUGE_VAL;
  int best_rank = INT_MAX;  // a, b, c, d as decimal digits: the search order.
  int pair = 0;
  for (int a = 0; a + 3 < kept; ++a) {
    for (int b = a + 1; b + 2 < kept; ++b, ++pair) {
      if (pair % 32 != lane) continue;
      const Line l0 = fit_between(job, candidates[a], candidates[b], kMaxLineFitMse);
      if (l0.mse > kMaxLineFitMse) continue;
      for (int c = b + 1; c + 1 < kept; ++c) {
        const Line l1 = fit_between(job, candidates[b], candidates[c], kMaxLineFitMse);
        if (l1.mse > kMaxLineFitMse) continue;
        const double dot01 = l0.direction[0] * l1.direction[0] + l0.direction[1] * l1.direction[1];
        if (fabs(dot01) > kMaxEdgeCos) continue;
        for (int d = c + 1; d < kept; ++d) {
          // Only the error counts here; the direction is not needed.
          const Line l2 = fit_between(job, candidates[c], candidates[d], 0.0);
          if (l2.mse > kMaxLineFitMse) continue;
          const Line l3 = fit_between(job, candidates[d], candidates[a], 0.0);
          if (l3.mse > kMaxLineFitMse) continue;
          const double sum = l0.error + l1.error + l2.error + l3.error;
          if (sum < best_error) {
            best_error = sum;
            best_rank = ((a * 10 + b) * 10 + c) * 10 + d;
          }
        }
      }
    }
  }
  // The warp's lowest (error, rank): a total order, so any reduction order agrees.
  struct Best {
    double error;
    int rank;
  };
  __shared__ cub::WarpReduce<Best>::TempStorage reduce_storage;
  const Best found =
      cub::WarpReduce<Best>(reduce_storage)
          .Reduce(Best{best_error, best_rank}, [](const Best& a, const Best& b) {
            return b.error < a.error || (b.error == a.error && b.rank < a.rank) ? b : a;
          });
  if (lane != 0 || found.error == HUGE_VAL) {
    return;
  }
  const int r = found.rank;
  const int best[4] = {r / 1000, r / 100 % 10, r / 10 % 10, r % 10};
  Line lines[4];
  for (int i = 0; i < 4; ++i) {
    lines[i] = fit_between(job, candidates[best[i]], candidates[best[(i + 1) % 4]]);
  }
  double corners[4][2];
  for (int i = 0; i < 4; ++i) {
    if (!intersect(lines[(i + 3) % 4], lines[i], corners[i])) {
      return;
    }
  }
  if (!corners_acceptable(corners, params.min_tag_width)) {
    return;
  }
  const std::uint32_t slot = atomicAdd(quad_count, 1U);
  if (slot >= kMaxQuads) {
    return;
  }
  Quad& quad = quads[slot];
  quad.cluster = job.cluster;
  quad.reversed = job.reversed;
  for (int i = 0; i < 4; ++i) {
    // A decimated pixel k spans full-resolution [k * d, (k + 1) * d).
    quad.corners[i][0] = corners[i][0] * params.decimate;
    quad.corners[i][1] = corners[i][1] * params.decimate;
  }
}

// Moves each edge of `quad` onto the strongest nearby intensity step of the
// full-resolution image, searching up to `decimate + 1` pixels to either side.
// The block's threads take the samples along each edge; one thread then sums
// them in order and fits the edges.
__device__ void refine_edges(Quad& quad, const DeviceGray& image, int decimate) {
  __shared__ double sample_x[kMaxEdgeSamples];
  __shared__ double sample_y[kMaxEdgeSamples];
  __shared__ bool sample_ok[kMaxEdgeSamples];
  __shared__ Line lines[4];
  __shared__ bool usable;
  double centroid[2] = {0.0, 0.0};
  for (const auto& corner : quad.corners) {
    centroid[0] += 0.25 * corner[0];
    centroid[1] += 0.25 * corner[1];
  }
  if (threadIdx.x == 0) {
    usable = true;
  }
  for (int edge = 0; edge < 4; ++edge) {
    const double* a = quad.corners[edge];
    const double* b = quad.corners[(edge + 1) % 4];
    const double length = hypot(b[0] - a[0], b[1] - a[1]);
    if (length < 1e-6) {
      return;  // The same on every thread.
    }
    double nx = (b[1] - a[1]) / length;
    double ny = -(b[0] - a[0]) / length;
    // Point the normal at the bright side: outward for a normal border.
    const double mid_x = 0.5 * (a[0] + b[0]) - centroid[0];
    const double mid_y = 0.5 * (a[1] + b[1]) - centroid[1];
    if ((nx * mid_x + ny * mid_y < 0.0) != quad.reversed) {
      nx = -nx;
      ny = -ny;
    }
    const int samples = min(kMaxEdgeSamples, max(16, static_cast<int>(length / 8.0)));
    const int range = decimate + 1;
    constexpr int kStepsPerPixel = 4;
    __syncthreads();  // The previous edge's samples are summed.
    for (int s = static_cast<int>(threadIdx.x); s < samples; s += static_cast<int>(blockDim.x)) {
      // Skip the ends: corners are the least reliable part of an edge.
      const double alpha = (1.0 + s) / (samples + 1);
      const double x0 = alpha * a[0] + (1 - alpha) * b[0];
      const double y0 = alpha * a[1] + (1 - alpha) * b[1];
      double weighted = 0.0;
      double total = 0.0;
      for (int step = 0; step <= 2 * kStepsPerPixel * range; ++step) {
        const double offset = -range + static_cast<double>(step) / kStepsPerPixel;
        double bright = 0.0;
        double dark = 0.0;
        if (!image.sample(x0 + (offset + 1.0) * nx, y0 + (offset + 1.0) * ny, bright) ||
            !image.sample(x0 + (offset - 1.0) * nx, y0 + (offset - 1.0) * ny, dark) ||
            bright < dark) {
          continue;
        }
        const double weight = (bright - dark) * (bright - dark);
        weighted += weight * offset;
        total += weight;
      }
      sample_ok[s] = total != 0.0;
      if (total != 0.0) {
        const double offset = weighted / total;
        sample_x[s] = x0 + offset * nx;
        sample_y[s] = y0 + offset * ny;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      LineMoments fit{};
      for (int s = 0; s < samples; ++s) {
        if (!sample_ok[s]) continue;
        const double x = sample_x[s];
        const double y = sample_y[s];
        fit.mx += x;
        fit.my += y;
        fit.mxx += x * x;
        fit.mxy += x * y;
        fit.myy += y * y;
        fit.w += 1.0;
      }
      if (fit.w < 2) {
        usable = false;  // Leave the quad as fitted.
      } else {
        lines[edge] = fit_line(fit, fit.w);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x != 0 || !usable) {
    return;
  }
  double refined[4][2];
  for (int i = 0; i < 4; ++i) {
    if (!intersect(lines[(i + 3) % 4], lines[i], refined[i])) {
      return;
    }
  }
  for (int i = 0; i < 4; ++i) {
    quad.corners[i][0] = refined[i][0];
    quad.corners[i][1] = refined[i][1];
  }
}

// A plane through gray samples, value = a * x + b * y + c in tag coordinates,
// so the black/white threshold can follow lighting across the tag.
struct GrayModel {
  double normal[3][3]{};
  double rhs[3]{};
  double coefficients[3]{};

  __device__ void add(double x, double y, double value) {
    const double row[3] = {x, y, 1.0};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        normal[i][j] += row[i] * row[j];
      }
      rhs[i] += row[i] * value;
    }
  }

  // Least squares; with too few independent samples, the mean only.
  __device__ void solve() {
    double m[3][4];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        m[i][j] = normal[i][j];
      }
      m[i][3] = rhs[i];
    }
    if (!holoscan::apriltag::solve(m, 1e-9, coefficients)) {
      coefficients[0] = 0.0;
      coefficients[1] = 0.0;
      coefficients[2] = normal[2][2] > 0.0 ? rhs[2] / normal[2][2] : 0.0;
    }
  }

  [[nodiscard]] __device__ double at(double x, double y) const {
    return coefficients[0] * x + coefficients[1] * y + coefficients[2];
  }
};

__device__ const TableEntry* find_code(const FamilyParams& family, std::uint64_t code) {
  for (std::uint64_t slot = mix(code) & family.table_mask;; slot = (slot + 1) & family.table_mask) {
    const TableEntry& entry = family.table[slot];
    if (entry.id < 0) return nullptr;
    if (entry.code == code) return &entry;
  }
}

__device__ std::uint64_t rotate(const FamilyParams& family, std::uint64_t code) {
  const int nbits = family.nbits;
  std::uint64_t rotated = 0;
  for (int i = 0; i < nbits; ++i) {
    if ((code >> (nbits - 1 - i)) & 1U) {
      rotated |= std::uint64_t{1} << (nbits - 1 - family.rotation[i]);
    }
  }
  return rotated;
}

// Reads `quad`'s bits and matches them against the family. False if it
// decodes to no code within the family's correctable bit errors.
__device__ bool decode(const Quad& quad, const DeviceGray& image, const FamilyParams& family,
                       DecodedTag& out) {
  // homography_from_corners maps (-1, 1), (1, 1), (1, -1), (-1, -1) in order.
  // The quad's corners run clockwise in the image, as tag (-1, -1), (1, -1),
  // (1, 1), (-1, 1) do, so this sends tag (-1, -1) to corner 0 without a mirror.
  const double reordered[4][2] = {{quad.corners[3][0], quad.corners[3][1]},
                                  {quad.corners[2][0], quad.corners[2][1]},
                                  {quad.corners[1][0], quad.corners[1][1]},
                                  {quad.corners[0][0], quad.corners[0][1]}};
  double h[9];
  if (!homography_from_corners(reordered, h)) {
    return false;
  }
  const double border = family.width_at_border;
  const auto to_tag = [border](double cell) { return 2.0 * ((cell + 0.5) / border - 0.5); };

  // Sample the known white ring outside the border and black ring inside it.
  GrayModel white;
  GrayModel black;
  struct Pattern {
    double x0, y0, dx, dy;
    bool white;
  };
  const Pattern patterns[8] = {
      {-1.0, 0.0, 0.0, 1.0, true},   {0.0, 0.0, 0.0, 1.0, false},
      {border, 0.0, 0.0, 1.0, true}, {border - 1.0, 0.0, 0.0, 1.0, false},
      {0.0, -1.0, 1.0, 0.0, true},   {0.0, 0.0, 1.0, 0.0, false},
      {0.0, border, 1.0, 0.0, true}, {0.0, border - 1.0, 1.0, 0.0, false},
  };
  for (const Pattern& pattern : patterns) {
    for (int i = 0; i < family.width_at_border; ++i) {
      const double tx = to_tag(pattern.x0 + i * pattern.dx);
      const double ty = to_tag(pattern.y0 + i * pattern.dy);
      double p[2];
      project(h, tx, ty, p);
      const double ix = floor(p[0]);
      const double iy = floor(p[1]);
      if (!(ix >= 0 && iy >= 0 && ix < image.width && iy < image.height)) continue;
      const double value =
          image.data[static_cast<std::size_t>(iy) * image.pitch + static_cast<std::size_t>(ix)];
      (pattern.white ? white : black).add(tx, ty, value);
    }
  }
  white.solve();
  black.solve();
  if ((white.at(0, 0) - black.at(0, 0) < 0) != family.reversed_border) {
    return false;
  }

  // Bit values relative to the local threshold, on the family's full grid so
  // they can be sharpened with their neighbors.
  const int size = family.total_width;
  const int min_coord = (family.width_at_border - family.total_width) / 2;
  double values[kMaxGrid * kMaxGrid];
  for (int i = 0; i < size * size; ++i) values[i] = 0.0;
  for (int i = 0; i < family.nbits; ++i) {
    const double tx = to_tag(family.bit_x[i]);
    const double ty = to_tag(family.bit_y[i]);
    double p[2];
    project(h, tx, ty, p);
    double value = 0.0;
    if (!image.sample(p[0], p[1], value)) continue;
    const double threshold = 0.5 * (black.at(tx, ty) + white.at(tx, ty));
    values[(family.bit_y[i] - min_coord) * size + family.bit_x[i] - min_coord] = value - threshold;
  }
  if (family.sharpening != 0.0) {
    double laplacian[kMaxGrid * kMaxGrid];
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        double sum = 4.0 * values[y * size + x];
        if (x > 0) sum -= values[y * size + x - 1];
        if (x + 1 < size) sum -= values[y * size + x + 1];
        if (y > 0) sum -= values[(y - 1) * size + x];
        if (y + 1 < size) sum -= values[(y + 1) * size + x];
        laplacian[y * size + x] = sum;
      }
    }
    for (int i = 0; i < size * size; ++i) values[i] += family.sharpening * laplacian[i];
  }

  std::uint64_t code = 0;
  double white_score = 0.0, black_score = 0.0;
  double white_count = 1.0, black_count = 1.0;
  for (int i = 0; i < family.nbits; ++i) {
    const double v = values[(family.bit_y[i] - min_coord) * size + family.bit_x[i] - min_coord];
    code <<= 1U;
    if (v > 0) {
      white_score += v;
      white_count += 1.0;
      code |= 1U;
    } else {
      black_score -= v;
      black_count += 1.0;
    }
  }
  const double margin = fmin(white_score / white_count, black_score / black_count);
  if (margin < 0.0) {
    return false;
  }

  for (int turns = 0; turns < 4; ++turns) {
    if (const TableEntry* entry = find_code(family, code)) {
      // The upright tag is the sampled one turned `turns` quarter turns, so its
      // tag coordinates map through the inverse turn first:
      // h * [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] per turn, (u, v) -> (v, -u).
      for (int t = 0; t < turns; ++t) {
        for (int row = 0; row < 3; ++row) {
          const double first = h[row * 3 + 0];
          h[row * 3 + 0] = -h[row * 3 + 1];
          h[row * 3 + 1] = first;
        }
      }
      out.valid = true;
      out.id = entry->id;
      out.hamming = entry->hamming;
      out.margin = static_cast<float>(margin);
      out.cluster = quad.cluster;
      for (int i = 0; i < 4; ++i) {
        project(h, ideal_x(i), ideal_y(i), out.corners[i]);
      }
      project(h, 0.0, 0.0, out.center);
      out.shortest_side = HUGE_VAL;
      for (int i = 0; i < 4; ++i) {
        const double* a = out.corners[i];
        const double* b = out.corners[(i + 1) % 4];
        out.shortest_side = fmin(out.shortest_side, hypot(b[0] - a[0], b[1] - a[1]));
      }
      return true;
    }
    code = rotate(family, code);
  }
  return false;
}

// One block per quad: refine its edges, then decode it on one thread.
__global__ void decode_kernel(const Quad* quads, const std::uint32_t* quad_count, DeviceGray image,
                              FamilyParams family, DecodedTag* decoded) {
  const std::uint32_t index = blockIdx.x;
  if (index >= min(*quad_count, kMaxQuads)) {
    return;
  }
  Quad quad = quads[index];
  if (family.refine_edges) {
    refine_edges(quad, image, family.decimate);
  }
  if (threadIdx.x == 0) {
    DecodedTag& out = decoded[index];
    out.valid = false;
    decode(quad, image, family, out);
  }
}

// Ranks a before b: fewer corrected bits, then a higher margin, then the
// earlier cluster.
__device__ bool ranks_before(const DecodedTag& a, const DecodedTag& b) {
  if (a.hamming != b.hamming) return a.hamming < b.hamming;
  if (a.margin != b.margin) return a.margin > b.margin;
  return a.cluster < b.cluster;
}

// One block. Several quads can frame one tag (outer and inner border edges,
// both polarities); keeps the best decode of each, in rank order. Each thread
// ranks its decodes against all the others, in the order a stable sort would
// give; one thread then drops the duplicates, comparing only decodes of one id
// in full.
__global__ void select_kernel(const DecodedTag* decoded, const std::uint32_t* quad_count,
                              const std::uint32_t* job_count, const SegmentCounts* counts,
                              FrameResult* result) {
  const std::uint32_t quads = min(*quad_count, kMaxQuads);
  __shared__ std::uint16_t order[kMaxQuads];
  __shared__ std::uint16_t kept[kMaxQuads];
  __shared__ std::uint32_t found;
  if (threadIdx.x == 0) {
    found = 0;
  }
  __syncthreads();
  for (std::uint32_t q = threadIdx.x; q < quads; q += blockDim.x) {
    if (!decoded[q].valid) continue;
    std::uint32_t rank = 0;
    for (std::uint32_t r = 0; r < quads; ++r) {
      if (r == q || !decoded[r].valid) continue;
      // Ties keep index order, as a stable sort would.
      if (ranks_before(decoded[r], decoded[q]) ||
          (r < q && !ranks_before(decoded[q], decoded[r]))) {
        ++rank;
      }
    }
    order[rank] = static_cast<std::uint16_t>(q);
    atomicAdd(&found, 1U);
  }
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }
  std::uint32_t count = 0;
  for (std::uint32_t k = 0; k < found; ++k) {
    const DecodedTag& candidate = decoded[order[k]];
    bool duplicate = false;
    for (std::uint32_t j = 0; j < count && !duplicate; ++j) {
      const DecodedTag& other = decoded[kept[j]];
      duplicate =
          other.id == candidate.id &&
          hypot(other.center[0] - candidate.center[0], other.center[1] - candidate.center[1]) <
              0.5 * fmin(other.shortest_side, candidate.shortest_side);
    }
    if (!duplicate) {
      kept[count++] = order[k];
    }
  }
  for (std::uint32_t k = 0; k < count && k < kMaxResults; ++k) {
    result->tags[k] = decoded[kept[k]];
  }
  result->count = count;
  result->quads = *quad_count;
  result->candidates = *job_count;
  result->points = counts->points;
}

class CudaBackend final : public DetectorBackend {
 public:
  CudaBackend(const DetectorOptions& options, int width, int height)
      : DetectorBackend(width, height),
        segmenter_(width, height, options.decimate, options.min_white_black_diff),
        jobs_(kMaxJobs),
        quads_(kMaxQuads),
        quad_count_(2),
        decoded_(kMaxQuads),
        result_(1) {
    const TagFamily tags(options.family);
    const apriltag_family_t* f = tags.get();
    if (static_cast<int>(f->nbits) > kMaxBits || f->total_width > kMaxGrid) {
      throw std::invalid_argument("CUDA backend: family " + options.family + " is too large");
    }
    family_.nbits = static_cast<std::int32_t>(f->nbits);
    family_.width_at_border = f->width_at_border;
    family_.total_width = f->total_width;
    family_.reversed_border = f->reversed_border;
    family_.sharpening = options.decode_sharpening;
    family_.refine_edges = options.refine_edges;
    family_.decimate = options.decimate;
    // The library stores bit coordinates, which may be negative, as unsigned.
    for (std::uint32_t i = 0; i < f->nbits; ++i) {
      family_.bit_x[i] = static_cast<std::int8_t>(static_cast<std::int32_t>(f->bit_x[i]));
      family_.bit_y[i] = static_cast<std::int8_t>(static_cast<std::int32_t>(f->bit_y[i]));
    }
    // A quarter turn about the tag center, (u, v) -> (-v, u) in bit cells.
    const double center = (f->width_at_border - 1) / 2.0;
    for (int i = 0; i < family_.nbits; ++i) {
      const double rx = center - (family_.bit_y[i] - center);
      const double ry = center + (family_.bit_x[i] - center);
      family_.rotation[i] = -1;
      for (int j = 0; j < family_.nbits; ++j) {
        if (std::abs(family_.bit_x[j] - rx) < 1e-6 && std::abs(family_.bit_y[j] - ry) < 1e-6) {
          family_.rotation[i] = static_cast<std::int8_t>(j);
        }
      }
      if (family_.rotation[i] < 0) {
        throw std::runtime_error("tag family layout is not symmetric under rotation");
      }
    }
    upload_decode_table(*f, decode_table_entries(options.family, options.max_hamming),
                        options.max_hamming);

    fit_.normal_border = !f->reversed_border;
    fit_.reversed_border = f->reversed_border;
    fit_.min_tag_width = std::max(3.0, static_cast<double>(f->width_at_border) / options.decimate);
    fit_.decimate = options.decimate;
    fit_.max_cluster_points = segmenter_.max_cluster_points();

    void* pinned = nullptr;
    cuda_check(cudaMallocHost(&pinned, sizeof(FrameResult)), "cudaMallocHost");
    host_result_.reset(static_cast<FrameResult*>(pinned));
  }

  [[nodiscard]] std::string_view name() const noexcept override { return "cuda"; }
  [[nodiscard]] MemoryLocation input_location() const noexcept override {
    return MemoryLocation::kDevice;
  }

  [[nodiscard]] FrameDetections find(const GrayImageView& image, cudaStream_t stream) override {
    segmenter_.segment(image, stream);
    cuda_check(cudaMemsetAsync(quad_count_.get(), 0, 2 * sizeof(std::uint32_t), stream),
               "cudaMemsetAsync");
    const ClusterData clusters = segmenter_.data();
    candidates_kernel<<<(segmenter_.capacity() + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        clusters, fit_, jobs_.get(), quad_count_.get() + 1);
    fit_quads_kernel<<<kMaxJobs, 32, 0, stream>>>(jobs_.get(), quad_count_.get() + 1, fit_,
                                                  quads_.get(), quad_count_.get());
    const DeviceGray gray{image.data, width_, height_, image.pitch};
    decode_kernel<<<kMaxQuads, kBlock, 0, stream>>>(quads_.get(), quad_count_.get(), gray, family_,
                                                    decoded_.get());
    select_kernel<<<1, kSelectThreads, 0, stream>>>(
        decoded_.get(), quad_count_.get(), quad_count_.get() + 1, clusters.counts, result_.get());
    cuda_check(cudaGetLastError(), "CUDA backend kernels");
    // The one copy to the host, and the one wait.
    cuda_check(cudaMemcpyAsync(host_result_.get(), result_.get(), sizeof(FrameResult),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(detections)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    const FrameResult& result = *host_result_;
    const auto excess = [](std::uint32_t found, std::uint32_t capacity) {
      return found > capacity ? found - capacity : 0U;
    };
    FrameDetections detections;
    detections.overflow = Overflow{.points = excess(result.points, segmenter_.capacity()),
                                   .candidates = excess(result.candidates, kMaxJobs),
                                   .quads = excess(result.quads, kMaxQuads),
                                   .detections = excess(result.count, kMaxResults)};
    for (std::uint32_t i = 0; i < std::min(result.count, kMaxResults); ++i) {
      const DecodedTag& tag = result.tags[i];
      detections.tags.push_back(
          make_detection(tag.id, tag.hamming, tag.margin, tag.center, tag.corners));
    }
    return detections;
  }

 private:
  struct HostFree {
    void operator()(FrameResult* result) const noexcept { cudaFreeHost(result); }
  };

  // The family's `entries` codes and variants within `max_hamming` bits, in an
  // open-addressed hash table on the device.
  void upload_decode_table(const apriltag_family_t& f, std::uint64_t entries, int max_hamming) {
    std::vector<TableEntry> table(std::bit_ceil(entries * 2), TableEntry{0, -1, 0});
    const std::uint64_t mask = table.size() - 1;
    const auto insert = [&](std::uint64_t code, int id, int hamming) {
      for (std::uint64_t slot = mix(code) & mask;; slot = (slot + 1) & mask) {
        TableEntry& entry = table[slot];
        if (entry.id < 0) {
          entry = {code, id, hamming};
          return;
        }
        if (entry.code == code) {
          // Two codes within reach of one word: keep the closer one.
          if (hamming < entry.hamming) entry = {code, id, hamming};
          return;
        }
      }
    };
    for (std::uint32_t id = 0; id < f.ncodes; ++id) {
      const std::uint64_t code = f.codes[id];
      insert(code, static_cast<int>(id), 0);
      if (max_hamming >= 1) {
        for (std::uint32_t a = 0; a < f.nbits; ++a) {
          const std::uint64_t one = code ^ (std::uint64_t{1} << a);
          insert(one, static_cast<int>(id), 1);
          if (max_hamming >= 2) {
            for (std::uint32_t b = 0; b < a; ++b) {
              insert(one ^ (std::uint64_t{1} << b), static_cast<int>(id), 2);
            }
          }
        }
      }
    }
    table_ = std::make_unique<DeviceBuffer<TableEntry>>(table.size());
    cuda_check(cudaMemcpy(table_->get(), table.data(), table.size() * sizeof(TableEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(decode table)");
    family_.table = table_->get();
    family_.table_mask = mask;
  }

  GpuSegmenter segmenter_;
  std::unique_ptr<DeviceBuffer<TableEntry>> table_;
  DeviceBuffer<FitJob> jobs_;
  DeviceBuffer<Quad> quads_;
  DeviceBuffer<std::uint32_t> quad_count_;  // Quads, then fit jobs.
  DeviceBuffer<DecodedTag> decoded_;
  DeviceBuffer<FrameResult> result_;
  std::unique_ptr<FrameResult, HostFree> host_result_;
  FamilyParams family_{};
  FitParams fit_{};
};

}  // namespace

std::unique_ptr<DetectorBackend> create_cuda_backend(const DetectorOptions& options, int width,
                                                     int height) {
  return std::make_unique<CudaBackend>(options, width, height);
}

}  // namespace holoscan::apriltag
