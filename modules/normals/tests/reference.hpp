// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CPU references, written from the definitions in normals.hpp and planes.hpp,
// for small clouds.
//
// The normals: the same neighbors, chosen by the same float expressions
// (window ranges, squared distances), and the covariance and eigenproblem in
// double, against which the GPU's float results are compared with a
// tolerance.
//
// The planes: the same algorithm step by step on one thread, with the same
// float and double expressions and the double sums added in the GPU's order
// (a fixed grid of blocks, each reduced as a tree), so that with nothing
// contracted it computes the very same planes and labels.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

#include <holoscan/normals/normals.hpp>
#include <holoscan/normals/planes.hpp>

namespace holoscan::normals::test {

inline bool finite3(const float* p) {
  return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// The eigenvalues and eigenvectors of symmetric `a`, by cyclic Jacobi
// rotations, as the kernels compute them.
template <typename T>
void jacobi3(T a[3][3], T values[3], T vectors[3][3]) {
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      vectors[row][column] = row == column ? T(1) : T(0);
    }
  }
  for (int sweep = 0; sweep < 16; ++sweep) {
    const T off = std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
    const T diagonal = std::fabs(a[0][0]) + std::fabs(a[1][1]) + std::fabs(a[2][2]);
    if (!(off > T(0)) || off <= diagonal * T(1e-12)) {
      break;
    }
    for (int p = 0; p < 2; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        if (a[p][q] == T(0)) {
          continue;
        }
        const T theta = (a[q][q] - a[p][p]) / (T(2) * a[p][q]);
        const T t =
            (theta >= T(0) ? T(1) : T(-1)) / (std::fabs(theta) + std::sqrt(theta * theta + T(1)));
        const T c = T(1) / std::sqrt(t * t + T(1));
        const T s = t * c;
        for (int k = 0; k < 3; ++k) {
          const T akp = a[k][p];
          const T akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
          const T apk = a[p][k];
          const T aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          const T vkp = vectors[k][p];
          const T vkq = vectors[k][q];
          vectors[k][p] = c * vkp - s * vkq;
          vectors[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  for (int k = 0; k < 3; ++k) {
    values[k] = a[k][k];
  }
}

// The normal and surface variation of a covariance, as pca_normal() in the
// kernels; false without a normal.
template <typename T>
bool pca(T covariance[3][3], T normal[3], T& curvature) {
  T values[3];
  T vectors[3][3];
  jacobi3(covariance, values, vectors);
  int order[3] = {0, 1, 2};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2 - i; ++j) {
      if (values[order[j + 1]] < values[order[j]]) {
        std::swap(order[j], order[j + 1]);
      }
    }
  }
  const T smallest = values[order[0]] > T(0) ? values[order[0]] : T(0);
  const T middle = values[order[1]];
  const T largest = values[order[2]];
  const T sum = smallest + middle + largest;
  if (!(largest > T(0)) || !(middle > largest * T(1e-5)) || !(sum > T(0))) {
    return false;
  }
  T length = T(0);
  for (int k = 0; k < 3; ++k) {
    normal[k] = vectors[k][order[0]];
    length += normal[k] * normal[k];
  }
  length = std::sqrt(length);
  if (!(length > T(0))) {
    return false;
  }
  for (int k = 0; k < 3; ++k) {
    normal[k] /= length;
  }
  curvature = smallest / sum;
  return true;
}

struct NormalResult {
  std::vector<float> normals;    // 3 per point, NaN without a normal.
  std::vector<float> curvature;  // 1 per point, NaN without a normal.
};

namespace detail {

// The normal of point p from the float offsets of its neighbors, in double.
inline void finish(const std::vector<float>& offsets, std::uint32_t min_neighbors, const float* p,
                   const float view[3], std::size_t i, NormalResult& out) {
  const std::size_t n = offsets.size() / 3;
  if (n < min_neighbors) {
    return;
  }
  double s[3] = {};
  double ss[6] = {};
  for (std::size_t j = 0; j < n; ++j) {
    const double x = offsets[3 * j];
    const double y = offsets[3 * j + 1];
    const double z = offsets[3 * j + 2];
    s[0] += x;
    s[1] += y;
    s[2] += z;
    ss[0] += x * x;
    ss[1] += x * y;
    ss[2] += x * z;
    ss[3] += y * y;
    ss[4] += y * z;
    ss[5] += z * z;
  }
  const double count = static_cast<double>(n);
  const double mean[3] = {s[0] / count, s[1] / count, s[2] / count};
  double cov[3][3];
  cov[0][0] = ss[0] / count - mean[0] * mean[0];
  cov[0][1] = ss[1] / count - mean[0] * mean[1];
  cov[0][2] = ss[2] / count - mean[0] * mean[2];
  cov[1][1] = ss[3] / count - mean[1] * mean[1];
  cov[1][2] = ss[4] / count - mean[1] * mean[2];
  cov[2][2] = ss[5] / count - mean[2] * mean[2];
  cov[1][0] = cov[0][1];
  cov[2][0] = cov[0][2];
  cov[2][1] = cov[1][2];
  double normal[3];
  double curvature = 0.0;
  if (!pca(cov, normal, curvature)) {
    return;
  }
  const double toward =
      normal[0] * (view[0] - p[0]) + normal[1] * (view[1] - p[1]) + normal[2] * (view[2] - p[2]);
  const double sign = toward < 0.0 ? -1.0 : 1.0;
  for (int k = 0; k < 3; ++k) {
    out.normals[3 * i + k] = static_cast<float>(sign * normal[k]);
  }
  out.curvature[i] = static_cast<float>(curvature);
}

inline NormalResult empty_result(std::size_t n) {
  NormalResult out;
  out.normals.assign(3 * n, std::numeric_limits<float>::quiet_NaN());
  out.curvature.assign(n, std::numeric_limits<float>::quiet_NaN());
  return out;
}

}  // namespace detail

// The window search on an organized cloud.
inline NormalResult window_reference(const std::vector<float>& xyz, int width, int height,
                                     const NormalParams& params) {
  NormalResult out = detail::empty_result(static_cast<std::size_t>(width) * height);
  const float* view = params.viewpoint;
  const auto range_of = [view](const float* q) {
    const float dx = q[0] - view[0];
    const float dy = q[1] - view[1];
    const float dz = q[2] - view[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  };
  std::vector<float> offsets;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * width + u;
      const float* p = xyz.data() + 3 * i;
      if (!finite3(p)) {
        continue;
      }
      const float range = range_of(p);
      const float step_limit = params.depth_change_factor * range;
      offsets.clear();
      for (int dv = -params.window_radius; dv <= params.window_radius; ++dv) {
        for (int du = -params.window_radius; du <= params.window_radius; ++du) {
          const int row = v + dv;
          const int column = u + du;
          if (row < 0 || row >= height || column < 0 || column >= width) {
            continue;
          }
          const float* q = xyz.data() + 3 * (static_cast<std::size_t>(row) * width + column);
          if (!finite3(q)) {
            continue;
          }
          const int steps = std::max(std::abs(du), std::abs(dv));
          if (std::fabs(range_of(q) - range) > step_limit * static_cast<float>(steps)) {
            continue;
          }
          offsets.insert(offsets.end(), {q[0] - p[0], q[1] - p[1], q[2] - p[2]});
        }
      }
      detail::finish(offsets, params.min_neighbors, p, view, i, out);
    }
  }
  return out;
}

// The radius (knn false) or k-nearest (knn true) search, by brute force.
inline NormalResult grid_reference(const std::vector<float>& xyz, const NormalParams& params,
                                   bool knn) {
  const std::size_t n = xyz.size() / 3;
  NormalResult out = detail::empty_result(n);
  const float squared_radius = params.radius * params.radius;
  std::vector<std::pair<float, std::uint32_t>> near;
  std::vector<float> offsets;
  for (std::size_t i = 0; i < n; ++i) {
    const float* p = xyz.data() + 3 * i;
    if (!finite3(p)) {
      continue;
    }
    near.clear();
    for (std::size_t j = 0; j < n; ++j) {
      const float* q = xyz.data() + 3 * j;
      if (!finite3(q)) {
        continue;
      }
      const float ex = q[0] - p[0];
      const float ey = q[1] - p[1];
      const float ez = q[2] - p[2];
      const float squared = ex * ex + ey * ey + ez * ez;
      if (squared <= squared_radius) {
        near.emplace_back(squared, static_cast<std::uint32_t>(j));
      }
    }
    if (knn) {
      std::sort(near.begin(), near.end());
      if (near.size() > params.k) {
        near.resize(params.k);
      }
    }
    offsets.clear();
    for (const auto& [squared, j] : near) {
      const float* q = xyz.data() + 3 * j;
      offsets.insert(offsets.end(), {q[0] - p[0], q[1] - p[1], q[2] - p[2]});
    }
    detail::finish(offsets, params.min_neighbors, p, params.viewpoint, i, out);
  }
  return out;
}

struct PlaneResult {
  std::vector<std::uint8_t> labels;
  std::vector<float> coefficients;  // [max_planes, 4]
  std::vector<float> centroids;     // [max_planes, 3]
  std::vector<std::uint32_t> inliers;
  std::uint32_t count{};
};

namespace detail {

inline std::uint64_t mix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

struct Plane {
  float x;
  float y;
  float z;
  float w;
};

inline float dot3(const float* a, const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline bool on_plane(const Plane& plane, const float* p, const float* n, float threshold,
                     float cosine) {
  const float normal[3] = {plane.x, plane.y, plane.z};
  return std::fabs(dot3(normal, p) + plane.w) <= threshold && std::fabs(dot3(normal, n)) >= cosine;
}

// The kernels' sums: 64 blocks of 256 threads over the points, each block
// reduced as a tree, then the blocks in order.
template <typename Include>
void ordered_sums(const std::vector<float>& xyz, std::size_t count, Include&& include,
                  double total[10]) {
  constexpr std::size_t kBlocks = 64;
  constexpr std::size_t kThreads = 256;
  std::vector<double> partials(kBlocks * 10, 0.0);
  std::vector<double> shared(kThreads * 10);
  for (std::size_t b = 0; b < kBlocks; ++b) {
    for (std::size_t t = 0; t < kThreads; ++t) {
      double s[10] = {};
      for (std::size_t i = b * kThreads + t; i < count; i += kBlocks * kThreads) {
        if (!include(i)) {
          continue;
        }
        const double x = xyz[3 * i];
        const double y = xyz[3 * i + 1];
        const double z = xyz[3 * i + 2];
        s[0] += 1.0;
        s[1] += x;
        s[2] += y;
        s[3] += z;
        s[4] += x * x;
        s[5] += x * y;
        s[6] += x * z;
        s[7] += y * y;
        s[8] += y * z;
        s[9] += z * z;
      }
      for (int k = 0; k < 10; ++k) {
        shared[k * kThreads + t] = s[k];
      }
    }
    for (std::size_t stride = kThreads / 2; stride > 0; stride /= 2) {
      for (std::size_t t = 0; t < stride; ++t) {
        for (int k = 0; k < 10; ++k) {
          shared[k * kThreads + t] += shared[k * kThreads + t + stride];
        }
      }
    }
    for (int k = 0; k < 10; ++k) {
      partials[b * 10 + k] = shared[k * kThreads];
    }
  }
  for (int k = 0; k < 10; ++k) {
    total[k] = 0.0;
  }
  for (std::size_t b = 0; b < kBlocks; ++b) {
    for (int k = 0; k < 10; ++k) {
      total[k] += partials[b * 10 + k];
    }
  }
}

inline bool fit(const double s[10], const float view[3], Plane& plane) {
  const double n = s[0];
  if (n < 3.0) {
    return false;
  }
  const double mean[3] = {s[1] / n, s[2] / n, s[3] / n};
  double cov[3][3];
  cov[0][0] = s[4] / n - mean[0] * mean[0];
  cov[0][1] = s[5] / n - mean[0] * mean[1];
  cov[0][2] = s[6] / n - mean[0] * mean[2];
  cov[1][1] = s[7] / n - mean[1] * mean[1];
  cov[1][2] = s[8] / n - mean[1] * mean[2];
  cov[2][2] = s[9] / n - mean[2] * mean[2];
  cov[1][0] = cov[0][1];
  cov[2][0] = cov[0][2];
  cov[2][1] = cov[1][2];
  double normal[3];
  double variation = 0.0;
  if (!pca(cov, normal, variation)) {
    return false;
  }
  double d = -(normal[0] * mean[0] + normal[1] * mean[1] + normal[2] * mean[2]);
  if (normal[0] * view[0] + normal[1] * view[1] + normal[2] * view[2] + d < 0.0) {
    for (double& value : normal) {
      value = -value;
    }
    d = -d;
  }
  plane = Plane{static_cast<float>(normal[0]), static_cast<float>(normal[1]),
                static_cast<float>(normal[2]), static_cast<float>(d)};
  return true;
}

inline void centroid(const double* s, float c[3]) {
  for (int k = 0; k < 3; ++k) {
    c[k] = static_cast<float>(s[1 + k] / s[0]);
  }
}

}  // namespace detail

inline float cosine_of(float degrees) {
  return static_cast<float>(std::cos(static_cast<double>(degrees) * std::numbers::pi / 180.0));
}

// The planes, step by step as the kernels take them.
inline PlaneResult plane_reference(const std::vector<float>& xyz, const std::vector<float>& normals,
                                   const PlaneParams& params) {
  using detail::Plane;
  const std::size_t count = xyz.size() / 3;
  constexpr std::uint32_t kTries = 16;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float threshold = params.distance_threshold;
  const float cosine = cosine_of(params.max_normal_angle_deg);
  const float* view = params.viewpoint;
  PlaneResult out;
  out.labels.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    out.labels[i] = finite3(xyz.data() + 3 * i) && finite3(normals.data() + 3 * i) ? kLabelNoPlane
                                                                                   : kLabelInvalid;
  }
  std::vector<Plane> found;
  std::vector<std::array<double, 10>> found_sums;
  const auto point = [&](std::uint32_t i) { return xyz.data() + 3 * i; };
  const auto normal = [&](std::uint32_t i) { return normals.data() + 3 * i; };
  bool active = count != 0;
  for (std::uint32_t round = 0; round < params.max_planes && active; ++round) {
    std::vector<std::uint32_t> candidates;
    for (std::size_t i = 0; i < count; ++i) {
      if (out.labels[i] == kLabelNoPlane) {
        candidates.push_back(static_cast<std::uint32_t>(i));
      }
    }
    const auto n = static_cast<std::uint32_t>(candidates.size());
    // Hypotheses.
    std::vector<Plane> hypotheses(params.iterations, Plane{nan, nan, nan, nan});
    for (std::uint32_t h = 0; h < params.iterations && n >= 3U; ++h) {
      const std::uint64_t key = detail::mix64(
          params.seed ^ detail::mix64((static_cast<std::uint64_t>(round) << 32U) | h));
      const auto draw = [&](std::uint64_t j) {
        return candidates[detail::mix64(key ^ detail::mix64(j)) % n];
      };
      std::uint32_t picks[3];
      picks[0] = draw(0U);
      bool ok = true;
      for (std::uint32_t j = 1; j < 3U && ok; ++j) {
        bool got = false;
        for (std::uint32_t attempt = 0; attempt < kTries && !got; ++attempt) {
          const std::uint32_t pick = draw(1U + (j - 1U) * kTries + attempt);
          if (pick != picks[0] && (j == 1U || pick != picks[1]) &&
              std::fabs(detail::dot3(normal(pick), normal(picks[0]))) >= cosine) {
            picks[j] = pick;
            got = true;
          }
        }
        ok = got;
      }
      if (!ok) {
        continue;
      }
      const float* p0 = point(picks[0]);
      const float* p1 = point(picks[1]);
      const float* p2 = point(picks[2]);
      const float a[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
      const float b[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
      float c[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0]};
      const float length = std::sqrt(detail::dot3(c, c));
      const float spread = std::sqrt(detail::dot3(a, a)) * std::sqrt(detail::dot3(b, b));
      if (!(length > 1e-6F * spread) || !(length > 0.0F)) {
        continue;
      }
      for (float& value : c) {
        value /= length;
      }
      float d = -detail::dot3(c, p0);
      if (detail::dot3(c, view) + d < 0.0F) {
        for (float& value : c) {
          value = -value;
        }
        d = -d;
      }
      bool agree = true;
      for (const std::uint32_t pick : picks) {
        agree = agree && std::fabs(detail::dot3(c, normal(pick))) >= cosine;
      }
      if (agree) {
        hypotheses[h] = Plane{c[0], c[1], c[2], d};
      }
    }
    // Scores on the sample, and the best.
    const std::uint32_t taken = std::min(params.samples, n);
    std::uint32_t best_score = 0;
    std::uint32_t best = 0;
    for (std::uint32_t h = 0; h < params.iterations; ++h) {
      if (!std::isfinite(hypotheses[h].x)) {
        continue;
      }
      std::uint32_t score = 0;
      for (std::uint32_t j = 0; j < taken; ++j) {
        const std::uint32_t i = candidates[static_cast<std::uint64_t>(j) * n / taken];
        score += detail::on_plane(hypotheses[h], point(i), normal(i), threshold, cosine) ? 1U : 0U;
      }
      if (score > best_score) {
        best_score = score;
        best = h;
      }
    }
    if (best_score == 0U) {
      break;
    }
    Plane plane = hypotheses[best];
    const auto inlier = [&](std::size_t i) {
      return out.labels[i] == kLabelNoPlane &&
             detail::on_plane(plane, point(static_cast<std::uint32_t>(i)),
                              normal(static_cast<std::uint32_t>(i)), threshold, cosine);
    };
    double s[10];
    for (std::uint32_t refit = 0; refit < params.refine_iterations; ++refit) {
      detail::ordered_sums(xyz, count, inlier, s);
      Plane fitted{};
      if (detail::fit(s, view, fitted)) {
        plane = fitted;
      }
    }
    detail::ordered_sums(xyz, count, inlier, s);
    if (s[0] < static_cast<double>(params.min_inliers) || s[0] < 3.0) {
      break;
    }
    Plane reported = plane;
    if (params.refine_iterations != 0U) {
      detail::fit(s, view, reported);
    }
    found.push_back(reported);
    std::array<double, 10> kept{};
    std::copy_n(s, 10, kept.begin());
    found_sums.push_back(kept);
    const auto label = static_cast<std::uint8_t>(found.size());
    for (std::size_t i = 0; i < count; ++i) {
      if (inlier(i)) {
        out.labels[i] = label;
      }
    }
  }
  if (params.merge_coplanar && !found.empty()) {
    const float merge_cosine = cosine_of(params.merge_angle_deg);
    std::vector<std::uint32_t> root(found.size());
    for (std::uint32_t j = 0; j < found.size(); ++j) {
      root[j] = j;
      for (std::uint32_t i = 0; i < j; ++i) {
        if (root[i] != i) {
          continue;
        }
        float ca[3];
        float cb[3];
        detail::centroid(found_sums[i].data(), ca);
        detail::centroid(found_sums[j].data(), cb);
        const float na[3] = {found[i].x, found[i].y, found[i].z};
        const float nb[3] = {found[j].x, found[j].y, found[j].z};
        if (detail::dot3(na, nb) >= merge_cosine &&
            std::fabs(detail::dot3(na, cb) + found[i].w) <= params.merge_distance &&
            std::fabs(detail::dot3(nb, ca) + found[j].w) <= params.merge_distance) {
          for (int k = 0; k < 10; ++k) {
            found_sums[i][k] += found_sums[j][k];
          }
          Plane merged{};
          if (detail::fit(found_sums[i].data(), view, merged)) {
            found[i] = merged;
          }
          root[j] = i;
          break;
        }
      }
    }
    std::vector<std::uint32_t> map(found.size());
    std::uint32_t kept = 0;
    for (std::uint32_t j = 0; j < found.size(); ++j) {
      if (root[j] == j) {
        map[j] = kept;
        found[kept] = found[j];
        found_sums[kept] = found_sums[j];
        ++kept;
      } else {
        map[j] = map[root[j]];
      }
    }
    found.resize(kept);
    found_sums.resize(kept);
    for (std::uint8_t& label : out.labels) {
      if (label != kLabelNoPlane && label != kLabelInvalid) {
        label = static_cast<std::uint8_t>(map[label - 1U] + 1U);
      }
    }
  }
  out.count = static_cast<std::uint32_t>(found.size());
  out.coefficients.assign(4U * params.max_planes, nan);
  out.centroids.assign(3U * params.max_planes, nan);
  out.inliers.assign(params.max_planes, 0U);
  for (std::size_t k = 0; k < found.size(); ++k) {
    out.coefficients[4 * k] = found[k].x;
    out.coefficients[4 * k + 1] = found[k].y;
    out.coefficients[4 * k + 2] = found[k].z;
    out.coefficients[4 * k + 3] = found[k].w;
    detail::centroid(found_sums[k].data(), out.centroids.data() + 3 * k);
    out.inliers[k] = static_cast<std::uint32_t>(found_sums[k][0]);
  }
  return out;
}

}  // namespace holoscan::normals::test
