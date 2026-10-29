// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "cpu_reference.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "normal_equations.hpp"

namespace holoscan::icp::test {
namespace {

using Vec3 = std::array<double, 3>;

std::vector<Vec3> finite_points(const std::vector<float>& xyz) {
  std::vector<Vec3> out;
  for (std::size_t i = 0; i + 2 < xyz.size(); i += 3) {
    if (std::isfinite(xyz[i]) && std::isfinite(xyz[i + 1]) && std::isfinite(xyz[i + 2])) {
      out.push_back({xyz[i], xyz[i + 1], xyz[i + 2]});
    }
  }
  return out;
}

double distance2(const Vec3& a, const Vec3& b) {
  const double x = a[0] - b[0];
  const double y = a[1] - b[1];
  const double z = a[2] - b[2];
  return x * x + y * y + z * z;
}

// Eigen decomposition of a symmetric 3x3 matrix by cyclic Jacobi rotations:
// eigenvalues in `values`, eigenvectors as the columns of `vectors`.
void jacobi(std::array<std::array<double, 3>, 3> a, Vec3& values,
            std::array<std::array<double, 3>, 3>& vectors) {
  vectors = {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
  for (int sweep = 0; sweep < 64; ++sweep) {
    const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    if (off < 1e-300) break;
    for (int p = 0; p < 2; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        if (std::fabs(a[p][q]) < 1e-300) continue;
        const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        const double t =
            (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;
        for (int k = 0; k < 3; ++k) {  // A <- A J
          const double akp = a[k][p];
          const double akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {  // A <- J^T A
          const double apk = a[p][k];
          const double aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          const double vkp = vectors[k][p];
          const double vkq = vectors[k][q];
          vectors[k][p] = c * vkp - s * vkq;
          vectors[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  values = {a[0][0], a[1][1], a[2][2]};
}

// Each point's normal from its neighbors within `radius`, facing the origin;
// none where fewer than 3 neighbors or the smallest eigenvalue is not unique.
std::vector<std::optional<Vec3>> normals_of(const std::vector<Vec3>& points, double radius) {
  std::vector<std::optional<Vec3>> normals(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    double n = 0.0;
    Vec3 sum{};
    std::array<std::array<double, 3>, 3> second{};
    for (const Vec3& q : points) {
      if (distance2(q, points[i]) > radius * radius) continue;
      const Vec3 d{q[0] - points[i][0], q[1] - points[i][1], q[2] - points[i][2]};
      n += 1.0;
      for (int a = 0; a < 3; ++a) {
        sum[a] += d[a];
        for (int b = 0; b < 3; ++b) second[a][b] += d[a] * d[b];
      }
    }
    if (n < 3.0) continue;
    std::array<std::array<double, 3>, 3> covariance{};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        covariance[a][b] = second[a][b] / n - (sum[a] / n) * (sum[b] / n);
      }
    }
    Vec3 values{};
    std::array<std::array<double, 3>, 3> vectors{};
    jacobi(covariance, values, vectors);
    int smallest = 0;
    int largest = 0;
    for (int k = 1; k < 3; ++k) {
      if (values[k] < values[smallest]) smallest = k;
      if (values[k] > values[largest]) largest = k;
    }
    const int middle = 3 - smallest - largest;
    if (!(values[middle] - values[smallest] > 1e-6 * std::fabs(values[largest]))) continue;
    Vec3 normal{vectors[0][smallest], vectors[1][smallest], vectors[2][smallest]};
    if (normal[0] * points[i][0] + normal[1] * points[i][1] + normal[2] * points[i][2] > 0.0) {
      normal = {-normal[0], -normal[1], -normal[2]};
    }
    normals[i] = normal;
  }
  return normals;
}

double weight(const IcpOptions& options, double residual) {
  const double r = std::fabs(residual);
  const double k = options.robust_scale;
  switch (options.robust_kernel) {
    case RobustKernel::kHuber:
      return r <= k ? 1.0 : k / r;
    case RobustKernel::kTukey:
      return r >= k ? 0.0 : (1.0 - (r / k) * (r / k)) * (1.0 - (r / k) * (r / k));
    case RobustKernel::kNone:
      break;
  }
  return 1.0;
}

Vec3 apply(const Matrix4& m, const Vec3& p) {
  return {m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
          m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
          m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]};
}

Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

void add_row(std::array<double, detail::kSums>& sums, const Vec3& a, const Vec3& b, double r,
             double w) {
  const double j[6] = {a[0], a[1], a[2], b[0], b[1], b[2]};
  for (int row = 0; row < 6; ++row) {
    for (int col = row; col < 6; ++col) {
      sums[detail::kJtJ + detail::upper_index(row, col)] += w * j[row] * j[col];
    }
    sums[detail::kJtr + row] += w * j[row] * r;
  }
}

std::array<double, detail::kSums> associate(const std::vector<Vec3>& source,
                                            const std::vector<Vec3>& target,
                                            const std::vector<std::optional<Vec3>>& normals,
                                            const IcpOptions& options, const Matrix4& transform) {
  std::array<double, detail::kSums> sums{};
  const double max2 = static_cast<double>(options.max_correspondence_distance) *
                      options.max_correspondence_distance;
  for (const Vec3& s : source) {
    const Vec3 p = apply(transform, s);
    double best = max2;
    std::size_t match = target.size();
    for (std::size_t j = 0; j < target.size(); ++j) {
      const double d2 = distance2(p, target[j]);
      if (d2 < best || (d2 == best && j < match)) {
        best = d2;
        match = j;
      }
    }
    if (match == target.size()) continue;
    const Vec3& q = target[match];
    const Vec3 diff{p[0] - q[0], p[1] - q[1], p[2] - q[2]};
    if (options.method == IcpMethod::kPointToPlane) {
      if (!normals[match]) continue;
      const Vec3& n = *normals[match];
      const double r = n[0] * diff[0] + n[1] * diff[1] + n[2] * diff[2];
      add_row(sums, cross(p, n), n, r, weight(options, r));
    } else {
      const double w = weight(options, std::sqrt(best));
      for (int axis = 0; axis < 3; ++axis) {
        Vec3 e{};
        e[axis] = 1.0;
        add_row(sums, cross(p, e), e, diff[axis], w);
      }
    }
    sums[detail::kSquaredDistance] += best;
    sums[detail::kCount] += 1.0;
  }
  return sums;
}

}  // namespace

ReferenceResult reference_icp(const std::vector<float>& source_xyz,
                              const std::vector<float>& target_xyz, const IcpOptions& options,
                              const Matrix4& initial, int iterations) {
  const std::vector<Vec3> source = finite_points(source_xyz);
  const std::vector<Vec3> target = finite_points(target_xyz);
  const double radius =
      options.normal_radius > 0.0F ? options.normal_radius : options.max_correspondence_distance;
  const std::vector<std::optional<Vec3>> normals =
      options.method == IcpMethod::kPointToPlane ? normals_of(target, radius)
                                                 : std::vector<std::optional<Vec3>>(target.size());
  Matrix4 transform = initial;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const auto sums = associate(source, target, normals, options, transform);
    const auto step = detail::solve_normal_equations(sums);
    if (!step) break;
    transform = compose(
        transform_from({(*step)[0], (*step)[1], (*step)[2]}, {(*step)[3], (*step)[4], (*step)[5]}),
        transform);
  }
  const auto sums = associate(source, target, normals, options, transform);
  ReferenceResult result;
  result.transform = transform;
  result.correspondences = static_cast<std::uint32_t>(sums[detail::kCount]);
  result.fitness = source.empty() ? 0.0 : sums[detail::kCount] / static_cast<double>(source.size());
  result.rmse = sums[detail::kCount] > 0.0
                    ? std::sqrt(sums[detail::kSquaredDistance] / sums[detail::kCount])
                    : 0.0;
  return result;
}

}  // namespace holoscan::icp::test
