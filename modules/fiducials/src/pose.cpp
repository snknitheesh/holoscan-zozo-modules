// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/fiducials/pose.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "detection_core.hpp"

namespace holoscan::fiducials {
namespace {

using Matrix3 = std::array<double, 9>;
using Vector3 = std::array<double, 3>;

Vector3 cross(const Vector3& a, const Vector3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double dot(const Vector3& a, const Vector3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double norm(const Vector3& a) { return std::sqrt(dot(a, a)); }

Matrix3 multiply(const Matrix3& a, const Matrix3& b) {
  Matrix3 c{};
  for (int r = 0; r < 3; ++r) {
    for (int k = 0; k < 3; ++k) {
      c[r * 3 + k] = a[r * 3] * b[k] + a[r * 3 + 1] * b[3 + k] + a[r * 3 + 2] * b[6 + k];
    }
  }
  return c;
}

// Rotation by the angle |w| about w (Rodrigues).
Matrix3 exp_so3(const Vector3& w) {
  const double angle = norm(w);
  if (angle < 1e-12) {
    return {1, -w[2], w[1], w[2], 1, -w[0], -w[1], w[0], 1};
  }
  const Vector3 k{w[0] / angle, w[1] / angle, w[2] / angle};
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const double v = 1.0 - c;
  return {c + k[0] * k[0] * v,        k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s,
          k[1] * k[0] * v + k[2] * s, c + k[1] * k[1] * v,        k[1] * k[2] * v - k[0] * s,
          k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v};
}

// The rotation nearest m: the polar factor, by Higham's iteration R <- (R + R^-T) / 2.
std::optional<Matrix3> nearest_rotation(Matrix3 m) {
  for (int iteration = 0; iteration < 30; ++iteration) {
    const Vector3 r0{m[0], m[1], m[2]};
    const Vector3 r1{m[3], m[4], m[5]};
    const Vector3 r2{m[6], m[7], m[8]};
    const double det = dot(r0, cross(r1, r2));
    if (!(std::abs(det) > 1e-12)) return std::nullopt;
    // Inverse transpose: the cofactor matrix over the determinant.
    const Vector3 c0 = cross(r1, r2);
    const Vector3 c1 = cross(r2, r0);
    const Vector3 c2 = cross(r0, r1);
    Matrix3 next{};
    for (int k = 0; k < 3; ++k) {
      next[k] = 0.5 * (m[k] + c0[k] / det);
      next[3 + k] = 0.5 * (m[3 + k] + c1[k] / det);
      next[6 + k] = 0.5 * (m[6 + k] + c2[k] / det);
    }
    double change = 0.0;
    for (int k = 0; k < 9; ++k) change = std::max(change, std::abs(next[k] - m[k]));
    m = next;
    if (change < 1e-14) break;
  }
  return m;
}

Point3 transform(const Pose& pose, const Point3& p) {
  const auto& r = pose.rotation;
  const auto& t = pose.translation;
  return {r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0],
          r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1],
          r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2]};
}

// Squared reprojection errors summed, and each residual into `residuals` when given.
double reprojection_cost(const CameraIntrinsics& k, const Pose& pose,
                         std::span<const Point2> target_xy, std::span<const Point2> image,
                         std::vector<double>* residuals = nullptr) {
  double cost = 0.0;
  for (std::size_t i = 0; i < image.size(); ++i) {
    const Point3 camera = transform(pose, {target_xy[i][0], target_xy[i][1], 0.0});
    double dx = 1e3;
    double dy = 1e3;
    if (camera[2] > 1e-9) {
      const Point2 seen = project(k, camera);
      dx = seen[0] - image[i][0];
      dy = seen[1] - image[i][1];
    }
    if (residuals != nullptr) {
      (*residuals)[2 * i] = dx;
      (*residuals)[2 * i + 1] = dy;
    }
    cost += dx * dx + dy * dy;
  }
  return cost;
}

// The pose moved by `delta`: rotation exp(delta[0..2]) R, translation t + delta[3..5].
Pose moved(const Pose& pose, const std::array<double, 6>& delta) {
  Pose result = pose;
  result.rotation = multiply(exp_so3({delta[0], delta[1], delta[2]}), pose.rotation);
  for (int i = 0; i < 3; ++i) result.translation[i] += delta[3 + i];
  return result;
}

// Levenberg-Marquardt on the reprojection error, from `pose`.
Pose refine(const CameraIntrinsics& k, Pose pose, std::span<const Point2> target_xy,
            std::span<const Point2> image) {
  const std::size_t rows = 2 * image.size();
  std::vector<double> residuals(rows);
  std::vector<double> plus(rows);
  std::vector<double> minus(rows);
  std::vector<double> jacobian(rows * 6);
  double cost = reprojection_cost(k, pose, target_xy, image, &residuals);
  double lambda = 1e-3;
  for (int iteration = 0; iteration < 50 && cost > 0.0; ++iteration) {
    const double scale = std::max(1.0, norm(pose.translation));
    for (int p = 0; p < 6; ++p) {
      const double step = p < 3 ? 1e-6 : 1e-6 * scale;
      std::array<double, 6> delta{};
      delta[p] = step;
      reprojection_cost(k, moved(pose, delta), target_xy, image, &plus);
      delta[p] = -step;
      reprojection_cost(k, moved(pose, delta), target_xy, image, &minus);
      for (std::size_t r = 0; r < rows; ++r) {
        jacobian[r * 6 + p] = (plus[r] - minus[r]) / (2.0 * step);
      }
    }
    double jtj[6][6] = {};
    double jtr[6] = {};
    for (std::size_t r = 0; r < rows; ++r) {
      for (int a = 0; a < 6; ++a) {
        jtr[a] += jacobian[r * 6 + a] * residuals[r];
        for (int b = 0; b < 6; ++b) jtj[a][b] += jacobian[r * 6 + a] * jacobian[r * 6 + b];
      }
    }
    bool improved = false;
    for (int attempt = 0; attempt < 10 && !improved; ++attempt) {
      double system[6][7];
      for (int a = 0; a < 6; ++a) {
        for (int b = 0; b < 6; ++b) system[a][b] = jtj[a][b];
        system[a][a] += lambda * (jtj[a][a] + 1e-12);
        system[a][6] = -jtr[a];
      }
      std::array<double, 6> delta{};
      if (!core::solve(system, 1e-30, delta.data())) {
        lambda *= 10.0;
        continue;
      }
      const Pose candidate = moved(pose, delta);
      std::vector<double> candidate_residuals(rows);
      const double candidate_cost =
          reprojection_cost(k, candidate, target_xy, image, &candidate_residuals);
      if (candidate_cost < cost) {
        const double gain = cost - candidate_cost;
        pose = candidate;
        residuals.swap(candidate_residuals);
        lambda = std::max(lambda * 0.1, 1e-12);
        improved = true;
        if (gain < 1e-12 * cost) iteration = 50;  // Converged.
        cost = candidate_cost;
      } else {
        lambda *= 10.0;
      }
    }
    if (!improved) break;
  }
  pose.reprojection_error = std::sqrt(cost / static_cast<double>(image.size()));
  return pose;
}

// A first pose from the homography between the target plane and the normalized image.
std::optional<Pose> initial_pose(std::span<const Point2> target_xy,
                                 std::span<const Point2> normalized) {
  // Points centered and scaled (Hartley), for a well-conditioned least-squares system.
  const auto similarity = [](std::span<const Point2> points) {
    double mx = 0.0, my = 0.0;
    for (const Point2& p : points) {
      mx += p[0];
      my += p[1];
    }
    mx /= static_cast<double>(points.size());
    my /= static_cast<double>(points.size());
    double spread = 0.0;
    for (const Point2& p : points) spread += std::hypot(p[0] - mx, p[1] - my);
    spread /= static_cast<double>(points.size());
    const double s = spread > 0.0 ? std::sqrt(2.0) / spread : 1.0;
    return std::array<double, 3>{s, mx, my};  // p' = s (p - m)
  };
  const auto a = similarity(target_xy);
  const auto b = similarity(normalized);
  double system[8][9] = {};
  for (std::size_t i = 0; i < target_xy.size(); ++i) {
    const double x = a[0] * (target_xy[i][0] - a[1]);
    const double y = a[0] * (target_xy[i][1] - a[2]);
    const double u = b[0] * (normalized[i][0] - b[1]);
    const double v = b[0] * (normalized[i][1] - b[2]);
    const double rows[2][9] = {{x, y, 1, 0, 0, 0, -x * u, -y * u, u},
                               {0, 0, 0, x, y, 1, -x * v, -y * v, v}};
    for (const auto& row : rows) {
      for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 9; ++c) system[r][c] += row[r] * row[c];
      }
    }
  }
  double hn[9];
  if (!core::solve(system, 1e-12, hn)) return std::nullopt;
  hn[8] = 1.0;
  // Undo the normalizations: H = Tb^-1 Hn Ta.
  const Matrix3 ta{a[0], 0, -a[0] * a[1], 0, a[0], -a[0] * a[2], 0, 0, 1};
  const Matrix3 tb_inv{1 / b[0], 0, b[1], 0, 1 / b[0], b[2], 0, 0, 1};
  const Matrix3 h =
      multiply(tb_inv, multiply({hn[0], hn[1], hn[2], hn[3], hn[4], hn[5], hn[6], hn[7], hn[8]}, ta));
  const Vector3 h1{h[0], h[3], h[6]};
  const Vector3 h2{h[1], h[4], h[7]};
  const Vector3 h3{h[2], h[5], h[8]};
  double lambda = 2.0 / (norm(h1) + norm(h2));
  if (!std::isfinite(lambda)) return std::nullopt;
  if (h3[2] * lambda < 0.0) lambda = -lambda;  // In front of the camera.
  const Vector3 r1{lambda * h1[0], lambda * h1[1], lambda * h1[2]};
  const Vector3 r2{lambda * h2[0], lambda * h2[1], lambda * h2[2]};
  const Vector3 r3 = cross(r1, r2);
  const auto rotation =
      nearest_rotation({r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]});
  if (!rotation) return std::nullopt;
  Pose pose;
  pose.rotation = *rotation;
  pose.translation = {lambda * h3[0], lambda * h3[1], lambda * h3[2]};
  return pose;
}

// The other pose a plane seen from afar can take: its normal mirrored about the line of sight.
Pose flipped(const Pose& pose) {
  const Vector3 n{pose.rotation[2], pose.rotation[5], pose.rotation[8]};
  const double distance = norm(pose.translation);
  const Vector3 v{pose.translation[0] / distance, pose.translation[1] / distance,
                  pose.translation[2] / distance};
  const double along = dot(n, v);
  const Vector3 mirrored{2 * along * v[0] - n[0], 2 * along * v[1] - n[1], 2 * along * v[2] - n[2]};
  Vector3 axis = cross(n, mirrored);
  const double sin_angle = norm(axis);
  const double angle = std::atan2(sin_angle, dot(n, mirrored));
  Pose result = pose;
  if (sin_angle > 1e-12) {
    axis = {axis[0] / sin_angle * angle, axis[1] / sin_angle * angle, axis[2] / sin_angle * angle};
    result.rotation = multiply(exp_so3(axis), pose.rotation);
  }
  return result;
}

}  // namespace

void validate_intrinsics(const CameraIntrinsics& k) {
  bool finite = std::isfinite(k.cx) && std::isfinite(k.cy);
  for (const double d : k.distortion) finite = finite && std::isfinite(d);
  if (!(std::isfinite(k.fx) && k.fx > 0 && std::isfinite(k.fy) && k.fy > 0 && finite)) {
    throw std::invalid_argument(
        "intrinsics need finite, positive fx and fy and finite cx, cy and distortion");
  }
}

Point2 project(const CameraIntrinsics& k, const Point3& p) {
  const double x = p[0] / p[2];
  const double y = p[1] / p[2];
  const auto& d = k.distortion;
  const double r2 = x * x + y * y;
  const double radial = 1.0 + r2 * (d[0] + r2 * (d[1] + r2 * d[4]));
  const double xd = x * radial + 2.0 * d[2] * x * y + d[3] * (r2 + 2.0 * x * x);
  const double yd = y * radial + d[2] * (r2 + 2.0 * y * y) + 2.0 * d[3] * x * y;
  return {k.fx * xd + k.cx, k.fy * yd + k.cy};
}

Point2 project(const CameraIntrinsics& k, const Pose& pose, const Point3& p) {
  return project(k, transform(pose, p));
}

Point2 undistort(const CameraIntrinsics& k, const Point2& pixel) {
  const double x0 = (pixel[0] - k.cx) / k.fx;
  const double y0 = (pixel[1] - k.cy) / k.fy;
  const auto& d = k.distortion;
  double x = x0;
  double y = y0;
  // OpenCV's fixed-point iteration.
  for (int i = 0; i < 20; ++i) {
    const double r2 = x * x + y * y;
    const double radial = 1.0 + r2 * (d[0] + r2 * (d[1] + r2 * d[4]));
    const double dx = 2.0 * d[2] * x * y + d[3] * (r2 + 2.0 * x * x);
    const double dy = d[2] * (r2 + 2.0 * y * y) + 2.0 * d[3] * x * y;
    x = (x0 - dx) / radial;
    y = (y0 - dy) / radial;
  }
  return {x, y};
}

std::optional<Pose> solve_planar_pose(std::span<const Point2> target_xy,
                                      std::span<const Point2> image, const CameraIntrinsics& k) {
  if (target_xy.size() != image.size() || image.size() < 4) return std::nullopt;
  std::vector<Point2> normalized;
  normalized.reserve(image.size());
  for (const Point2& p : image) normalized.push_back(undistort(k, p));
  const auto first = initial_pose(target_xy, normalized);
  if (!first) return std::nullopt;
  Pose best = refine(k, *first, target_xy, image);
  const Pose other = refine(k, flipped(*first), target_xy, image);
  if (other.translation[2] > 0.0 && other.reprojection_error < best.reprojection_error) {
    best = other;
  }
  if (!(best.translation[2] > 0.0) || !std::isfinite(best.reprojection_error)) return std::nullopt;
  return best;
}

std::array<Point3, 4> marker_object_corners(double length) {
  const double h = 0.5 * length;
  return {{{-h, -h, 0.0}, {h, -h, 0.0}, {h, h, 0.0}, {-h, h, 0.0}}};
}

std::optional<Pose> marker_pose(const Marker& marker, const CameraIntrinsics& k, double length) {
  const auto corners = marker_object_corners(length);
  std::array<Point2, 4> target_xy;
  for (int i = 0; i < 4; ++i) target_xy[i] = {corners[i][0], corners[i][1]};
  return solve_planar_pose(target_xy, marker.corners, k);
}

}  // namespace holoscan::fiducials
