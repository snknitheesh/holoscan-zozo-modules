// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Undistortion and Bouguet stereo rectification. The output cameras follow
// OpenCV's getOptimalNewCameraMatrix() and stereoRectify(): the same
// rotations, focal length and principal points, and the same alpha scaling
// between the rectangles inside and around each input's outline.

#include "holoscan/rectify/rectification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace holoscan::rectify {
namespace {

using Vector3 = std::array<double, 3>;

constexpr Matrix3 kIdentity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

// Border samples per side of an image for its outline; OpenCV uses 9.
constexpr int kOutlineSamples = 65;

Vector3 cross(const Vector3& a, const Vector3& b) noexcept {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

double norm(const Vector3& v) noexcept { return std::hypot(v[0], v[1], v[2]); }

// The rotation halfway from identity to `r` (its square root), from its
// quaternion: q_half = normalize(q + 1) for the q with a non-negative scalar.
Matrix3 half_rotation(const Matrix3& r) {
  // Shepperd's method.
  const double trace = r[0] + r[4] + r[8];
  double w = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  if (trace > 0) {
    const double s = std::sqrt(trace + 1.0) * 2;
    w = 0.25 * s;
    x = (r[7] - r[5]) / s;
    y = (r[2] - r[6]) / s;
    z = (r[3] - r[1]) / s;
  } else if (r[0] > r[4] && r[0] > r[8]) {
    const double s = std::sqrt(1.0 + r[0] - r[4] - r[8]) * 2;
    w = (r[7] - r[5]) / s;
    x = 0.25 * s;
    y = (r[1] + r[3]) / s;
    z = (r[2] + r[6]) / s;
  } else if (r[4] > r[8]) {
    const double s = std::sqrt(1.0 + r[4] - r[0] - r[8]) * 2;
    w = (r[2] - r[6]) / s;
    x = (r[1] + r[3]) / s;
    y = 0.25 * s;
    z = (r[5] + r[7]) / s;
  } else {
    const double s = std::sqrt(1.0 + r[8] - r[0] - r[4]) * 2;
    w = (r[3] - r[1]) / s;
    x = (r[2] + r[6]) / s;
    y = (r[5] + r[7]) / s;
    z = 0.25 * s;
  }
  if (w < 0) {
    w = -w;
    x = -x;
    y = -y;
    z = -z;
  }
  const double vx = x;
  const double vy = y;
  const double vz = z;
  const double half_angle = std::atan2(std::hypot(vx, vy, vz), w);  // The full angle over 2.
  return axis_angle({vx, vy, vz}, half_angle);
}

// The bounding box (outer) and the largest box inside (inner) of an image's
// outline, sampled along its border, after undistorting each sample, rotating
// it by `rotation` and projecting it with `k` (no projection: normalized).
struct Rectangles {
  double inner_x0{};
  double inner_y0{};
  double inner_x1{};
  double inner_y1{};
  double outer_x0{};
  double outer_y0{};
  double outer_x1{};
  double outer_y1{};
};

Rectangles outline(const CameraModel& camera, const Matrix3& rotation, const Intrinsics& k) {
  Rectangles r{.inner_x0 = -std::numeric_limits<double>::infinity(),
               .inner_y0 = -std::numeric_limits<double>::infinity(),
               .inner_x1 = std::numeric_limits<double>::infinity(),
               .inner_y1 = std::numeric_limits<double>::infinity(),
               .outer_x0 = std::numeric_limits<double>::infinity(),
               .outer_y0 = std::numeric_limits<double>::infinity(),
               .outer_x1 = -std::numeric_limits<double>::infinity(),
               .outer_y1 = -std::numeric_limits<double>::infinity()};
  const double last_x = camera.width - 1;
  const double last_y = camera.height - 1;
  const auto sample = [&](double u, double v, int side) {
    const auto ideal =
        undistort(camera, {(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy});
    if (!ideal) {
      throw std::invalid_argument(
          "the image border cannot be undistorted (the lens sees 90 degrees or more from its "
          "axis there, or the distortion folds over), so no pinhole camera holds the whole "
          "image; give explicit output intrinsics");
    }
    const Vector3 ray = rotate(rotation, {ideal->x, ideal->y, 1.0});
    if (!(ray[2] > 1e-9)) {
      throw std::invalid_argument(
          "the rectifying rotation turns part of the image behind the camera");
    }
    const double x = k.fx * ray[0] / ray[2] + k.cx;
    const double y = k.fy * ray[1] / ray[2] + k.cy;
    r.outer_x0 = std::min(r.outer_x0, x);
    r.outer_x1 = std::max(r.outer_x1, x);
    r.outer_y0 = std::min(r.outer_y0, y);
    r.outer_y1 = std::max(r.outer_y1, y);
    switch (side) {
      case 0:
        r.inner_x0 = std::max(r.inner_x0, x);
        break;
      case 1:
        r.inner_x1 = std::min(r.inner_x1, x);
        break;
      case 2:
        r.inner_y0 = std::max(r.inner_y0, y);
        break;
      default:
        r.inner_y1 = std::min(r.inner_y1, y);
        break;
    }
  };
  for (int i = 0; i < kOutlineSamples; ++i) {
    const double t = static_cast<double>(i) / (kOutlineSamples - 1);
    sample(0.0, t * last_y, 0);
    sample(last_x, t * last_y, 1);
    sample(t * last_x, 0.0, 2);
    sample(t * last_x, last_y, 3);
  }
  return r;
}

PixelRect to_pixels(double x0, double y0, double x1, double y1, int width, int height) {
  const int left = std::clamp(static_cast<int>(std::ceil(x0 - 1e-6)), 0, width);
  const int top = std::clamp(static_cast<int>(std::ceil(y0 - 1e-6)), 0, height);
  const int right = std::clamp(static_cast<int>(std::floor(x1 + 1e-6)), -1, width - 1);
  const int bottom = std::clamp(static_cast<int>(std::floor(y1 + 1e-6)), -1, height - 1);
  return {left, top, std::max(0, right - left + 1), std::max(0, bottom - top + 1)};
}

void check_alpha_and_size(double alpha, int width, int height) {
  if (!(alpha >= 0.0 && alpha <= 1.0)) throw std::invalid_argument("alpha must be in 0..1");
  if (width < 0 || height < 0 || width > kMaxImageSide || height > kMaxImageSide ||
      (width == 0) != (height == 0)) {
    throw std::invalid_argument("the output width and height must both be 0 (the input's) or 1.." +
                                std::to_string(kMaxImageSide));
  }
}

CameraModel pinhole(int width, int height, double fx, double fy, double cx, double cy) {
  CameraModel camera;
  camera.width = width;
  camera.height = height;
  camera.fx = fx;
  camera.fy = fy;
  camera.cx = cx;
  camera.cy = cy;
  return camera;
}

}  // namespace

Matrix3 multiply(const Matrix3& a, const Matrix3& b) noexcept {
  Matrix3 m{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
    }
  }
  return m;
}

Matrix3 transpose(const Matrix3& m) noexcept {
  return {m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]};
}

std::array<double, 3> rotate(const Matrix3& m, const std::array<double, 3>& v) noexcept {
  return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
          m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

Matrix3 axis_angle(const std::array<double, 3>& axis, double angle) noexcept {
  const double n = norm(axis);
  if (n < 1e-300) return kIdentity;
  const double x = axis[0] / n;
  const double y = axis[1] / n;
  const double z = axis[2] / n;
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const double t = 1 - c;
  return {t * x * x + c,     t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c,
          t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c};
}

Undistortion compute_undistortion(const CameraModel& camera, const UndistortOptions& options) {
  validate(camera);
  check_alpha_and_size(options.alpha, options.width, options.height);
  const int width = options.width != 0 ? options.width : camera.width;
  const int height = options.height != 0 ? options.height : camera.height;
  Undistortion result;
  result.input = camera;
  Intrinsics k;
  if (options.intrinsics) {
    k = *options.intrinsics;
    if (!(std::isfinite(k.fx) && k.fx > 0 && std::isfinite(k.fy) && k.fy > 0 &&
          std::isfinite(k.cx) && std::isfinite(k.cy))) {
      throw std::invalid_argument("output intrinsics need finite, positive fx and fy");
    }
  } else {
    // getOptimalNewCameraMatrix(): fit the inner (alpha 0) or the outer
    // (alpha 1) rectangle of the normalized outline to the output.
    const Rectangles n = outline(camera, kIdentity, Intrinsics{1.0, 1.0, 0.0, 0.0});
    const double inner_w = n.inner_x1 - n.inner_x0;
    const double inner_h = n.inner_y1 - n.inner_y0;
    if (!(inner_w > 0 && inner_h > 0)) {
      throw std::invalid_argument("the distortion folds the image over; no valid region remains");
    }
    const double fx0 = (width - 1) / inner_w;
    const double fy0 = (height - 1) / inner_h;
    const double fx1 = (width - 1) / (n.outer_x1 - n.outer_x0);
    const double fy1 = (height - 1) / (n.outer_y1 - n.outer_y0);
    const double a = options.alpha;
    k.fx = fx0 * (1 - a) + fx1 * a;
    k.fy = fy0 * (1 - a) + fy1 * a;
    k.cx = -fx0 * n.inner_x0 * (1 - a) - fx1 * n.outer_x0 * a;
    k.cy = -fy0 * n.inner_y0 * (1 - a) - fy1 * n.outer_y0 * a;
    if (width == 1) k.fx = k.fy;  // One column: any focal length fits.
    if (height == 1) k.fy = k.fx;
  }
  result.output = pinhole(width, height, k.fx, k.fy, k.cx, k.cy);
  const Rectangles out = outline(camera, kIdentity, k);
  result.valid = to_pixels(out.inner_x0, out.inner_y0, out.inner_x1, out.inner_y1, width, height);
  return result;
}

CameraModel StereoRectification::left_camera() const {
  return pinhole(width, height, fx, fx, cx_left, cy);
}

CameraModel StereoRectification::right_camera() const {
  return pinhole(width, height, fx, fx, cx_right, cy);
}

StereoRectification compute_stereo_rectification(const StereoCalibration& calibration,
                                                 const StereoRectifyOptions& options) {
  validate(calibration);
  check_alpha_and_size(options.alpha, options.width, options.height);
  const CameraModel& left = calibration.left;
  const CameraModel& right = calibration.right;
  const Matrix3& r = calibration.right_from_left.rotation;
  const Vector3& t = calibration.right_from_left.translation;

  // Turn each camera half of the way towards the other, so that they are
  // parallel; then both together so that the baseline lies along x.
  const Matrix3 half = half_rotation(r);  // r = half * half
  const Matrix3 half_t = transpose(half);
  const Vector3 t_half = rotate(half_t, t);
  if (std::abs(t_half[0]) < std::abs(t_half[1])) {
    throw std::invalid_argument(
        "the cameras are one above the other; only horizontal stereo pairs are rectified");
  }
  if (t_half[0] > 0) {
    throw std::invalid_argument(
        "the right camera is to the left of the left one (right_from_left.translation x > 0); "
        "swap the cameras");
  }
  const Vector3 target{t_half[0] > 0 ? 1.0 : -1.0, 0.0, 0.0};
  const Vector3 axis = cross(t_half, target);
  const double axis_norm = norm(axis);
  Matrix3 align = kIdentity;
  if (axis_norm > 0) {
    align = axis_angle(axis, std::acos(std::clamp(std::abs(t_half[0]) / norm(t_half), -1.0, 1.0)));
  }
  StereoRectification result;
  result.rect_from_left = multiply(align, half);
  result.rect_from_right = multiply(align, half_t);
  const Vector3 t_rect = rotate(result.rect_from_right, t);

  // The smaller vertical focal length, shortened for barrel distortion as
  // OpenCV does, so that the rectified images do not magnify.
  const int in_width = left.width;
  const int in_height = left.height;
  if (right.width != in_width || right.height != in_height) {
    throw std::invalid_argument("the left and right images must have the same size");
  }
  double f = std::numeric_limits<double>::infinity();
  for (const CameraModel* camera : {&left, &right}) {
    double fc = camera->fy;
    const double k1 = camera->distortion_model == DistortionModel::kKannalaBrandt4
                          ? 0.0
                          : coefficient(*camera, 0);
    if (k1 < 0) {
      fc *= 1 + k1 *
                    (static_cast<double>(in_width) * in_width +
                     static_cast<double>(in_height) * in_height) /
                    (4 * fc * fc);
    }
    f = std::min(f, fc);
  }
  if (!(f > 0)) throw std::invalid_argument("the distortion leaves no usable focal length");

  // Principal points that center each image's corners.
  double cc[2][2];
  const Matrix3* rotations[2] = {&result.rect_from_left, &result.rect_from_right};
  const CameraModel* cameras[2] = {&left, &right};
  for (int c = 0; c < 2; ++c) {
    double sx = 0.0;
    double sy = 0.0;
    for (int i = 0; i < 4; ++i) {
      const double u = (i % 2) * (in_width - 1.0);
      const double v = (i / 2) * (in_height - 1.0);
      const CameraModel& camera = *cameras[c];
      const auto ideal =
          undistort(camera, {(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy});
      if (!ideal) {
        throw std::invalid_argument(
            "an image corner cannot be undistorted (90 degrees or more from the lens axis, or "
            "where the distortion folds over); rectification needs a narrower lens");
      }
      const Vector3 ray = rotate(*rotations[c], {ideal->x, ideal->y, 1.0});
      if (!(ray[2] > 1e-9)) {
        throw std::invalid_argument("rectifying turns an image corner behind the camera");
      }
      sx += f * ray[0] / ray[2];
      sy += f * ray[1] / ray[2];
    }
    cc[c][0] = (in_width - 1) / 2.0 - sx / 4;
    cc[c][1] = (in_height - 1) / 2.0 - sy / 4;
  }
  if (options.zero_disparity) {
    cc[0][0] = cc[1][0] = (cc[0][0] + cc[1][0]) / 2;
  }
  cc[0][1] = cc[1][1] = (cc[0][1] + cc[1][1]) / 2;

  // Alpha: between the scale where the inner rectangles fill the output (0)
  // and the one where the outer rectangles fit in it (1).
  const int width = options.width != 0 ? options.width : in_width;
  const int height = options.height != 0 ? options.height : in_height;
  const double sw = static_cast<double>(width) / in_width;
  const double sh = static_cast<double>(height) / in_height;
  double s0 = 0.0;
  double s1 = std::numeric_limits<double>::infinity();
  double cx_new[2];
  double cy_new[2];
  Rectangles rects[2];
  for (int c = 0; c < 2; ++c) {
    rects[c] = outline(*cameras[c], *rotations[c], Intrinsics{f, f, cc[c][0], cc[c][1]});
    const Rectangles& q = rects[c];
    const double cx0 = cc[c][0];
    const double cy0 = cc[c][1];
    const double cx = cx0 * sw;
    const double cy = cy0 * sh;
    cx_new[c] = cx;
    cy_new[c] = cy;
    s0 = std::max({s0, cx / (cx0 - q.inner_x0), cy / (cy0 - q.inner_y0),
                   (width - 1 - cx) / (q.inner_x1 - cx0), (height - 1 - cy) / (q.inner_y1 - cy0)});
    s1 = std::min({s1, cx / (cx0 - q.outer_x0), cy / (cy0 - q.outer_y0),
                   (width - 1 - cx) / (q.outer_x1 - cx0), (height - 1 - cy) / (q.outer_y1 - cy0)});
  }
  const double s = s0 * (1 - options.alpha) + s1 * options.alpha;
  if (!(std::isfinite(s) && s > 0)) {
    throw std::invalid_argument("the cameras share no view to rectify into");
  }
  const double f_new = f * s;

  result.width = width;
  result.height = height;
  result.fx = f_new;
  result.cx_left = cx_new[0];
  result.cx_right = cx_new[1];
  result.cy = cy_new[0];
  result.baseline = -t_rect[0];
  result.doffs = result.cx_right - result.cx_left;
  result.projection_left = {f_new, 0, result.cx_left, 0, 0, f_new, result.cy, 0, 0, 0, 1, 0};
  result.projection_right = {
      f_new, 0.0, result.cx_right, f_new * t_rect[0], 0.0, f_new, result.cy, 0.0, 0.0, 0.0,
      1.0,   0.0};
  const double tx = t_rect[0];
  result.disparity_to_depth = {
      1.0, 0.0, 0.0, -result.cx_left, 0.0, 1.0, 0.0,       -result.cy,
      0.0, 0.0, 0.0, f_new,           0.0, 0.0, -1.0 / tx, (result.cx_left - result.cx_right) / tx};
  // The inner rectangles, scaled into the output.
  for (int c = 0; c < 2; ++c) {
    const Rectangles& q = rects[c];
    const auto map_x = [&](double x) { return cx_new[c] + (x - cc[c][0]) * s; };
    const auto map_y = [&](double y) { return cy_new[c] + (y - cc[c][1]) * s; };
    (c == 0 ? result.valid_left : result.valid_right) = to_pixels(
        map_x(q.inner_x0), map_y(q.inner_y0), map_x(q.inner_x1), map_y(q.inner_y1), width, height);
  }
  return result;
}

RemapModel::RemapModel(CameraModel input, const Matrix3& rect_from_camera, const Intrinsics& output)
    : input_(std::move(input)), camera_from_rect_(transpose(rect_from_camera)), output_(output) {}

std::optional<Point2> RemapModel::source(double u, double v) const noexcept {
  const Vector3 ray = rotate(camera_from_rect_,
                             {(u - output_.cx) / output_.fx, (v - output_.cy) / output_.fy, 1.0});
  Point2 distorted;
  if (input_.distortion_model == DistortionModel::kKannalaBrandt4) {
    // The fisheye angle is defined beyond 90 degrees, so take it from the ray.
    const double r = std::hypot(ray[0], ray[1]);
    if (r < 1e-15) {
      if (!(ray[2] > 0)) return std::nullopt;
      distorted = {0.0, 0.0};
    } else {
      const double theta = std::atan2(r, ray[2]);
      const double t2 = theta * theta;
      const double theta_d =
          theta * (1 + t2 * (coefficient(input_, 0) +
                             t2 * (coefficient(input_, 1) +
                                   t2 * (coefficient(input_, 2) + t2 * coefficient(input_, 3)))));
      distorted = {ray[0] / r * theta_d, ray[1] / r * theta_d};
    }
  } else {
    if (!(ray[2] > 1e-12)) return std::nullopt;
    distorted = distort(input_, {ray[0] / ray[2], ray[1] / ray[2]});
  }
  const Point2 pixel{input_.fx * distorted.x + input_.cx, input_.fy * distorted.y + input_.cy};
  if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y)) return std::nullopt;
  return pixel;
}

Point2 RemapModel::clamped_source(double u, double v) const noexcept {
  const auto p = source(u, v);
  if (!p) return {-2.0, -2.0};
  // Within half a pixel outside the image, onto its edge: that is still the
  // edge pixel's area, and the backends sample nothing outside [0, size - 1].
  const auto edge = [](double c, int size) {
    if (c < 0.0 && c >= -0.5) return 0.0;
    if (c > size - 1.0 && c <= size - 0.5) return size - 1.0;
    return std::clamp(c, -2.0, size + 1.0);
  };
  return {edge(p->x, input_.width), edge(p->y, input_.height)};
}

RemapModel undistortion_remap(const Undistortion& undistortion) {
  const CameraModel& out = undistortion.output;
  return RemapModel(undistortion.input, kIdentity, Intrinsics{out.fx, out.fy, out.cx, out.cy});
}

RemapModel left_remap(const StereoCalibration& calibration,
                      const StereoRectification& rectification) {
  return RemapModel(
      calibration.left, rectification.rect_from_left,
      Intrinsics{rectification.fx, rectification.fx, rectification.cx_left, rectification.cy});
}

RemapModel right_remap(const StereoCalibration& calibration,
                       const StereoRectification& rectification) {
  return RemapModel(
      calibration.right, rectification.rect_from_right,
      Intrinsics{rectification.fx, rectification.fx, rectification.cx_right, rectification.cy});
}

RemapTable build_remap_table(const RemapModel& model, int width, int height) {
  RemapTable table{width, height, {}};
  table.xy.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 2U);
  std::size_t i = 0;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const Point2 p = model.clamped_source(u, v);
      table.xy[i++] = static_cast<float>(p.x);
      table.xy[i++] = static_cast<float>(p.y);
    }
  }
  return table;
}

}  // namespace holoscan::rectify
