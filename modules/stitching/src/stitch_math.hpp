// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The arithmetic of stitching, shared by the CUDA kernels and the CPU so that
// both compute the same: an output pixel's 3D point, a camera's projection of
// it, bilinear sampling, the blend of one output pixel, the overlap statistics
// and the gain solve. Everything that must agree to the byte is compiled
// without floating-point contraction (CMakeLists.txt). The geometry runs in
// double; the synthetic renderer instantiates the camera model in float.

#pragma once

#include <math.h>

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define STITCH_HD __host__ __device__ __forceinline__
#define STITCH_UNROLL _Pragma("unroll")
#else
#define STITCH_HD inline
#define STITCH_UNROLL
#endif

namespace holoscan::stitching {
struct Camera;
struct OutputGeometry;
struct BlendSettings;
}  // namespace holoscan::stitching

namespace holoscan::stitching::detail {

inline constexpr int kMaxCameras = 8;
// Overlaps with fewer samples than this are left out of the gain solve.
inline constexpr std::uint32_t kMinOverlapSamples = 16U;

// The C math functions, which host and device code both have, for each type.
STITCH_HD double m_sqrt(double x) { return sqrt(x); }
STITCH_HD float m_sqrt(float x) { return sqrtf(x); }
STITCH_HD double m_atan2(double y, double x) { return atan2(y, x); }
STITCH_HD float m_atan2(float y, float x) { return atan2f(y, x); }
STITCH_HD double m_sin(double x) { return sin(x); }
STITCH_HD float m_sin(float x) { return sinf(x); }
STITCH_HD double m_cos(double x) { return cos(x); }
STITCH_HD float m_cos(float x) { return cosf(x); }
STITCH_HD double m_fabs(double x) { return fabs(x); }
STITCH_HD float m_fabs(float x) { return fabsf(x); }
template <typename T>
STITCH_HD T m_min(T a, T b) {
  return b < a ? b : a;
}
template <typename T>
STITCH_HD T m_max(T a, T b) {
  return a < b ? b : a;
}

// --- Cameras ---------------------------------------------------------------

enum : int { kModelNone = 0, kModelBrownConrady = 1, kModelRational = 2, kModelKannalaBrandt4 = 3 };

// A camera as the kernels read it. k holds k1, k2, p1, p2, k3, k4, k5, k6 for
// the pinhole models (zero where a model has none), k1..k4 for the fisheye.
struct CameraParams {
  int width;
  int height;
  double fx;
  double fy;
  double cx;
  double cy;
  int model;
  double k[8];
  double rotation[9];  // camera_from_rig, row-major.
  double translation[3];
  double max_angle;  // Radians from the optical axis; see max_valid_angle().
};

// Host only: the parameters of a validated camera.
[[nodiscard]] CameraParams camera_params(const Camera& camera);

// The lens' distortion of a ray at angle `theta` from the axis whose
// direction in the image plane is (dx, dy), unit length (or 0, 0 on the
// axis), giving its distorted normalized coordinates.
template <typename T>
STITCH_HD void distort_ray(const CameraParams& c, T theta, T dx, T dy, T x, T y, T& xd, T& yd) {
  const T k0 = static_cast<T>(c.k[0]);
  const T k1 = static_cast<T>(c.k[1]);
  const T k2 = static_cast<T>(c.k[2]);
  const T k3 = static_cast<T>(c.k[3]);
  if (c.model == kModelKannalaBrandt4) {
    const T t2 = theta * theta;
    const T theta_d = theta * (T(1) + t2 * (k0 + t2 * (k1 + t2 * (k2 + t2 * k3))));
    xd = theta_d * dx;
    yd = theta_d * dy;
    return;
  }
  // Pinhole: x, y are the tangent-plane coordinates.
  const T k4 = static_cast<T>(c.k[4]);
  const T k5 = static_cast<T>(c.k[5]);
  const T k6 = static_cast<T>(c.k[6]);
  const T k7 = static_cast<T>(c.k[7]);
  const T r2 = x * x + y * y;
  const T a = T(1) + r2 * (k0 + r2 * (k1 + r2 * k4));
  const T b = T(1) + r2 * (k5 + r2 * (k6 + r2 * k7));
  const T radial = a / b;
  xd = x * radial + T(2) * k2 * x * y + k3 * (r2 + T(2) * x * x);
  yd = y * radial + k2 * (r2 + T(2) * y * y) + T(2) * k3 * x * y;
}

// Projects a point (or, with `at_infinity`, a direction) of the rig frame
// into the camera: its pixel (u, v) and its margin, the distance in pixels to
// where the camera stops seeing. False when it does not see the point.
template <typename T>
STITCH_HD bool project(const CameraParams& c, const T p[3], bool at_infinity, T& u, T& v,
                       T& margin) {
  T X[3];
  for (int row = 0; row < 3; ++row) {
    X[row] = static_cast<T>(c.rotation[3 * row]) * p[0] +
             static_cast<T>(c.rotation[3 * row + 1]) * p[1] +
             static_cast<T>(c.rotation[3 * row + 2]) * p[2];
    if (!at_infinity) {
      X[row] = X[row] + static_cast<T>(c.translation[row]);
    }
  }
  const T r = m_sqrt(X[0] * X[0] + X[1] * X[1]);
  const T theta = m_atan2(r, X[2]);
  const T max_angle = static_cast<T>(c.max_angle);
  if (!(theta < max_angle)) {
    return false;
  }
  T xd;
  T yd;
  if (c.model == kModelKannalaBrandt4) {
    const T dx = r > T(0) ? X[0] / r : T(0);
    const T dy = r > T(0) ? X[1] / r : T(0);
    distort_ray(c, theta, dx, dy, T(0), T(0), xd, yd);
  } else {
    // theta < max_angle < pi/2, so X[2] > 0.
    distort_ray(c, theta, T(0), T(0), X[0] / X[2], X[1] / X[2], xd, yd);
  }
  const T fx = static_cast<T>(c.fx);
  const T fy = static_cast<T>(c.fy);
  u = fx * xd + static_cast<T>(c.cx);
  v = fy * yd + static_cast<T>(c.cy);
  const T half = T(0.5);
  margin = m_min(m_min(u + half, static_cast<T>(c.width) - half - u),
                 m_min(v + half, static_cast<T>(c.height) - half - v));
  margin = m_min(margin, (max_angle - theta) * m_min(fx, fy));
  return margin > T(0);
}

// The ray of pixel (u, v): a unit vector in the camera's frame, by Newton's
// method. False when no ray within the lens' valid range lands there.
template <typename T>
STITCH_HD bool pixel_ray(const CameraParams& c, T u, T v, T ray[3]) {
  const T xd = (u - static_cast<T>(c.cx)) / static_cast<T>(c.fx);
  const T yd = (v - static_cast<T>(c.cy)) / static_cast<T>(c.fy);
  const T max_angle = static_cast<T>(c.max_angle);
  const T tolerance = sizeof(T) == sizeof(double) ? T(1e-12) : T(1e-5);
  if (c.model == kModelKannalaBrandt4) {
    const T theta_d = m_sqrt(xd * xd + yd * yd);
    if (theta_d == T(0)) {
      ray[0] = T(0);
      ray[1] = T(0);
      ray[2] = T(1);
      return true;
    }
    const T k0 = static_cast<T>(c.k[0]);
    const T k1 = static_cast<T>(c.k[1]);
    const T k2 = static_cast<T>(c.k[2]);
    const T k3 = static_cast<T>(c.k[3]);
    T theta = m_min(theta_d, max_angle);
    for (int i = 0; i < 20; ++i) {
      const T t2 = theta * theta;
      const T f = theta * (T(1) + t2 * (k0 + t2 * (k1 + t2 * (k2 + t2 * k3)))) - theta_d;
      const T df = T(1) + t2 * (T(3) * k0 + t2 * (T(5) * k1 + t2 * (T(7) * k2 + t2 * T(9) * k3)));
      theta = theta - f / df;
      theta = m_max(T(0), m_min(theta, max_angle));
    }
    const T t2 = theta * theta;
    const T residual = theta * (T(1) + t2 * (k0 + t2 * (k1 + t2 * (k2 + t2 * k3)))) - theta_d;
    if (!(theta < max_angle) || !(m_fabs(residual) <= tolerance * m_max(T(1), theta_d))) {
      return false;
    }
    const T s = m_sin(theta) / theta_d;
    ray[0] = xd * s;
    ray[1] = yd * s;
    ray[2] = m_cos(theta);
    return true;
  }
  // Pinhole: Newton on the distortion, from the distorted point.
  const T k0 = static_cast<T>(c.k[0]);
  const T k1 = static_cast<T>(c.k[1]);
  const T p1 = static_cast<T>(c.k[2]);
  const T p2 = static_cast<T>(c.k[3]);
  const T k4 = static_cast<T>(c.k[4]);
  const T k5 = static_cast<T>(c.k[5]);
  const T k6 = static_cast<T>(c.k[6]);
  const T k7 = static_cast<T>(c.k[7]);
  T x = xd;
  T y = yd;
  T ex = T(0);
  T ey = T(0);
  for (int i = 0; i < 20; ++i) {
    const T r2 = x * x + y * y;
    const T a = T(1) + r2 * (k0 + r2 * (k1 + r2 * k4));
    const T b = T(1) + r2 * (k5 + r2 * (k6 + r2 * k7));
    const T da = k0 + r2 * (T(2) * k1 + r2 * T(3) * k4);
    const T db = k5 + r2 * (T(2) * k6 + r2 * T(3) * k7);
    const T radial = a / b;
    const T dradial = (da * b - a * db) / (b * b);  // d radial / d r2
    T dx;
    T dy;
    distort_ray(c, T(0), T(0), T(0), x, y, dx, dy);
    ex = dx - xd;
    ey = dy - yd;
    const T j00 = radial + T(2) * x * x * dradial + T(2) * p1 * y + T(6) * p2 * x;
    const T j01 = T(2) * x * y * dradial + T(2) * p1 * x + T(2) * p2 * y;
    const T j11 = radial + T(2) * y * y * dradial + T(6) * p1 * y + T(2) * p2 * x;
    const T det = j00 * j11 - j01 * j01;
    if (det == T(0)) {
      return false;
    }
    x = x - (j11 * ex - j01 * ey) / det;
    y = y - (j00 * ey - j01 * ex) / det;
  }
  T dx;
  T dy;
  distort_ray(c, T(0), T(0), T(0), x, y, dx, dy);
  ex = dx - xd;
  ey = dy - yd;
  if (!(m_fabs(ex) <= tolerance && m_fabs(ey) <= tolerance)) {
    return false;
  }
  const T norm = m_sqrt(x * x + y * y + T(1));
  ray[0] = x / norm;
  ray[1] = y / norm;
  ray[2] = T(1) / norm;
  return m_atan2(m_sqrt(x * x + y * y), T(1)) < max_angle;
}

// --- Output geometry -------------------------------------------------------

enum : int { kCylindrical = 0, kSpherical = 1, kBirdsEye = 2 };

// The geometry as the kernels read it: affine maps from pixel centres.
struct GeometryParams {
  int projection;
  int width;
  int height;
  // Panorama: azimuth at x = -0.5 and its decrease per pixel; elevation
  // (spherical) or its tangent (cylindrical) at y = -0.5 and its decrease per
  // pixel, in radians.
  double azimuth_left;
  double azimuth_step;
  double row_top;
  double row_step;
  double center[3];
  double radius;
  // Bird's-eye: x at y = -0.5 and its decrease per row; y at x = -0.5 and its
  // decrease per column, in metres.
  double x_top;
  double x_step;
  double y_left;
  double y_step;
};

// Host only: the parameters of a validated geometry.
[[nodiscard]] GeometryParams geometry_params(const OutputGeometry& geometry);

// The rig-frame point of output pixel (x, y); true when it is a direction.
STITCH_HD bool output_point(const GeometryParams& g, double x, double y, double p[3]) {
  if (g.projection == kBirdsEye) {
    p[0] = g.x_top - (y + 0.5) * g.x_step;
    p[1] = g.y_left - (x + 0.5) * g.y_step;
    p[2] = 0.0;
    return false;
  }
  const double azimuth = g.azimuth_left - (x + 0.5) * g.azimuth_step;
  const double row = g.row_top - (y + 0.5) * g.row_step;
  double d[3];
  if (g.projection == kSpherical) {
    const double ce = cos(row);
    d[0] = ce * cos(azimuth);
    d[1] = ce * sin(azimuth);
    d[2] = sin(row);
  } else {
    d[0] = cos(azimuth);
    d[1] = sin(azimuth);
    d[2] = row;
  }
  if (g.radius > 0.0) {
    for (int i = 0; i < 3; ++i) {
      p[i] = g.center[i] + g.radius * d[i];
    }
    return false;
  }
  for (int i = 0; i < 3; ++i) {
    p[i] = d[i];
  }
  return true;
}

// Where camera `c` sees output pixel (x, y): false when it does not.
STITCH_HD bool remap_entry(const GeometryParams& g, const CameraParams& c, int x, int y, float& u,
                           float& v, float& margin) {
  double p[3];
  const bool at_infinity = output_point(g, static_cast<double>(x), static_cast<double>(y), p);
  double du;
  double dv;
  double dm;
  if (!project(c, p, at_infinity, du, dv, dm)) {
    return false;
  }
  u = static_cast<float>(du);
  v = static_cast<float>(dv);
  margin = static_cast<float>(dm);
  // A margin that rounds to 0 in float would read as "not seen".
  return margin > 0.0F;
}

// --- Sampling and blending -------------------------------------------------

enum : int { kMono8 = 0, kRgb8 = 1, kRgba8 = 2, kBgr8 = 3, kBgra8 = 4 };

// One camera's image and table as the blend reads them.
struct CameraSampler {
  const std::uint8_t* image;
  std::size_t pitch;
  int width;
  int height;
  int channels;  // 1, 3 or 4.
  bool bgr;
  int x0;
  int y0;
  int roi_width;
  int roi_height;
  const float* uv;
  const float* margin;
};

struct BlendParams {
  int cameras;
  CameraSampler camera[kMaxCameras];
  int output_width;
  int output_height;
  bool wraps;
  bool hard_seam;
  float feather_width;
  int output_channels;  // 3 or 4.
  std::uint8_t background[3];
};

// The index of output pixel (x, y) in camera `s`'s table, or -1 outside its
// region of interest.
STITCH_HD std::ptrdiff_t table_index(const CameraSampler& s, int output_width, bool wraps, int x,
                                     int y) {
  int dx = x - s.x0;
  if (wraps && dx < 0) {
    dx += output_width;
  }
  const int dy = y - s.y0;
  if (dx < 0 || dx >= s.roi_width || dy < 0 || dy >= s.roi_height) {
    return -1;
  }
  return static_cast<std::ptrdiff_t>(dy) * s.roi_width + dx;
}

// Bilinear RGB at camera pixel (u, v), pixel centres at integers, the taps
// clamped to the image.
STITCH_HD void sample_rgb(const CameraSampler& s, float u, float v, float rgb[3]) {
  const float fu = floorf(u);
  const float fv = floorf(v);
  const float ax = u - fu;
  const float ay = v - fv;
  const int xi = static_cast<int>(fu);
  const int yi = static_cast<int>(fv);
  const int xa = m_max(0, m_min(xi, s.width - 1));
  const int xb = m_max(0, m_min(xi + 1, s.width - 1));
  const int ya = m_max(0, m_min(yi, s.height - 1));
  const int yb = m_max(0, m_min(yi + 1, s.height - 1));
  const std::uint8_t* top = s.image + static_cast<std::size_t>(ya) * s.pitch;
  const std::uint8_t* bottom = s.image + static_cast<std::size_t>(yb) * s.pitch;
  const int n = s.channels;
  for (int ch = 0; ch < 3; ++ch) {
    const int k = n == 1 ? 0 : (s.bgr ? 2 - ch : ch);
    const float p00 = static_cast<float>(top[xa * n + k]);
    const float p10 = static_cast<float>(top[xb * n + k]);
    const float p01 = static_cast<float>(bottom[xa * n + k]);
    const float p11 = static_cast<float>(bottom[xb * n + k]);
    const float upper = p00 + ax * (p10 - p00);
    const float lower = p01 + ax * (p11 - p01);
    rgb[ch] = upper + ay * (lower - upper);
  }
}

STITCH_HD std::uint8_t to_byte(float value) {
  const float rounded = floorf(value + 0.5F);
  return static_cast<std::uint8_t>(m_max(0.0F, m_min(255.0F, rounded)));
}

// Blends output pixel (x, y) into `out` (3 or 4 bytes).
STITCH_HD void blend_pixel(const BlendParams& p, const float* gains, int x, int y,
                           std::uint8_t* out) {
  float acc[3] = {0.0F, 0.0F, 0.0F};
  float weight_sum = 0.0F;
  int best = -1;
  float best_margin = 0.0F;
  float best_u = 0.0F;
  float best_v = 0.0F;
  // Unrolled with constant indices, so that the kernel reads its parameters
  // where they are instead of copying them to local memory.
  STITCH_UNROLL
  for (int c = 0; c < kMaxCameras; ++c) {
    if (c >= p.cameras) {
      break;
    }
    const CameraSampler& s = p.camera[c];
    const std::ptrdiff_t i = table_index(s, p.output_width, p.wraps, x, y);
    if (i < 0) {
      continue;
    }
    const float margin = s.margin[i];
    if (!(margin > 0.0F)) {
      continue;
    }
    const float u = s.uv[2 * i];
    const float v = s.uv[2 * i + 1];
    if (p.hard_seam) {
      if (margin > best_margin) {
        best = c;
        best_margin = margin;
        best_u = u;
        best_v = v;
      }
      continue;
    }
    const float w = p.feather_width > 0.0F ? m_min(margin, p.feather_width) : margin;
    float rgb[3];
    sample_rgb(s, u, v, rgb);
    const float wg = w * gains[c];
    for (int ch = 0; ch < 3; ++ch) {
      acc[ch] = acc[ch] + wg * rgb[ch];
    }
    weight_sum = weight_sum + w;
  }
  if (best >= 0) {
    STITCH_UNROLL
    for (int c = 0; c < kMaxCameras; ++c) {
      if (c == best) {
        float rgb[3];
        sample_rgb(p.camera[c], best_u, best_v, rgb);
        for (int ch = 0; ch < 3; ++ch) {
          acc[ch] = gains[c] * rgb[ch];
        }
      }
    }
    weight_sum = 1.0F;
  }
  if (weight_sum > 0.0F) {
    for (int ch = 0; ch < 3; ++ch) {
      out[ch] = to_byte(acc[ch] / weight_sum);
    }
    if (p.output_channels == 4) {
      out[3] = 255;
    }
    return;
  }
  for (int ch = 0; ch < 3; ++ch) {
    out[ch] = p.background[ch];
  }
  if (p.output_channels == 4) {
    out[3] = 0;
  }
}

// --- Exposure gains --------------------------------------------------------

// The luminance levels of each camera that sees exposure sample (x, y), or -1
// where it does not.
STITCH_HD void sample_luminance(const BlendParams& p, int x, int y, int luminance[kMaxCameras]) {
  STITCH_UNROLL
  for (int c = 0; c < kMaxCameras; ++c) {
    luminance[c] = -1;
    if (c >= p.cameras) {
      continue;
    }
    const CameraSampler& s = p.camera[c];
    const std::ptrdiff_t i = table_index(s, p.output_width, p.wraps, x, y);
    if (i < 0 || !(s.margin[i] > 0.0F)) {
      continue;
    }
    float rgb[3];
    sample_rgb(s, s.uv[2 * i], s.uv[2 * i + 1], rgb);
    const float level = 0.299F * rgb[0] + 0.587F * rgb[1] + 0.114F * rgb[2];
    luminance[c] = static_cast<int>(to_byte(level));
  }
}

// The gains from the overlap statistics (see reference_overlap_stats()), by
// solving the normal equations of Brown and Lowe's gain error with Gaussian
// elimination, then blending with `previous`. `alpha` = 1 / sigma_N^2,
// `beta` = 1 / sigma_g^2. A camera without overlaps keeps gain 1.
STITCH_HD void solve_gains(int n, const unsigned long long* sum,  // NOLINT(runtime/int)
                           const std::uint32_t* count, double alpha, double beta, double smoothing,
                           const float* previous, float* gains) {
  double a[kMaxCameras][kMaxCameras + 1];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= n; ++j) {
      a[i][j] = 0.0;
    }
    for (int j = 0; j < n; ++j) {
      const std::uint32_t nij = count[i * n + j];
      if (i == j || nij < kMinOverlapSamples) {
        continue;
      }
      const double mean_i = static_cast<double>(sum[i * n + j]) / static_cast<double>(nij);
      const double mean_j = static_cast<double>(sum[j * n + i]) / static_cast<double>(nij);
      const double weight = static_cast<double>(nij);
      a[i][i] = a[i][i] + weight * (2.0 * alpha * mean_i * mean_i + beta);
      a[i][j] = a[i][j] - weight * 2.0 * alpha * mean_i * mean_j;
      a[i][n] = a[i][n] + weight * beta;
    }
    if (a[i][i] == 0.0) {
      a[i][i] = 1.0;
      a[i][n] = 1.0;
    }
  }
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    for (int row = col + 1; row < n; ++row) {
      if (m_fabs(a[row][col]) > m_fabs(a[pivot][col])) {
        pivot = row;
      }
    }
    if (pivot != col) {
      for (int j = 0; j <= n; ++j) {
        const double t = a[col][j];
        a[col][j] = a[pivot][j];
        a[pivot][j] = t;
      }
    }
    for (int row = col + 1; row < n; ++row) {
      const double f = a[row][col] / a[col][col];
      for (int j = col; j <= n; ++j) {
        a[row][j] = a[row][j] - f * a[col][j];
      }
    }
  }
  double g[kMaxCameras];
  for (int i = n - 1; i >= 0; --i) {
    double s = a[i][n];
    for (int j = i + 1; j < n; ++j) {
      s = s - a[i][j] * g[j];
    }
    g[i] = s / a[i][i];
  }
  for (int i = 0; i < n; ++i) {
    const double blended = (1.0 - smoothing) * g[i] + smoothing * static_cast<double>(previous[i]);
    gains[i] = static_cast<float>(blended);
  }
}

}  // namespace holoscan::stitching::detail
