// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/messages/camera_info.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "geometry.hpp"

namespace holoscan::messages {
namespace {

using detail::kPairTolerance;

std::string field(std::string_view name, std::string_view member) {
  std::string result(name);
  result += '.';
  result += member;
  return result;
}

std::string number(double value) {
  std::ostringstream text;
  text.precision(17);
  text << value;
  return text.str();
}

[[noreturn]] void fail(const std::string& what) { throw std::invalid_argument(what); }

void check_positive(double value, std::string_view name, std::string_view member) {
  if (!(std::isfinite(value) && value > 0)) {
    fail(field(name, member) + " must be finite and > 0, not " + number(value));
  }
}

void check_finite(double value, std::string_view name, std::string_view member) {
  if (!std::isfinite(value)) fail(field(name, member) + " must be finite, not " + number(value));
}

void check_frame_id(const Header& header, std::string_view name) {
  if (header.frame_id.size() > kMaxFrameIdBytes) {
    fail(field(name, "header.frame_id") + " has " + std::to_string(header.frame_id.size()) +
         " bytes; at most " + std::to_string(kMaxFrameIdBytes) + " fit in a message");
  }
}

bool has_distortion(const CameraInfo& info) {
  return std::any_of(info.coefficients.begin(), info.coefficients.end(),
                     [](double c) { return c != 0.0; });
}

void check_distortion(const CameraInfo& info, std::string_view name) {
  const std::size_t expected = coefficient_count(info.distortion_model);
  if (info.coefficients.size() != expected) {
    fail(field(name, "coefficients") + " has " + std::to_string(info.coefficients.size()) +
         " values; the " + std::string(distortion_model_name(info.distortion_model)) +
         " model has " + std::to_string(expected));
  }
  for (std::size_t i = 0; i < info.coefficients.size(); ++i) {
    check_finite(info.coefficients[i], name, "coefficients[" + std::to_string(i) + "]");
  }
  const bool distorted = has_distortion(info);
  if (info.state == CameraState::kRaw && !distorted) {
    fail(field(name, "state") +
         " is raw, but every distortion coefficient is zero: a camera without distortion is "
         "undistorted");
  }
  if (info.state != CameraState::kRaw && distorted) {
    fail(field(name, "state") + " is " + std::string(camera_state_name(info.state)) +
         ", but a distortion coefficient is not zero");
  }
}

void check_roi(const CameraInfo& info, std::string_view name) {
  if (!info.roi) return;
  const RegionOfInterest& roi = *info.roi;
  if (roi.width == 0 || roi.height == 0) fail(field(name, "roi") + " is empty");
  const std::uint64_t sensor_width = std::uint64_t{info.width} * info.binning_x;
  const std::uint64_t sensor_height = std::uint64_t{info.height} * info.binning_y;
  if (std::uint64_t{roi.x} + roi.width > sensor_width ||
      std::uint64_t{roi.y} + roi.height > sensor_height) {
    fail(field(name, "roi") + " reaches past the " + std::to_string(sensor_width) + "x" +
         std::to_string(sensor_height) + " sensor (the image size times the binning)");
  }
}

void check_rotation(const Matrix3& m, std::string_view name, std::string_view member) {
  if (!detail::is_rotation(m)) {
    fail(field(name, member) + " is not a rotation (orthonormal, determinant +1)");
  }
}

void check_rectified_pair(const StereoCameraInfo& info, std::string_view name) {
  const CameraInfo& l = info.left;
  const CameraInfo& r = info.right;
  const auto same = [&](double a, double b, std::string_view what) {
    if (!detail::nearly_equal(a, b, kPairTolerance)) {
      fail(std::string(name) + ": a rectified pair has one " + std::string(what) + ", not " +
           number(a) + " and " + number(b));
    }
  };
  same(l.fx, r.fx, "fx");
  same(l.fy, r.fy, "fy");
  same(l.cy, r.cy, "cy");
  if (l.skew != 0.0 || r.skew != 0.0) fail(std::string(name) + ": a rectified pair has no skew");
  check_positive(info.baseline, name, "baseline");
  if (detail::max_abs_difference(info.right_from_left.rotation, kIdentity3) >
      detail::kRotationTolerance) {
    fail(field(name, "right_from_left.rotation") + " of a rectified pair must be the identity");
  }
  const Vector3& t = info.right_from_left.translation;
  if (!detail::nearly_equal(t[0], -info.baseline, kPairTolerance) ||
      std::abs(t[1]) > kPairTolerance * info.baseline ||
      std::abs(t[2]) > kPairTolerance * info.baseline) {
    fail(field(name, "right_from_left.translation") + " of a rectified pair must be (-" +
         number(info.baseline) + ", 0, 0), the baseline along -x");
  }
  if (!detail::nearly_equal(info.doffs, r.cx - l.cx, kPairTolerance)) {
    fail(field(name, "doffs") + " must be cx_right - cx_left = " + number(r.cx - l.cx) + ", not " +
         number(info.doffs));
  }
}

void check_rectifications(const StereoCameraInfo& info, std::string_view name) {
  if (!info.left.rectification && !info.right.rectification) return;
  if (!detail::rectifications_agree(info)) {
    fail(std::string(name) +
         ": the rectifications do not agree with right_from_left: right.rectification^T * "
         "left.rectification must be its rotation, and right.rectification must turn its "
         "translation onto -x");
  }
}

}  // namespace

std::size_t coefficient_count(DistortionModel model) noexcept {
  switch (model) {
    case DistortionModel::kPlumbBob:
      return 5;
    case DistortionModel::kRationalPolynomial:
      return 8;
    case DistortionModel::kEquidistant:
      return 4;
    case DistortionModel::kNone:
      break;
  }
  return 0;
}

std::string_view distortion_model_name(DistortionModel model) noexcept {
  switch (model) {
    case DistortionModel::kPlumbBob:
      return "plumb_bob";
    case DistortionModel::kRationalPolynomial:
      return "rational_polynomial";
    case DistortionModel::kEquidistant:
      return "equidistant";
    case DistortionModel::kNone:
      break;
  }
  return "none";
}

std::optional<DistortionModel> parse_distortion_model(std::string_view name) noexcept {
  if (name.empty() || name == "none") return DistortionModel::kNone;
  if (name == "plumb_bob") return DistortionModel::kPlumbBob;
  if (name == "rational_polynomial") return DistortionModel::kRationalPolynomial;
  if (name == "equidistant") return DistortionModel::kEquidistant;
  return std::nullopt;
}

std::string_view camera_state_name(CameraState state) noexcept {
  switch (state) {
    case CameraState::kUndistorted:
      return "undistorted";
    case CameraState::kRectified:
      return "rectified";
    case CameraState::kRaw:
      break;
  }
  return "raw";
}

void validate(const CameraInfo& info, std::string_view name) {
  check_frame_id(info.header, name);
  if (info.width == 0 || info.height == 0) {
    fail(field(name, "width") + " and " + field(name, "height") + " must be > 0, not " +
         std::to_string(info.width) + "x" + std::to_string(info.height));
  }
  if (info.state != CameraState::kRaw && info.state != CameraState::kUndistorted &&
      info.state != CameraState::kRectified) {
    fail(field(name, "state") + " is not raw, undistorted or rectified");
  }
  check_positive(info.fx, name, "fx");
  check_positive(info.fy, name, "fy");
  check_finite(info.cx, name, "cx");
  check_finite(info.cy, name, "cy");
  check_finite(info.skew, name, "skew");
  if (coefficient_count(info.distortion_model) == 0 &&
      info.distortion_model != DistortionModel::kNone) {
    fail(field(name, "distortion_model") + " is not a known model");
  }
  check_distortion(info, name);
  if (info.rectification) {
    if (info.state == CameraState::kRectified) {
      fail(field(name, "rectification") +
           " must be absent: a rectified camera has no rotation left to apply");
    }
    check_rotation(*info.rectification, name, "rectification");
  }
  if (info.binning_x == 0 || info.binning_y == 0) {
    fail(field(name, "binning_x") + " and " + field(name, "binning_y") + " must be >= 1");
  }
  check_roi(info, name);
}

void validate(const StereoCameraInfo& info, std::string_view name) {
  check_frame_id(info.header, name);
  validate(info.left, field(name, "left"));
  validate(info.right, field(name, "right"));
  check_rotation(info.right_from_left.rotation, name, "right_from_left.rotation");
  const Vector3& t = info.right_from_left.translation;
  if (!(std::isfinite(t[0]) && std::isfinite(t[1]) && std::isfinite(t[2])) ||
      detail::norm(t) == 0.0) {
    fail(field(name, "right_from_left.translation") + " must be finite and not zero");
  }
  check_finite(info.baseline, name, "baseline");
  check_finite(info.doffs, name, "doffs");
  const bool left_rectified = info.left.state == CameraState::kRectified;
  const bool right_rectified = info.right.state == CameraState::kRectified;
  if (left_rectified != right_rectified) {
    fail(std::string(name) + ": one camera is rectified and the other " +
         std::string(camera_state_name(left_rectified ? info.right.state : info.left.state)) +
         "; a pair is rectified as a whole");
  }
  if (left_rectified) {
    check_rectified_pair(info, name);
    return;
  }
  if (info.baseline != 0.0 || info.doffs != 0.0) {
    fail(std::string(name) +
         ": baseline and doffs describe a rectified pair, and must be 0 for this one");
  }
  check_rectifications(info, name);
}

double fx_baseline(const StereoCameraInfo& info) noexcept {
  if (info.left.state != CameraState::kRectified) return 0.0;
  return info.left.fx * info.baseline;
}

double depth_from_disparity(const StereoCameraInfo& info, double disparity) noexcept {
  const double denominator = disparity + info.doffs;
  const double numerator = fx_baseline(info);
  if (!(denominator > 0) || numerator == 0.0) return std::numeric_limits<double>::quiet_NaN();
  return numerator / denominator;
}

bool same_calibration(const CameraInfo& a, const CameraInfo& b) noexcept {
  const auto calibration = [](const CameraInfo& c) {
    return std::tie(c.width, c.height, c.state, c.fx, c.fy, c.cx, c.cy, c.skew, c.distortion_model,
                    c.coefficients, c.rectification, c.roi, c.binning_x, c.binning_y);
  };
  return calibration(a) == calibration(b);
}

bool same_calibration(const StereoCameraInfo& a, const StereoCameraInfo& b) noexcept {
  return same_calibration(a.left, b.left) && same_calibration(a.right, b.right) &&
         a.right_from_left == b.right_from_left && a.baseline == b.baseline && a.doffs == b.doffs;
}

}  // namespace holoscan::messages
