// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stitching/camera_rig.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numbers>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "json.hpp"
#include "stitch_math.hpp"

namespace holoscan::stitching {
namespace {

using Matrix3 = std::array<double, 9>;

Matrix3 multiply(const Matrix3& a, const Matrix3& b) noexcept {
  Matrix3 c{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += a[3 * i + k] * b[3 * k + j];
      c[3 * i + j] = s;
    }
  }
  return c;
}

Matrix3 transpose(const Matrix3& a) noexcept {
  return {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]};
}

std::array<double, 3> rotate(const Matrix3& r, const std::array<double, 3>& v) noexcept {
  return {r[0] * v[0] + r[1] * v[1] + r[2] * v[2], r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
          r[6] * v[0] + r[7] * v[1] + r[8] * v[2]};
}

Matrix3 rotation_z(double angle) noexcept {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return {c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0};
}

Matrix3 rotation_y(double angle) noexcept {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return {c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c};
}

Matrix3 rotation_x(double angle) noexcept {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return {1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c};
}

constexpr double kDegree = std::numbers::pi / 180.0;

template <typename Values>
bool all_finite(const Values& values) {
  return std::all_of(std::begin(values), std::end(values),
                     [](double v) { return std::isfinite(v); });
}
// Steps of the search for where a lens' projection stops moving outwards.
constexpr double kAngleStep = 1e-3;

// The distorted radius (normalized) of a ray at `theta`, ignoring tangential
// distortion, for the search below.
double distorted_radius(const Camera& camera, double theta) noexcept {
  const auto k = [&camera](std::size_t i) { return coefficient(camera, i); };
  if (camera.distortion_model == DistortionModel::kKannalaBrandt4) {
    const double t2 = theta * theta;
    return theta * (1.0 + t2 * (k(0) + t2 * (k(1) + t2 * (k(2) + t2 * k(3)))));
  }
  const double r = std::tan(theta);
  const double r2 = r * r;
  double a = 1.0 + r2 * k(0) + r2 * r2 * k(1);
  double b = 1.0;
  if (camera.distortion_model != DistortionModel::kNone) {
    a += r2 * r2 * r2 * k(4);
  }
  if (camera.distortion_model == DistortionModel::kRationalPolynomial) {
    b += r2 * k(5) + r2 * r2 * k(6) + r2 * r2 * r2 * k(7);
  }
  return r * a / b;
}

// --- JSON ----------------------------------------------------------------------

constexpr std::uintmax_t kMaxFileBytes = 1U << 20;

std::string read_text(const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error) {
    throw std::invalid_argument(path.string() + ": cannot read: " + error.message());
  }
  if (size > kMaxFileBytes) {
    throw std::invalid_argument(path.string() + ": larger than a rig file can be");
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::invalid_argument(path.string() + ": cannot open");
  std::ostringstream text;
  text << file.rdbuf();
  return text.str();
}

double number(const json::Value& parent, std::string_view key, const std::string& where) {
  const json::Value* value = parent.find(key);
  if (value == nullptr || value->number() == nullptr) {
    throw std::invalid_argument(where + "." + std::string(key) + " must be a number");
  }
  return *value->number();
}

int integer(const json::Value& parent, std::string_view key, const std::string& where) {
  const double value = number(parent, key, where);
  if (value != std::floor(value) || value < 1 || value > kMaxImageSide) {
    throw std::invalid_argument(where + "." + std::string(key) + " must be a whole number 1.." +
                                std::to_string(kMaxImageSide));
  }
  return static_cast<int>(value);
}

std::vector<double> numbers(const json::Value& parent, std::string_view key,
                            const std::string& where) {
  const json::Value* value = parent.find(key);
  const json::Array* array = value != nullptr ? value->array() : nullptr;
  if (array == nullptr) {
    throw std::invalid_argument(where + "." + std::string(key) + " must be an array of numbers");
  }
  std::vector<double> result;
  for (const json::Value& item : *array) {
    if (item.number() == nullptr) {
      throw std::invalid_argument(where + "." + std::string(key) + " must hold numbers only");
    }
    result.push_back(*item.number());
  }
  return result;
}

RigidTransform transform_from_json(const json::Value& node, const std::string& where) {
  if (node.object() == nullptr) {
    throw std::invalid_argument(where + " must be an object");
  }
  const std::vector<double> rotation = numbers(node, "rotation", where);
  const std::vector<double> translation = numbers(node, "translation", where);
  if (rotation.size() != 9 || translation.size() != 3) {
    throw std::invalid_argument(where + " needs a rotation of 9 numbers and a translation of 3");
  }
  RigidTransform transform;
  std::copy(rotation.begin(), rotation.end(), transform.rotation.begin());
  std::copy(translation.begin(), translation.end(), transform.translation.begin());
  return transform;
}

// A camera object of a rig file or a recording, without its pose.
Camera camera_from_json(const json::Value& node, const std::string& where) {
  if (node.object() == nullptr) {
    throw std::invalid_argument(where + " must be an object");
  }
  Camera camera;
  camera.width = integer(node, "width", where);
  camera.height = integer(node, "height", where);
  camera.fx = number(node, "fx", where);
  camera.fy = number(node, "fy", where);
  camera.cx = number(node, "cx", where);
  camera.cy = number(node, "cy", where);
  const json::Value* distortion = node.find("distortion");
  if (distortion != nullptr) {
    if (distortion->object() == nullptr) {
      throw std::invalid_argument(where + ".distortion must be an object");
    }
    const json::Value* model = distortion->find("model");
    if (model == nullptr || model->string() == nullptr) {
      throw std::invalid_argument(where + ".distortion.model must be a string");
    }
    const auto parsed = parse_distortion_model(*model->string());
    if (!parsed) {
      throw std::invalid_argument(where + ".distortion.model '" + *model->string() +
                                  "' is not supported; use none, brown_conrady, "
                                  "rational_polynomial or kannala_brandt4");
    }
    camera.distortion_model = *parsed;
    camera.distortion = numbers(*distortion, "coefficients", where + ".distortion");
    // Four Brown-Conrady coefficients are k1, k2, p1, p2, with k3 = 0.
    if (camera.distortion_model == DistortionModel::kBrownConrady &&
        camera.distortion.size() == 4) {
      camera.distortion.push_back(0.0);
    }
    if (std::all_of(camera.distortion.begin(), camera.distortion.end(),
                    [](double v) { return v == 0.0; })) {
      camera.distortion.clear();
    }
  }
  return camera;
}

CameraRig rig_from_recording(const json::Value& root) {
  CameraRig rig;
  Camera left = camera_from_json(*root.find("left"), "left");
  left.name = "left";
  rig.cameras.push_back(std::move(left));
  if (const json::Value* node = root.find("right"); node != nullptr) {
    Camera right = camera_from_json(*node, "right");
    right.name = "right";
    const json::Value* pose = root.find("right_from_left");
    if (pose == nullptr) {
      throw std::invalid_argument("right_from_left must be an object");
    }
    right.camera_from_rig = transform_from_json(*pose, "right_from_left");
    rig.cameras.push_back(std::move(right));
  }
  if (const json::Value* node = root.find("rgb"); node != nullptr) {
    Camera rgb = camera_from_json(*node, "rgb");
    rgb.name = "rgb";
    const json::Value* pose = node->find("from_left");
    if (pose == nullptr) {
      throw std::invalid_argument("rgb.from_left must be an object");
    }
    rgb.camera_from_rig = transform_from_json(*pose, "rgb.from_left");
    rig.cameras.push_back(std::move(rgb));
  }
  return rig;
}

std::string number_text(double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  return buffer;
}

std::string array_text(const double* values, std::size_t count) {
  std::string text = "[";
  for (std::size_t i = 0; i < count; ++i) {
    if (i != 0) text += ", ";
    text += number_text(values[i]);
  }
  return text + "]";
}

std::string quoted(const std::string& text) {
  std::string result = "\"";
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      result += '\\';
      result += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      result += '?';
    } else {
      result += c;
    }
  }
  return result + "\"";
}

}  // namespace

std::string_view distortion_model_name(DistortionModel model) noexcept {
  switch (model) {
    case DistortionModel::kNone:
      return "none";
    case DistortionModel::kBrownConrady:
      return "brown_conrady";
    case DistortionModel::kRationalPolynomial:
      return "rational_polynomial";
    case DistortionModel::kKannalaBrandt4:
      return "kannala_brandt4";
  }
  return "unknown";
}

std::optional<DistortionModel> parse_distortion_model(std::string_view name) noexcept {
  if (name == "none") return DistortionModel::kNone;
  if (name == "brown_conrady" || name == "plumb_bob") return DistortionModel::kBrownConrady;
  if (name == "rational_polynomial") return DistortionModel::kRationalPolynomial;
  if (name == "kannala_brandt4" || name == "equidistant" || name == "fisheye") {
    return DistortionModel::kKannalaBrandt4;
  }
  return std::nullopt;
}

std::size_t distortion_coefficient_count(DistortionModel model) noexcept {
  switch (model) {
    case DistortionModel::kNone:
      return 0;
    case DistortionModel::kBrownConrady:
      return 5;
    case DistortionModel::kRationalPolynomial:
      return 8;
    case DistortionModel::kKannalaBrandt4:
      return 4;
  }
  return 0;
}

RigidTransform inverse(const RigidTransform& b_from_a) noexcept {
  RigidTransform a_from_b;
  a_from_b.rotation = transpose(b_from_a.rotation);
  const std::array<double, 3> t = rotate(a_from_b.rotation, b_from_a.translation);
  a_from_b.translation = {-t[0], -t[1], -t[2]};
  return a_from_b;
}

RigidTransform compose(const RigidTransform& a_from_b, const RigidTransform& b_from_c) noexcept {
  RigidTransform a_from_c;
  a_from_c.rotation = multiply(a_from_b.rotation, b_from_c.rotation);
  const std::array<double, 3> t = rotate(a_from_b.rotation, b_from_c.translation);
  for (int i = 0; i < 3; ++i) a_from_c.translation[i] = t[i] + a_from_b.translation[i];
  return a_from_c;
}

RigidTransform mount(const std::array<double, 3>& position, double yaw_deg, double pitch_deg,
                     double roll_deg) noexcept {
  // The optical axes of an upright camera looking along +x: x right (-y),
  // y down (-z), z forward (+x), as the columns of rig_from_camera.
  constexpr Matrix3 kOptical{0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0};
  const Matrix3 body =
      multiply(multiply(rotation_z(yaw_deg * kDegree), rotation_y(-pitch_deg * kDegree)),
               rotation_x(roll_deg * kDegree));
  RigidTransform rig_from_camera;
  rig_from_camera.rotation = multiply(body, kOptical);
  rig_from_camera.translation = position;
  return inverse(rig_from_camera);
}

void validate(const Camera& camera, std::string_view what) {
  const std::string where(what);
  if (camera.width < 1 || camera.height < 1 || camera.width > kMaxImageSide ||
      camera.height > kMaxImageSide) {
    throw std::invalid_argument(where + ": width and height must be 1.." +
                                std::to_string(kMaxImageSide));
  }
  if (!(std::isfinite(camera.fx) && camera.fx > 0 && std::isfinite(camera.fy) && camera.fy > 0)) {
    throw std::invalid_argument(where + ": fx and fy must be finite and positive");
  }
  if (!(std::isfinite(camera.cx) && std::isfinite(camera.cy))) {
    throw std::invalid_argument(where + ": cx and cy must be finite");
  }
  const std::size_t count = distortion_coefficient_count(camera.distortion_model);
  if (!camera.distortion.empty() && camera.distortion.size() != count) {
    throw std::invalid_argument(where + ": the " +
                                std::string(distortion_model_name(camera.distortion_model)) +
                                " model takes " + std::to_string(count) + " coefficients, not " +
                                std::to_string(camera.distortion.size()));
  }
  if (!all_finite(camera.distortion)) {
    throw std::invalid_argument(where + ": distortion coefficients must be finite");
  }
  const Matrix3& r = camera.camera_from_rig.rotation;
  if (!all_finite(r) || !all_finite(camera.camera_from_rig.translation)) {
    throw std::invalid_argument(where + ": camera_from_rig must be finite");
  }
  const Matrix3 should_be_identity = multiply(r, transpose(r));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (std::abs(should_be_identity[3 * i + j] - (i == j ? 1.0 : 0.0)) > 1e-6) {
        throw std::invalid_argument(where + ": camera_from_rig's rotation is not orthonormal");
      }
    }
  }
  const double det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) +
                     r[2] * (r[3] * r[7] - r[4] * r[6]);
  if (std::abs(det - 1.0) > 1e-6) {
    throw std::invalid_argument(where + ": camera_from_rig's rotation is a reflection");
  }
  if (!(max_valid_angle(camera) > 0.01)) {
    throw std::invalid_argument(where + ": the distortion folds back at the optical axis");
  }
}

void validate(const CameraRig& rig) {
  if (rig.cameras.empty() || rig.cameras.size() > kMaxCameras) {
    throw std::invalid_argument("a rig has 1.." + std::to_string(kMaxCameras) + " cameras, not " +
                                std::to_string(rig.cameras.size()));
  }
  std::set<std::string> names;
  for (std::size_t i = 0; i < rig.cameras.size(); ++i) {
    const Camera& camera = rig.cameras[i];
    const std::string what = "camera " + std::to_string(i) +
                             (camera.name.empty() ? std::string() : " (" + camera.name + ")");
    if (camera.name.empty()) {
      throw std::invalid_argument(what + ": name must not be empty");
    }
    if (!names.insert(camera.name).second) {
      throw std::invalid_argument(what + ": two cameras are named " + camera.name);
    }
    validate(camera, what);
  }
}

double coefficient(const Camera& camera, std::size_t index) noexcept {
  return index < camera.distortion.size() ? camera.distortion[index] : 0.0;
}

double max_valid_angle(const Camera& camera) noexcept {
  const bool fisheye = camera.distortion_model == DistortionModel::kKannalaBrandt4;
  // A pinhole sees less than 90 degrees from its axis; a fisheye less than 180.
  const double limit = (fisheye ? std::numbers::pi : std::numbers::pi / 2.0) - 1e-3;
  double previous = 0.0;
  for (double theta = kAngleStep; theta <= limit; theta += kAngleStep) {
    const double radius = distorted_radius(camera, theta);
    if (!(radius > previous)) {
      return theta - kAngleStep;
    }
    previous = radius;
  }
  return limit;
}

Point2 distort(const Camera& camera, Point2 ideal) noexcept {
  const detail::CameraParams params = detail::camera_params(camera);
  double xd = 0.0;
  double yd = 0.0;
  if (camera.distortion_model == DistortionModel::kKannalaBrandt4) {
    const double r = std::hypot(ideal.x, ideal.y);
    const double theta = std::atan(r);
    detail::distort_ray(params, theta, r > 0 ? ideal.x / r : 0.0, r > 0 ? ideal.y / r : 0.0, 0.0,
                        0.0, xd, yd);
  } else {
    detail::distort_ray(params, 0.0, 0.0, 0.0, ideal.x, ideal.y, xd, yd);
  }
  return {xd, yd};
}

std::optional<Point2> project_to_pixel(const Camera& camera,
                                       const std::array<double, 3>& point) noexcept {
  // Through the camera's own frame: an identity pose.
  detail::CameraParams params = detail::camera_params(camera);
  constexpr double kIdentity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::copy(std::begin(kIdentity), std::end(kIdentity), params.rotation);
  std::fill(std::begin(params.translation), std::end(params.translation), 0.0);
  // Only the lens counts here, not the image's border.
  params.width = kMaxImageSide * 4;
  params.height = kMaxImageSide * 4;
  params.cx = camera.cx + kMaxImageSide * 2;
  params.cy = camera.cy + kMaxImageSide * 2;
  const double p[3] = {point[0], point[1], point[2]};
  double u = 0.0;
  double v = 0.0;
  double margin = 0.0;
  if (!detail::project(params, p, false, u, v, margin)) {
    return std::nullopt;
  }
  return Point2{u - kMaxImageSide * 2, v - kMaxImageSide * 2};
}

std::optional<std::array<double, 3>> pixel_ray(const Camera& camera, Point2 pixel) noexcept {
  const detail::CameraParams params = detail::camera_params(camera);
  double ray[3];
  if (!detail::pixel_ray(params, pixel.x, pixel.y, ray)) {
    return std::nullopt;
  }
  return std::array<double, 3>{ray[0], ray[1], ray[2]};
}

CameraRig parse_rig(std::string_view text) {
  const json::Value root = json::parse(text);
  if (root.object() == nullptr) {
    throw std::invalid_argument("rig: the document must be an object");
  }
  if (const json::Value* version = root.find("version");
      version != nullptr && (version->number() == nullptr || *version->number() != 1.0)) {
    throw std::invalid_argument("rig: version must be 1");
  }
  CameraRig rig;
  const json::Value* cameras = root.find("cameras");
  if (cameras == nullptr && root.find("left") != nullptr) {
    rig = rig_from_recording(root);
  } else {
    const json::Array* list = cameras != nullptr ? cameras->array() : nullptr;
    if (list == nullptr) {
      throw std::invalid_argument("rig: cameras must be an array");
    }
    for (std::size_t i = 0; i < list->size(); ++i) {
      const json::Value& node = (*list)[i];
      const std::string where = "cameras[" + std::to_string(i) + "]";
      Camera camera = camera_from_json(node, where);
      const json::Value* name = node.find("name");
      if (name == nullptr || name->string() == nullptr) {
        throw std::invalid_argument(where + ".name must be a string");
      }
      camera.name = *name->string();
      const json::Value* camera_from_rig = node.find("camera_from_rig");
      const json::Value* rig_from_camera = node.find("rig_from_camera");
      if ((camera_from_rig == nullptr) == (rig_from_camera == nullptr)) {
        throw std::invalid_argument(where + " needs camera_from_rig or rig_from_camera, not both");
      }
      camera.camera_from_rig =
          camera_from_rig != nullptr
              ? transform_from_json(*camera_from_rig, where + ".camera_from_rig")
              : inverse(transform_from_json(*rig_from_camera, where + ".rig_from_camera"));
      rig.cameras.push_back(std::move(camera));
    }
  }
  validate(rig);
  return rig;
}

CameraRig read_rig(const std::filesystem::path& path) {
  const std::string text = read_text(path);
  try {
    return parse_rig(text);
  } catch (const std::invalid_argument& error) {
    throw std::invalid_argument(path.string() + ": " + error.what());
  }
}

std::string to_json(const CameraRig& rig) {
  std::string text = "{\n  \"version\": 1,\n  \"cameras\": [";
  for (std::size_t i = 0; i < rig.cameras.size(); ++i) {
    const Camera& c = rig.cameras[i];
    std::vector<double> coefficients = c.distortion;
    coefficients.resize(distortion_coefficient_count(c.distortion_model), 0.0);
    text += i == 0 ? "\n" : ",\n";
    text += "    {\n      \"name\": " + quoted(c.name) + ",\n";
    text += "      \"width\": " + std::to_string(c.width) +
            ",\n      \"height\": " + std::to_string(c.height) + ",\n";
    text += "      \"fx\": " + number_text(c.fx) + ",\n      \"fy\": " + number_text(c.fy) +
            ",\n      \"cx\": " + number_text(c.cx) + ",\n      \"cy\": " + number_text(c.cy) +
            ",\n";
    text += "      \"distortion\": {\"model\": \"" +
            std::string(distortion_model_name(c.distortion_model)) +
            "\", \"coefficients\": " + array_text(coefficients.data(), coefficients.size()) +
            "},\n";
    text += "      \"camera_from_rig\": {\"rotation\": " +
            array_text(c.camera_from_rig.rotation.data(), 9) +
            ", \"translation\": " + array_text(c.camera_from_rig.translation.data(), 3) +
            "}\n    }";
  }
  return text + "\n  ]\n}\n";
}

namespace detail {

CameraParams camera_params(const Camera& camera) {
  CameraParams p{};
  p.width = camera.width;
  p.height = camera.height;
  p.fx = camera.fx;
  p.fy = camera.fy;
  p.cx = camera.cx;
  p.cy = camera.cy;
  switch (camera.distortion_model) {
    case DistortionModel::kNone:
      p.model = kModelNone;
      break;
    case DistortionModel::kBrownConrady:
      p.model = kModelBrownConrady;
      break;
    case DistortionModel::kRationalPolynomial:
      p.model = kModelRational;
      break;
    case DistortionModel::kKannalaBrandt4:
      p.model = kModelKannalaBrandt4;
      break;
  }
  for (std::size_t i = 0; i < 8; ++i) p.k[i] = coefficient(camera, i);
  std::copy(camera.camera_from_rig.rotation.begin(), camera.camera_from_rig.rotation.end(),
            p.rotation);
  std::copy(camera.camera_from_rig.translation.begin(), camera.camera_from_rig.translation.end(),
            p.translation);
  p.max_angle = max_valid_angle(camera);
  return p;
}

}  // namespace detail
}  // namespace holoscan::stitching
