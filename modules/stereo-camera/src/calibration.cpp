// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_camera/calibration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "json.hpp"

namespace holoscan::stereo_camera {
namespace {

constexpr std::array<std::pair<DistortionModel, std::string_view>, 7> kModelNames{{
    {DistortionModel::kNone, "none"},
    {DistortionModel::kBrownConrady, "brown_conrady"},
    {DistortionModel::kModifiedBrownConrady, "modified_brown_conrady"},
    {DistortionModel::kInverseBrownConrady, "inverse_brown_conrady"},
    {DistortionModel::kKannalaBrandt4, "kannala_brandt4"},
    {DistortionModel::kFTheta, "ftheta"},
    {DistortionModel::kRationalPolynomial, "rational_polynomial"},
}};

// Largest image side a calibration may describe.
constexpr int kMaxSide = 16384;
// How far a rotation may be from orthonormal.
constexpr double kRotationTolerance = 1e-6;

[[noreturn]] void bad(const std::string& field, const std::string& problem) {
  throw std::invalid_argument(field + ": " + problem);
}

void validate_camera(const CameraModel& camera, const std::string& field) {
  if (camera.width < 1 || camera.height < 1 || camera.width > kMaxSide ||
      camera.height > kMaxSide) {
    bad(field, "width and height must be 1.." + std::to_string(kMaxSide));
  }
  if (!(std::isfinite(camera.fx) && camera.fx > 0 && std::isfinite(camera.fy) && camera.fy > 0)) {
    bad(field, "fx and fy must be finite and positive");
  }
  if (!(std::isfinite(camera.cx) && std::isfinite(camera.cy))) {
    bad(field, "cx and cy must be finite");
  }
  const std::size_t count = distortion_coefficient_count(camera.distortion_model);
  if (!camera.distortion.empty() && camera.distortion.size() != count) {
    bad(field, "distortion model " + std::string(distortion_model_name(camera.distortion_model)) +
                   " takes " + std::to_string(count) + " coefficients, not " +
                   std::to_string(camera.distortion.size()));
  }
  for (const double k : camera.distortion) {
    if (!std::isfinite(k)) {
      bad(field, "distortion coefficients must be finite");
    }
  }
}

void validate_transform(const RigidTransform& transform, const std::string& field) {
  for (const double value : transform.rotation) {
    if (!std::isfinite(value)) bad(field, "rotation must be finite");
  }
  for (const double value : transform.translation) {
    if (!std::isfinite(value)) bad(field, "translation must be finite");
  }
  const auto& r = transform.rotation;
  // R * R^T must be the identity, and det(R) one.
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double dot = 0.0;
      for (int k = 0; k < 3; ++k) dot += r[i * 3 + k] * r[j * 3 + k];
      if (std::abs(dot - (i == j ? 1.0 : 0.0)) > kRotationTolerance) {
        bad(field, "rotation is not orthonormal");
      }
    }
  }
  const double det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) +
                     r[2] * (r[3] * r[7] - r[4] * r[6]);
  if (std::abs(det - 1.0) > kRotationTolerance) {
    bad(field, "rotation must have determinant 1, not a reflection");
  }
}

// ---- Reading --------------------------------------------------------------------

const json::Value& member(const json::Value& object, std::string_view key,
                          const std::string& where) {
  const json::Value* value = object.find(key);
  if (value == nullptr) {
    bad(where, "missing \"" + std::string(key) + "\"");
  }
  return *value;
}

std::string path_of(const std::string& where, std::string_view key) {
  return where.empty() ? std::string(key) : where + "." + std::string(key);
}

const json::Value& typed(const json::Value& value, json::Value::Kind kind,
                         const std::string& where) {
  if (value.kind != kind) {
    bad(where,
        std::string("expected ") + json::kind_name(kind) + ", not " + json::kind_name(value.kind));
  }
  return value;
}

double number_of(const json::Value& object, std::string_view key, const std::string& where) {
  return typed(member(object, key, where), json::Value::Kind::kNumber, path_of(where, key)).number;
}

int integer_of(const json::Value& object, std::string_view key, const std::string& where) {
  const double value = number_of(object, key, where);
  if (value != std::floor(value) || std::abs(value) > 1e9) {
    bad(path_of(where, key), "expected a whole number");
  }
  return static_cast<int>(value);
}

std::vector<double> numbers_of(const json::Value& value, const std::string& where) {
  std::vector<double> result;
  for (std::size_t i = 0; i < typed(value, json::Value::Kind::kArray, where).items.size(); ++i) {
    result.push_back(
        typed(value.items[i], json::Value::Kind::kNumber, where + "[" + std::to_string(i) + "]")
            .number);
  }
  return result;
}

std::string string_of(const json::Value& object, std::string_view key, const std::string& where) {
  const json::Value* value = object.find(key);
  if (value == nullptr) {
    return {};
  }
  return typed(*value, json::Value::Kind::kString, path_of(where, key)).string;
}

CameraModel read_camera(const json::Value& value, const std::string& where) {
  typed(value, json::Value::Kind::kObject, where);
  CameraModel camera;
  camera.width = integer_of(value, "width", where);
  camera.height = integer_of(value, "height", where);
  camera.fx = number_of(value, "fx", where);
  camera.fy = number_of(value, "fy", where);
  camera.cx = number_of(value, "cx", where);
  camera.cy = number_of(value, "cy", where);
  if (const json::Value* distortion = value.find("distortion")) {
    const std::string at = path_of(where, "distortion");
    typed(*distortion, json::Value::Kind::kObject, at);
    const std::string name = string_of(*distortion, "model", at);
    const auto model = parse_distortion_model(name);
    if (!model) {
      bad(path_of(at, "model"), "unknown distortion model \"" + name + "\"");
    }
    camera.distortion_model = *model;
    if (const json::Value* coefficients = distortion->find("coefficients")) {
      camera.distortion = numbers_of(*coefficients, path_of(at, "coefficients"));
    }
  }
  return camera;
}

RigidTransform read_transform(const json::Value& value, const std::string& where) {
  typed(value, json::Value::Kind::kObject, where);
  RigidTransform transform;
  const std::vector<double> rotation =
      numbers_of(member(value, "rotation", where), path_of(where, "rotation"));
  const std::vector<double> translation =
      numbers_of(member(value, "translation", where), path_of(where, "translation"));
  if (rotation.size() != 9) {
    bad(path_of(where, "rotation"), "expected 9 numbers, row-major");
  }
  if (translation.size() != 3) {
    bad(path_of(where, "translation"), "expected 3 numbers, in metres");
  }
  std::copy(rotation.begin(), rotation.end(), transform.rotation.begin());
  std::copy(translation.begin(), translation.end(), transform.translation.begin());
  return transform;
}

// ---- Writing --------------------------------------------------------------------

std::string numbers(const double* values, std::size_t count) {
  std::string out = "[";
  for (std::size_t i = 0; i < count; ++i) {
    out += (i == 0 ? "" : ", ") + json::number(values[i]);
  }
  return out + "]";
}

std::string camera_json(const CameraModel& camera, const std::string& indent) {
  std::string out = "{\n";
  const std::string in = indent + "  ";
  out += in + "\"width\": " + std::to_string(camera.width) + ",\n";
  out += in + "\"height\": " + std::to_string(camera.height) + ",\n";
  out += in + "\"fx\": " + json::number(camera.fx) + ",\n";
  out += in + "\"fy\": " + json::number(camera.fy) + ",\n";
  out += in + "\"cx\": " + json::number(camera.cx) + ",\n";
  out += in + "\"cy\": " + json::number(camera.cy) + ",\n";
  out += in + "\"distortion\": {\n";
  out += in + "  \"model\": " + json::quote(distortion_model_name(camera.distortion_model)) + ",\n";
  out += in + "  \"coefficients\": " + numbers(camera.distortion.data(), camera.distortion.size()) +
         "\n";
  out += in + "}\n";
  return out + indent + "}";
}

std::string transform_json(const RigidTransform& transform, const std::string& indent) {
  return "{\n" + indent + "  \"rotation\": " + numbers(transform.rotation.data(), 9) + ",\n" +
         indent + "  \"translation\": " + numbers(transform.translation.data(), 3) + "\n" + indent +
         "}";
}

}  // namespace

std::string_view distortion_model_name(DistortionModel model) noexcept {
  for (const auto& [kind, name] : kModelNames) {
    if (kind == model) return name;
  }
  return "unknown";
}

std::optional<DistortionModel> parse_distortion_model(std::string_view name) noexcept {
  for (const auto& [kind, model_name] : kModelNames) {
    if (model_name == name) return kind;
  }
  return std::nullopt;
}

std::size_t distortion_coefficient_count(DistortionModel model) noexcept {
  switch (model) {
    case DistortionModel::kNone:
      return 0;
    case DistortionModel::kBrownConrady:
    case DistortionModel::kModifiedBrownConrady:
    case DistortionModel::kInverseBrownConrady:
      return 5;
    case DistortionModel::kKannalaBrandt4:
      return 4;
    case DistortionModel::kFTheta:
      return 1;
    case DistortionModel::kRationalPolynomial:
      return 8;
  }
  return 0;
}

bool CameraModel::undistorted() const noexcept {
  for (const double k : distortion) {
    if (k != 0.0) return false;
  }
  return true;
}

double StereoCalibration::baseline() const noexcept {
  const auto& t = right_from_left.translation;
  return std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
}

double StereoCalibration::depth_from_disparity() const noexcept { return left.fx * baseline(); }

void validate(const StereoCalibration& calibration) {
  validate_camera(calibration.left, "left");
  validate_camera(calibration.right, "right");
  if (calibration.left.width != calibration.right.width ||
      calibration.left.height != calibration.right.height) {
    bad("right", "the eyes must have the same size");
  }
  validate_transform(calibration.right_from_left, "right_from_left");
  if (!(calibration.baseline() > 0.0)) {
    bad("right_from_left", "the baseline must not be zero");
  }
  if (calibration.rectified &&
      !(calibration.left.undistorted() && calibration.right.undistorted())) {
    bad("rectified", "a rectified pair has no distortion");
  }
  if (calibration.rgb) {
    validate_camera(calibration.rgb->model, "rgb");
    validate_transform(calibration.rgb->from_left, "rgb.from_left");
  }
  if (calibration.depth) {
    validate_camera(calibration.depth->model, "depth");
    validate_transform(calibration.depth->from_left, "depth.from_left");
    if (!(std::isfinite(calibration.depth->scale) && calibration.depth->scale > 0.0)) {
      bad("depth.scale", "must be finite and positive");
    }
  }
}

StereoCalibration parse_calibration(std::string_view text) {
  const json::Value root = json::parse(text);
  typed(root, json::Value::Kind::kObject, "calibration");
  const int version = integer_of(root, "version", "");
  if (version != kCalibrationFormatVersion) {
    bad("version", "this library reads version " + std::to_string(kCalibrationFormatVersion) +
                       ", not " + std::to_string(version));
  }
  StereoCalibration calibration;
  calibration.left = read_camera(member(root, "left", ""), "left");
  calibration.right = read_camera(member(root, "right", ""), "right");
  calibration.right_from_left =
      read_transform(member(root, "right_from_left", ""), "right_from_left");
  calibration.rectified =
      typed(member(root, "rectified", ""), json::Value::Kind::kBool, "rectified").boolean;
  if (const json::Value* rgb = root.find("rgb"); rgb != nullptr) {
    typed(*rgb, json::Value::Kind::kObject, "rgb");
    calibration.rgb = AuxiliaryCamera{
        .model = read_camera(*rgb, "rgb"),
        .from_left = read_transform(member(*rgb, "from_left", "rgb"), "rgb.from_left")};
  }
  if (const json::Value* depth = root.find("depth"); depth != nullptr) {
    typed(*depth, json::Value::Kind::kObject, "depth");
    calibration.depth = DepthCamera{
        .model = read_camera(*depth, "depth"),
        .from_left = read_transform(member(*depth, "from_left", "depth"), "depth.from_left"),
        .scale = number_of(*depth, "scale", "depth")};
  }
  if (const json::Value* camera = root.find("camera"); camera != nullptr) {
    typed(*camera, json::Value::Kind::kObject, "camera");
    calibration.camera.name = string_of(*camera, "name", "camera");
    calibration.camera.serial = string_of(*camera, "serial", "camera");
    calibration.camera.firmware = string_of(*camera, "firmware", "camera");
    if (const json::Value* settings = camera->find("settings"); settings != nullptr) {
      typed(*settings, json::Value::Kind::kObject, "camera.settings");
      for (std::size_t i = 0; i < settings->keys.size(); ++i) {
        calibration.camera.settings[settings->keys[i]] =
            typed(settings->items[i], json::Value::Kind::kString,
                  "camera.settings." + settings->keys[i])
                .string;
      }
    }
  }
  validate(calibration);
  return calibration;
}

std::string format_calibration(const StereoCalibration& calibration) {
  validate(calibration);
  std::string out = "{\n";
  out += "  \"version\": " + std::to_string(kCalibrationFormatVersion) + ",\n";
  out += "  \"camera\": {\n";
  out += "    \"name\": " + json::quote(calibration.camera.name) + ",\n";
  out += "    \"serial\": " + json::quote(calibration.camera.serial) + ",\n";
  out += "    \"firmware\": " + json::quote(calibration.camera.firmware) + ",\n";
  out += "    \"settings\": {";
  bool first = true;
  for (const auto& [key, value] : calibration.camera.settings) {
    out +=
        std::string(first ? "\n" : ",\n") + "      " + json::quote(key) + ": " + json::quote(value);
    first = false;
  }
  out += first ? "}\n" : "\n    }\n";
  out += "  },\n";
  out += "  \"rectified\": " + std::string(calibration.rectified ? "true" : "false") + ",\n";
  out += "  \"left\": " + camera_json(calibration.left, "  ") + ",\n";
  out += "  \"right\": " + camera_json(calibration.right, "  ") + ",\n";
  out += "  \"right_from_left\": " + transform_json(calibration.right_from_left, "  ");
  // An auxiliary camera is its model's members plus its own.
  const auto auxiliary = [](const CameraModel& model, const RigidTransform& from_left,
                            const std::string& extra) {
    std::string body = camera_json(model, "  ");
    body.erase(body.size() - 4);  // "\n  }"
    return body + ",\n    \"from_left\": " + transform_json(from_left, "    ") + extra + "\n  }";
  };
  if (calibration.rgb) {
    out += ",\n  \"rgb\": " + auxiliary(calibration.rgb->model, calibration.rgb->from_left, "");
  }
  if (calibration.depth) {
    out += ",\n  \"depth\": " +
           auxiliary(calibration.depth->model, calibration.depth->from_left,
                     ",\n    \"scale\": " + json::number(calibration.depth->scale));
  }
  return out + "\n}\n";
}

StereoCalibration read_calibration(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::invalid_argument(path.string() + ": cannot open");
  }
  const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (file.bad()) {
    throw std::invalid_argument(path.string() + ": cannot read");
  }
  try {
    return parse_calibration(text);
  } catch (const std::invalid_argument& error) {
    throw std::invalid_argument(path.string() + ": " + error.what());
  }
}

void write_calibration(const std::filesystem::path& path, const StereoCalibration& calibration) {
  const std::string text = format_calibration(calibration);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file << text;
  file.close();
  if (!file) {
    throw std::runtime_error(path.string() + ": cannot write");
  }
}

}  // namespace holoscan::stereo_camera
