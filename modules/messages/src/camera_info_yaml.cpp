// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/messages/camera_info_yaml.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "geometry.hpp"
#include "yaml_reader.hpp"

namespace holoscan::messages {
namespace {

using detail::YamlNode;
using Projection = std::array<double, 12>;

// K and P entries of a rectified pair's files are compared to this, relative.
constexpr double kMatrixTolerance = 1e-9;

constexpr std::string_view kKeys[] = {
    "image_width",          "image_height",      "camera_name",
    "camera_matrix",        "distortion_model",  "distortion_coefficients",
    "rectification_matrix", "projection_matrix",
};

[[noreturn]] void fail(const std::string& what) { throw std::invalid_argument(what); }

// --- reading ---------------------------------------------------------------

std::string read_text(const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error) fail(path.string() + ": cannot read: " + error.message());
  if (size > kMaxCameraInfoFileBytes) {
    fail(path.string() + ": " + std::to_string(size) + " bytes, more than a camera_info file has");
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) fail(path.string() + ": cannot open");
  std::ostringstream text;
  text << file.rdbuf();
  if (!file && !file.eof()) fail(path.string() + ": cannot read");
  return text.str();
}

double to_double(const YamlNode& node, const std::string& what) {
  if (node.kind != YamlNode::Kind::kScalar || node.quoted) {
    fail(what + " (line " + std::to_string(node.line) + ") must be a number");
  }
  std::string_view text = node.scalar;
  if (text.starts_with('+')) text.remove_prefix(1);
  double value = 0.0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size() || !std::isfinite(value)) {
    fail(what + " (line " + std::to_string(node.line) + ") holds '" + node.scalar +
         "', not a finite number");
  }
  return value;
}

std::uint64_t to_count(const YamlNode& node, const std::string& what) {
  const double value = to_double(node, what);
  if (value < 0 || value != std::floor(value) ||
      value > std::numeric_limits<std::uint32_t>::max()) {
    fail(what + " (line " + std::to_string(node.line) + ") must be a whole number >= 0, not " +
         node.scalar);
  }
  return static_cast<std::uint64_t>(value);
}

const YamlNode& required(const YamlNode& root, std::string_view key) {
  const YamlNode* node = root.find(key);
  if (node == nullptr) fail(std::string(key) + " is missing");
  return *node;
}

std::uint32_t image_size(const YamlNode& root, std::string_view key) {
  const std::uint64_t value = to_count(required(root, key), std::string(key));
  if (value == 0) fail(std::string(key) + " must be > 0");
  return static_cast<std::uint32_t>(value);
}

// A {rows, cols, data} matrix: `rows` rows and `cols` columns, or any number
// of columns when `cols` is 0.
std::vector<double> matrix(const YamlNode& node, std::string_view key, std::uint64_t rows,
                           std::uint64_t cols) {
  const std::string name(key);
  if (node.kind != YamlNode::Kind::kMapping) {
    fail(name + " (line " + std::to_string(node.line) +
         ") must be a mapping of rows, cols and data");
  }
  for (const auto& [entry, value] : node.entries) {
    if (entry != "rows" && entry != "cols" && entry != "data") {
      fail(name + "." + entry + " (line " + std::to_string(value.line) +
           ") is not a matrix key; a matrix has rows, cols and data");
    }
  }
  const std::uint64_t got_rows = to_count(required(node, "rows"), name + ".rows");
  const std::uint64_t got_cols = to_count(required(node, "cols"), name + ".cols");
  const YamlNode& data = required(node, "data");
  if (data.kind != YamlNode::Kind::kSequence) {
    fail(name + ".data (line " + std::to_string(data.line) + ") must be a list of numbers");
  }
  const std::string shape =
      std::to_string(rows) + "x" + (cols == 0 ? std::string("N") : std::to_string(cols));
  if (got_rows != rows || (cols != 0 && got_cols != cols)) {
    fail(name + " must be " + shape + ", not " + std::to_string(got_rows) + "x" +
         std::to_string(got_cols));
  }
  if (data.items.size() != got_rows * got_cols) {
    fail(name + " is " + std::to_string(got_rows) + "x" + std::to_string(got_cols) +
         " but its data has " + std::to_string(data.items.size()) + " values");
  }
  std::vector<double> values;
  values.reserve(data.items.size());
  for (std::size_t i = 0; i < data.items.size(); ++i) {
    values.push_back(to_double(data.items[i], name + ".data[" + std::to_string(i) + "]"));
  }
  return values;
}

// One file's camera, and its projection matrix if it has one.
struct ParsedCamera {
  CameraInfo info;
  std::optional<Projection> projection;
};

void check_keys(const YamlNode& root) {
  if (root.kind != YamlNode::Kind::kMapping) fail("the file must be a mapping of keys");
  for (const auto& [key, value] : root.entries) {
    if (std::find(std::begin(kKeys), std::end(kKeys), key) == std::end(kKeys)) {
      fail("'" + key + "' (line " + std::to_string(value.line) +
           ") is not a camera_info key; they are image_width, image_height, camera_name, "
           "camera_matrix, distortion_model, distortion_coefficients, rectification_matrix and "
           "projection_matrix");
    }
  }
}

void read_distortion(const YamlNode& root, CameraInfo& info) {
  const YamlNode* model = root.find("distortion_model");
  const YamlNode* coefficients = root.find("distortion_coefficients");
  if (model != nullptr && model->kind != YamlNode::Kind::kScalar) {
    fail("distortion_model (line " + std::to_string(model->line) + ") must be a name");
  }
  const std::string name = model != nullptr ? model->scalar : std::string();
  const auto parsed = parse_distortion_model(name);
  if (!parsed) {
    fail("distortion_model '" + name +
         "' is not supported; use plumb_bob, rational_polynomial, equidistant or none");
  }
  info.distortion_model = *parsed;
  if (coefficients != nullptr) {
    info.coefficients = matrix(*coefficients, "distortion_coefficients", 1, 0);
  }
  const std::size_t expected = coefficient_count(info.distortion_model);
  if (info.coefficients.size() != expected) {
    fail("distortion_coefficients has " + std::to_string(info.coefficients.size()) +
         " values, but the " + std::string(distortion_model_name(info.distortion_model)) +
         " model has " + std::to_string(expected) +
         (model == nullptr ? " (distortion_model is missing)" : ""));
  }
}

ParsedCamera parse_camera(std::string_view text) {
  YamlNode root;
  try {
    root = detail::parse_yaml(text);
  } catch (const std::invalid_argument& error) {
    fail(std::string("not a camera_info file: ") + error.what());
  }
  check_keys(root);
  ParsedCamera parsed;
  CameraInfo& info = parsed.info;
  info.width = image_size(root, "image_width");
  info.height = image_size(root, "image_height");
  if (const YamlNode* name = root.find("camera_name")) {
    if (name->kind != YamlNode::Kind::kScalar) fail("camera_name must be a name");
    info.header.frame_id = name->scalar;
  }
  const std::vector<double> k = matrix(required(root, "camera_matrix"), "camera_matrix", 3, 3);
  if (k[3] != 0.0 || k[6] != 0.0 || k[7] != 0.0 || k[8] != 1.0) {
    fail("camera_matrix must be [fx, skew, cx, 0, fy, cy, 0, 0, 1]");
  }
  info.fx = k[0];
  info.skew = k[1];
  info.cx = k[2];
  info.fy = k[4];
  info.cy = k[5];
  read_distortion(root, info);
  const bool distorted = std::any_of(info.coefficients.begin(), info.coefficients.end(),
                                     [](double c) { return c != 0.0; });
  info.state = distorted ? CameraState::kRaw : CameraState::kUndistorted;
  if (const YamlNode* r = root.find("rectification_matrix")) {
    Matrix3 rotation{};
    const std::vector<double> values = matrix(*r, "rectification_matrix", 3, 3);
    std::copy(values.begin(), values.end(), rotation.begin());
    if (detail::max_abs_difference(rotation, kIdentity3) > detail::kRotationTolerance) {
      info.rectification = rotation;
    }
  }
  if (const YamlNode* p = root.find("projection_matrix")) {
    Projection projection{};
    const std::vector<double> values = matrix(*p, "projection_matrix", 3, 4);
    std::copy(values.begin(), values.end(), projection.begin());
    if (projection[4] != 0.0 || projection[8] != 0.0 || projection[9] != 0.0 ||
        projection[10] != 1.0 || projection[11] != 0.0) {
      fail("projection_matrix must be [fx', s', cx', Tx, 0, fy', cy', Ty, 0, 0, 1, 0]");
    }
    if (!(projection[0] > 0 && projection[5] > 0)) {
      fail("projection_matrix must have fx' = P[0] > 0 and fy' = P[5] > 0");
    }
    parsed.projection = projection;
  }
  validate(info, "camera_info");
  return parsed;
}

// The right camera's offset in the rectified frame: P = K' [I | offset].
Vector3 rectified_offset(const ParsedCamera& left, const ParsedCamera& right) {
  const Projection& l = *left.projection;
  const Projection& r = *right.projection;
  return {r[3] / r[0] - l[3] / l[0], r[7] / r[5] - l[7] / l[5], 0.0};
}

bool k_equals_p(const CameraInfo& k, const Projection& p) {
  return detail::nearly_equal(k.fx, p[0], kMatrixTolerance) &&
         detail::nearly_equal(k.skew, p[1], kMatrixTolerance) &&
         detail::nearly_equal(k.cx, p[2], kMatrixTolerance) &&
         detail::nearly_equal(k.fy, p[5], kMatrixTolerance) &&
         detail::nearly_equal(k.cy, p[6], kMatrixTolerance);
}

// Files of a rectified pair: no distortion or rotation left, P = K, and one
// fx, fy and cy.
bool describes_rectified_pair(const ParsedCamera& left, const ParsedCamera& right,
                              const Vector3& offset) {
  const CameraInfo& l = left.info;
  const CameraInfo& r = right.info;
  return l.state == CameraState::kUndistorted && r.state == CameraState::kUndistorted &&
         !l.rectification && !r.rectification && k_equals_p(l, *left.projection) &&
         k_equals_p(r, *right.projection) && l.skew == 0.0 && r.skew == 0.0 &&
         detail::nearly_equal(l.fx, r.fx, kMatrixTolerance) &&
         detail::nearly_equal(l.fy, r.fy, kMatrixTolerance) &&
         detail::nearly_equal(l.cy, r.cy, kMatrixTolerance) &&
         std::abs(offset[1]) <= kMatrixTolerance * std::abs(offset[0]);
}

StereoCameraInfo stereo_from_cameras(ParsedCamera left, ParsedCamera right) {
  if (!left.projection || !right.projection) {
    fail(std::string(!left.projection ? "left" : "right") +
         ": projection_matrix is missing; a stereo pair's files carry the baseline in it");
  }
  const Vector3 offset = rectified_offset(left, right);
  if (!(offset[0] < 0)) {
    fail(
        "right: projection_matrix carries no baseline: Tx = P[3] must be -fx' * baseline, below "
        "the left camera's; use the files a stereo calibration writes");
  }
  StereoCameraInfo stereo;
  stereo.header.frame_id = left.info.header.frame_id;
  if (describes_rectified_pair(left, right, offset)) {
    left.info.state = CameraState::kRectified;
    right.info.state = CameraState::kRectified;
    stereo.baseline = -offset[0];
    stereo.doffs = right.info.cx - left.info.cx;
    stereo.right_from_left.translation = {-stereo.baseline, 0.0, 0.0};
  } else {
    const Matrix3 rect_left = left.info.rectification.value_or(kIdentity3);
    const Matrix3 right_t = detail::transpose(right.info.rectification.value_or(kIdentity3));
    stereo.right_from_left.rotation = detail::multiply(right_t, rect_left);
    stereo.right_from_left.translation = detail::rotate(right_t, offset);
  }
  stereo.left = std::move(left.info);
  stereo.right = std::move(right.info);
  validate(stereo, "stereo_camera_info");
  return stereo;
}

template <typename Parse>
auto with_context(const std::string& context, Parse&& parse) {
  try {
    return parse();
  } catch (const std::invalid_argument& error) {
    fail(context + ": " + error.what());
  }
}

// --- writing ---------------------------------------------------------------

std::string format_number(double value) {
  std::array<char, 32> buffer{};
  const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  if (error != std::errc()) fail("cannot format the number " + std::to_string(value));
  return std::string(buffer.data(), end);
}

// A name as a YAML scalar: plain when that reads back as the same text.
std::string format_name(const std::string& name) {
  const bool plain = !name.empty() && std::all_of(name.begin(), name.end(), [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '/' || c == '.' ||
           c == '-';
  }) && std::string_view("-.").find(name.front()) == std::string_view::npos;
  if (plain) return name;
  std::string quoted = "\"";
  for (const char c : name) {
    if (c == '"' || c == '\\') {
      quoted += '\\';
      quoted += c;
    } else if (c == '\n') {
      quoted += "\\n";
    } else if (c == '\t') {
      quoted += "\\t";
    } else {
      quoted += c;
    }
  }
  return quoted + "\"";
}

void write_matrix(std::ostringstream& out, std::string_view key, std::size_t rows, std::size_t cols,
                  const double* data) {
  out << key << ":\n  rows: " << rows << "\n  cols: " << cols << "\n  data: [";
  for (std::size_t i = 0; i < rows * cols; ++i)
    out << (i == 0 ? "" : ", ") << format_number(data[i]);
  out << "]\n";
}

std::string format_camera(const CameraInfo& info, const Matrix3& rectification,
                          const Projection& projection) {
  std::ostringstream out;
  out << "image_width: " << info.width << "\nimage_height: " << info.height
      << "\ncamera_name: " << format_name(info.header.frame_id) << '\n';
  const Matrix3 k{info.fx, info.skew, info.cx, 0, info.fy, info.cy, 0, 0, 1};
  write_matrix(out, "camera_matrix", 3, 3, k.data());
  out << "distortion_model: " << distortion_model_name(info.distortion_model) << '\n';
  write_matrix(out, "distortion_coefficients", 1, info.coefficients.size(),
               info.coefficients.data());
  write_matrix(out, "rectification_matrix", 3, 3, rectification.data());
  write_matrix(out, "projection_matrix", 3, 4, projection.data());
  return out.str();
}

Projection projection_of(double fx, double fy, double cx, double cy, double skew, double tx) {
  return {fx, skew, cx, tx, 0, fy, cy, 0, 0, 0, 1, 0};
}

void write_text(const std::string& text, const std::filesystem::path& path) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) fail(path.string() + ": cannot open for writing");
  file << text;
  file.close();
  if (!file) fail(path.string() + ": cannot write");
}

}  // namespace

CameraInfo parse_camera_info_yaml(std::string_view text) {
  return with_context("camera_info", [&] { return parse_camera(text).info; });
}

CameraInfo read_camera_info_yaml(const std::filesystem::path& path) {
  const std::string text = read_text(path);
  return with_context(path.string(), [&] { return parse_camera(text).info; });
}

StereoCameraInfo parse_stereo_camera_info_yaml(std::string_view left_text,
                                               std::string_view right_text) {
  ParsedCamera left = with_context("left camera_info", [&] { return parse_camera(left_text); });
  ParsedCamera right = with_context("right camera_info", [&] { return parse_camera(right_text); });
  return with_context("stereo camera_info",
                      [&] { return stereo_from_cameras(std::move(left), std::move(right)); });
}

StereoCameraInfo read_stereo_camera_info_yaml(const std::filesystem::path& left_path,
                                              const std::filesystem::path& right_path) {
  const std::string left_text = read_text(left_path);
  const std::string right_text = read_text(right_path);
  ParsedCamera left = with_context(left_path.string(), [&] { return parse_camera(left_text); });
  ParsedCamera right = with_context(right_path.string(), [&] { return parse_camera(right_text); });
  return with_context(left_path.string() + " and " + right_path.string(),
                      [&] { return stereo_from_cameras(std::move(left), std::move(right)); });
}

std::string format_camera_info_yaml(const CameraInfo& info) {
  validate(info);
  return format_camera(info, info.rectification.value_or(kIdentity3),
                       projection_of(info.fx, info.fy, info.cx, info.cy, info.skew, 0.0));
}

void write_camera_info_yaml(const CameraInfo& info, const std::filesystem::path& path) {
  write_text(format_camera_info_yaml(info), path);
}

std::pair<std::string, std::string> format_stereo_camera_info_yaml(const StereoCameraInfo& info) {
  validate(info);
  const CameraInfo& l = info.left;
  const CameraInfo& r = info.right;
  if (l.state == CameraState::kRectified) {
    return {format_camera(l, kIdentity3, projection_of(l.fx, l.fy, l.cx, l.cy, 0.0, 0.0)),
            format_camera(r, kIdentity3,
                          projection_of(r.fx, r.fy, r.cx, r.cy, 0.0, -r.fx * info.baseline))};
  }
  // P holds the right camera's offset in the frame the rectifications rotate
  // both into. validate() checks that they agree with right_from_left only
  // when a camera has one; here an absent pair must agree too.
  if (!detail::rectifications_agree(info)) {
    fail(
        "stereo_camera_info: a camera_info file holds the extrinsics only as the rectifying "
        "rotations R and the right camera's projection P; give both cameras the rectifications "
        "that turn the pair into one frame with the right camera on its -x axis");
  }
  const Matrix3 rect_left = l.rectification.value_or(kIdentity3);
  const Matrix3 rect_right = r.rectification.value_or(kIdentity3);
  const double baseline =
      detail::norm(detail::rotate(rect_right, info.right_from_left.translation));
  const double f = l.fy;
  return {format_camera(l, rect_left, projection_of(f, f, l.cx, l.cy, 0.0, 0.0)),
          format_camera(r, rect_right, projection_of(f, f, l.cx, l.cy, 0.0, -f * baseline))};
}

void write_stereo_camera_info_yaml(const StereoCameraInfo& info,
                                   const std::filesystem::path& left_path,
                                   const std::filesystem::path& right_path) {
  const auto [left, right] = format_stereo_camera_info_yaml(info);
  write_text(left, left_path);
  write_text(right, right_path);
}

}  // namespace holoscan::messages
