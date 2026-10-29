// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Calibration files: a recording's calibration.json and ROS camera_info YAML.

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/rectify/camera_model.hpp"
#include "holoscan/rectify/rectification.hpp"
#include "json.hpp"

namespace holoscan::rectify {
namespace {

// Largest calibration file read, far above any real one.
constexpr std::uintmax_t kMaxFileBytes = 1U << 20;

std::string read_text(const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error) {
    throw std::invalid_argument(path.string() + ": cannot read: " + error.message());
  }
  if (size > kMaxFileBytes) {
    throw std::invalid_argument(path.string() + ": larger than a calibration file can be");
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::invalid_argument(path.string() + ": cannot open");
  std::ostringstream text;
  text << file.rdbuf();
  return text.str();
}

// --- calibration.json ------------------------------------------------------

double number(const json::Value& parent, std::string_view key, const std::string& where) {
  const json::Value* value = parent.find(key);
  if (value == nullptr || value->number() == nullptr) {
    throw std::invalid_argument(where + "." + std::string(key) + " must be a number");
  }
  return *value->number();
}

int integer(const json::Value& parent, std::string_view key, const std::string& where) {
  const double value = number(parent, key, where);
  if (value != std::floor(value) || value < 0 || value > kMaxImageSide) {
    throw std::invalid_argument(where + "." + std::string(key) + " must be a whole number 0.." +
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

CameraModel camera_from_json(const json::Value& root, std::string_view key) {
  const std::string where(key);
  const json::Value* node = root.find(key);
  if (node == nullptr || node->object() == nullptr) {
    throw std::invalid_argument(where + " must be an object");
  }
  CameraModel camera;
  camera.width = integer(*node, "width", where);
  camera.height = integer(*node, "height", where);
  camera.fx = number(*node, "fx", where);
  camera.fy = number(*node, "fy", where);
  camera.cx = number(*node, "cx", where);
  camera.cy = number(*node, "cy", where);
  const json::Value* distortion = node->find("distortion");
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
  }
  validate(camera, where);
  return camera;
}

// --- camera_info YAML -----------------------------------------------------

// The few YAML forms camera_info files use: top-level `key: value`, and a
// matrix as an indented `rows`, `cols` and `data` with a flow list (possibly
// over several lines) or a block list.
struct YamlNode {
  std::string scalar;
  std::map<std::string, std::string> scalars;  // Indented `key: value` lines.
  std::vector<std::string> list;               // `data`'s items.
  bool has_list{};
};

std::string trim(std::string_view text) {
  const auto begin = text.find_first_not_of(" \t\r");
  if (begin == std::string_view::npos) return {};
  const auto end = text.find_last_not_of(" \t\r");
  return std::string(text.substr(begin, end - begin + 1));
}

std::string unquote(std::string text) {
  if (text.size() >= 2 && ((text.front() == '"' && text.back() == '"') ||
                           (text.front() == '\'' && text.back() == '\''))) {
    return text.substr(1, text.size() - 2);
  }
  return text;
}

std::string strip_comment(const std::string& line) {
  bool quoted = false;
  char quote = 0;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quoted) {
      if (c == quote) quoted = false;
    } else if (c == '"' || c == '\'') {
      quoted = true;
      quote = c;
    } else if (c == '#' && (i == 0 || line[i - 1] == ' ' || line[i - 1] == '\t')) {
      return line.substr(0, i);
    }
  }
  return line;
}

void split_list(std::string_view items, std::vector<std::string>& out) {
  std::size_t start = 0;
  while (start <= items.size()) {
    const std::size_t comma = items.find(',', start);
    const std::string item = trim(items.substr(
        start, comma == std::string_view::npos ? std::string_view::npos : comma - start));
    if (!item.empty()) out.push_back(item);
    if (comma == std::string_view::npos) break;
    start = comma + 1;
  }
}

std::map<std::string, YamlNode> parse_yaml(std::string_view text) {
  std::map<std::string, YamlNode> nodes;
  std::istringstream stream{std::string(text)};
  std::string raw;
  YamlNode* current = nullptr;
  std::string flow;  // A flow list being collected over several lines.
  bool in_flow = false;
  int line_number = 0;
  while (std::getline(stream, raw)) {
    ++line_number;
    const std::string line = strip_comment(raw);
    const std::string content = trim(line);
    if (in_flow) {
      flow += " " + content;
      if (content.find(']') != std::string::npos) {
        in_flow = false;
        const auto close = flow.find(']');
        split_list(std::string_view(flow).substr(0, close), current->list);
      }
      continue;
    }
    if (content.empty() || content == "---" || content.starts_with("%")) continue;
    const bool indented = line.front() == ' ' || line.front() == '\t';
    if (content.starts_with("- ") || content == "-") {
      if (current == nullptr || !current->has_list) {
        throw std::invalid_argument("camera_info line " + std::to_string(line_number) +
                                    ": a list item outside a data list");
      }
      current->list.push_back(trim(content.substr(1)));
      continue;
    }
    const auto colon = content.find(':');
    if (colon == std::string::npos) {
      throw std::invalid_argument("camera_info line " + std::to_string(line_number) +
                                  ": expected 'key: value'");
    }
    const std::string key = trim(std::string_view(content).substr(0, colon));
    const std::string value = trim(std::string_view(content).substr(colon + 1));
    if (!indented) {
      current = &nodes[key];
      current->scalar = unquote(value);
      continue;
    }
    if (current == nullptr) {
      throw std::invalid_argument("camera_info line " + std::to_string(line_number) +
                                  ": indented line outside a block");
    }
    if (key != "data") {
      current->scalars[key] = value;
      continue;
    }
    current->has_list = true;
    if (value.starts_with("[")) {
      const auto close = value.find(']');
      if (close == std::string::npos) {
        in_flow = true;
        flow = value.substr(1);
      } else {
        split_list(std::string_view(value).substr(1, close - 1), current->list);
      }
    }
  }
  if (in_flow) throw std::invalid_argument("camera_info: unterminated data list");
  return nodes;
}

double yaml_number(const std::string& text, const std::string& what) {
  double value = 0.0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size() || !std::isfinite(value)) {
    throw std::invalid_argument("camera_info: " + what + " holds '" + text + "', not a number");
  }
  return value;
}

const YamlNode* node(const std::map<std::string, YamlNode>& nodes, const std::string& key) {
  const auto found = nodes.find(key);
  return found == nodes.end() ? nullptr : &found->second;
}

std::vector<double> yaml_matrix(const std::map<std::string, YamlNode>& nodes,
                                const std::string& key, std::size_t expected, bool required) {
  const YamlNode* matrix = node(nodes, key);
  if (matrix == nullptr) {
    if (required) throw std::invalid_argument("camera_info: " + key + " is missing");
    return {};
  }
  std::vector<double> values;
  for (const std::string& item : matrix->list) values.push_back(yaml_number(item, key));
  if (expected != 0 && values.size() != expected) {
    throw std::invalid_argument("camera_info: " + key + " needs " + std::to_string(expected) +
                                " values, not " + std::to_string(values.size()));
  }
  return values;
}

}  // namespace

StereoCalibration parse_stereo_calibration(std::string_view text) {
  const json::Value root = json::parse(text);
  if (root.object() == nullptr) {
    throw std::invalid_argument("calibration: the document must be an object");
  }
  if (const json::Value* version = root.find("version");
      version != nullptr && (version->number() == nullptr || *version->number() != 1.0)) {
    throw std::invalid_argument("calibration: version must be 1");
  }
  StereoCalibration calibration;
  calibration.left = camera_from_json(root, "left");
  calibration.right = camera_from_json(root, "right");
  const json::Value* pose = root.find("right_from_left");
  if (pose == nullptr || pose->object() == nullptr) {
    throw std::invalid_argument("right_from_left must be an object");
  }
  const std::vector<double> rotation = numbers(*pose, "rotation", "right_from_left");
  const std::vector<double> translation = numbers(*pose, "translation", "right_from_left");
  if (rotation.size() != 9 || translation.size() != 3) {
    throw std::invalid_argument(
        "right_from_left needs a rotation of 9 numbers and a translation of 3");
  }
  std::copy(rotation.begin(), rotation.end(), calibration.right_from_left.rotation.begin());
  std::copy(translation.begin(), translation.end(),
            calibration.right_from_left.translation.begin());
  validate(calibration);
  return calibration;
}

StereoCalibration read_stereo_calibration(const std::filesystem::path& path) {
  const std::string text = read_text(path);
  try {
    return parse_stereo_calibration(text);
  } catch (const std::invalid_argument& error) {
    throw std::invalid_argument(path.string() + ": " + error.what());
  }
}

CameraInfo parse_camera_info(std::string_view text) {
  const auto nodes = parse_yaml(text);
  CameraInfo info;
  const auto size = [&nodes](const std::string& key) {
    const YamlNode* n = node(nodes, key);
    if (n == nullptr) throw std::invalid_argument("camera_info: " + key + " is missing");
    const double value = yaml_number(n->scalar, key);
    if (value != std::floor(value) || value < 1 || value > kMaxImageSide) {
      throw std::invalid_argument("camera_info: " + key + " must be 1.." +
                                  std::to_string(kMaxImageSide));
    }
    return static_cast<int>(value);
  };
  info.camera.width = size("image_width");
  info.camera.height = size("image_height");
  if (const YamlNode* name = node(nodes, "camera_name")) info.camera_name = name->scalar;
  const std::vector<double> k = yaml_matrix(nodes, "camera_matrix", 9, true);
  if (std::abs(k[1]) > 1e-9) {
    throw std::invalid_argument("camera_info: camera_matrix has a skew, which is not supported");
  }
  info.camera.fx = k[0];
  info.camera.cx = k[2];
  info.camera.fy = k[4];
  info.camera.cy = k[5];
  std::vector<double> d = yaml_matrix(nodes, "distortion_coefficients", 0, false);
  const bool all_zero = std::all_of(d.begin(), d.end(), [](double v) { return v == 0.0; });
  const YamlNode* model = node(nodes, "distortion_model");
  if (model != nullptr && !model->scalar.empty()) {
    const auto parsed = parse_distortion_model(model->scalar);
    if (!parsed) {
      throw std::invalid_argument("camera_info: distortion_model '" + model->scalar +
                                  "' is not supported; use plumb_bob, rational_polynomial or "
                                  "equidistant");
    }
    info.camera.distortion_model = *parsed;
  } else if (!all_zero) {
    throw std::invalid_argument("camera_info: distortion_coefficients without distortion_model");
  }
  // ROS writes plumb_bob with 5 values, and sometimes rational_polynomial with
  // more than 8 (thin prism terms): those must be zero.
  const std::size_t count = distortion_coefficient_count(info.camera.distortion_model);
  if (info.camera.distortion_model == DistortionModel::kBrownConrady && d.size() == 4) {
    d.push_back(0.0);
  }
  if (d.size() > count) {
    if (!std::all_of(d.begin() + static_cast<std::ptrdiff_t>(count), d.end(),
                     [](double v) { return v == 0.0; })) {
      throw std::invalid_argument("camera_info: distortion_coefficients has nonzero terms the " +
                                  std::string(distortion_model_name(info.camera.distortion_model)) +
                                  " model does not have");
    }
    d.resize(count);
  }
  info.camera.distortion = all_zero ? std::vector<double>{} : d;
  validate(info.camera, "camera_info");
  if (const auto r = yaml_matrix(nodes, "rectification_matrix", 9, false); !r.empty()) {
    std::copy(r.begin(), r.end(), info.rectification.begin());
  }
  if (const auto p = yaml_matrix(nodes, "projection_matrix", 12, false); !p.empty()) {
    std::copy(p.begin(), p.end(), info.projection.begin());
  }
  return info;
}

CameraInfo read_camera_info(const std::filesystem::path& path) {
  const std::string text = read_text(path);
  try {
    return parse_camera_info(text);
  } catch (const std::invalid_argument& error) {
    throw std::invalid_argument(path.string() + ": " + error.what());
  }
}

StereoCalibration stereo_from_camera_info(const CameraInfo& left, const CameraInfo& right) {
  const auto& p = right.projection;
  if (!(p[0] > 0 && p[5] > 0) || p[3] == 0.0) {
    throw std::invalid_argument(
        "the right camera_info's projection_matrix carries no baseline (P[3] = -fx * baseline); "
        "use the files a stereo calibration writes");
  }
  StereoCalibration calibration;
  calibration.left = left.camera;
  calibration.right = right.camera;
  // p_rect = R_left p_left = R_right p_right - t_rect, with t_rect the right
  // rectified camera's offset in the rectified frame.
  const Matrix3 right_t = transpose(right.rectification);
  calibration.right_from_left.rotation = multiply(right_t, left.rectification);
  calibration.right_from_left.translation = rotate(right_t, {p[3] / p[0], p[7] / p[5], 0.0});
  validate(calibration);
  return calibration;
}

}  // namespace holoscan::rectify
