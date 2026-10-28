// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "recording.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace stereo_disparity_app {
namespace {

// A JSON value: just enough JSON for calibration.json.
struct Json {
  using Object = std::map<std::string, Json>;
  using Array = std::vector<Json>;
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;

  [[nodiscard]] const Json* find(const std::string& key) const {
    const auto* object = std::get_if<Object>(&value);
    if (object == nullptr) return nullptr;
    const auto it = object->find(key);
    return it == object->end() ? nullptr : &it->second;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text) {}

  Json parse() {
    Json json = value();
    skip_space();
    if (at_ != text_.size()) fail("unexpected text after the JSON value");
    return json;
  }

 private:
  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error("JSON: " + what + " at offset " + std::to_string(at_));
  }
  void skip_space() {
    while (at_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[at_]))) ++at_;
  }
  bool take(char c) {
    skip_space();
    if (at_ < text_.size() && text_[at_] == c) {
      ++at_;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!take(c)) fail(std::string("expected '") + c + "'");
  }
  bool take_word(const char* word) {
    const std::string w(word);
    if (text_.compare(at_, w.size(), w) == 0) {
      at_ += w.size();
      return true;
    }
    return false;
  }
  Json value() {
    // Nesting deeper than any calibration file is refused, not recursed into.
    if (++depth_ > 32) fail("nesting too deep");
    skip_space();
    if (at_ >= text_.size()) fail("unexpected end");
    Json json;
    const char c = text_[at_];
    if (c == '{') {
      ++at_;
      Json::Object object;
      if (!take('}')) {
        do {
          skip_space();
          std::string key = string();
          expect(':');
          object.insert_or_assign(std::move(key), value());
        } while (take(','));
        expect('}');
      }
      json.value = std::move(object);
    } else if (c == '[') {
      ++at_;
      Json::Array array;
      if (!take(']')) {
        do {
          array.push_back(value());
        } while (take(','));
        expect(']');
      }
      json.value = std::move(array);
    } else if (c == '"') {
      json.value = string();
    } else if (take_word("true")) {
      json.value = true;
    } else if (take_word("false")) {
      json.value = false;
    } else if (take_word("null")) {
      json.value = nullptr;
    } else {
      const char* begin = text_.c_str() + at_;
      char* end = nullptr;
      const double number = std::strtod(begin, &end);
      if (end == begin) fail("unexpected character");
      at_ += static_cast<std::size_t>(end - begin);
      json.value = number;
    }
    --depth_;
    return json;
  }
  std::string string() {
    if (at_ >= text_.size() || text_[at_] != '"') fail("expected a string");
    ++at_;
    std::string out;
    while (at_ < text_.size() && text_[at_] != '"') {
      char c = text_[at_++];
      if (c == '\\') {
        if (at_ >= text_.size()) break;
        const char escaped = text_[at_++];
        switch (escaped) {
          case 'n':
            c = '\n';
            break;
          case 't':
            c = '\t';
            break;
          case 'u':
            // Not in calibration files; kept as a placeholder character.
            at_ = std::min(at_ + 4, text_.size());
            c = '?';
            break;
          default:
            c = escaped;
        }
      }
      out.push_back(c);
    }
    if (at_ >= text_.size()) fail("unterminated string");
    ++at_;
    return out;
  }

  const std::string& text_;
  std::size_t at_{};
  int depth_{};
};

const Json& member(const Json& json, const std::string& path) {
  const Json* at = &json;
  std::size_t begin = 0;
  while (begin <= path.size()) {
    const std::size_t end = std::min(path.find('.', begin), path.size());
    at = at->find(path.substr(begin, end - begin));
    if (at == nullptr) throw std::runtime_error("calibration: missing " + path);
    begin = end + 1;
  }
  return *at;
}

double number(const Json& json, const std::string& path) {
  const auto* value = std::get_if<double>(&member(json, path).value);
  if (value == nullptr || !std::isfinite(*value)) {
    throw std::runtime_error("calibration: " + path + " must be a number");
  }
  return *value;
}

double element(const Json& json, const std::string& path, std::size_t index) {
  const auto* array = std::get_if<Json::Array>(&member(json, path).value);
  if (array == nullptr || index >= array->size()) {
    throw std::runtime_error("calibration: " + path + " needs element " + std::to_string(index));
  }
  const auto* value = std::get_if<double>(&(*array)[index].value);
  if (value == nullptr) throw std::runtime_error("calibration: " + path + " must hold numbers");
  return *value;
}

std::string text(const Json& json, const std::string& path) {
  const Json* at = &json;
  try {
    at = &member(json, path);
  } catch (const std::runtime_error&) {
    return {};
  }
  const auto* value = std::get_if<std::string>(&at->value);
  return value == nullptr ? std::string() : *value;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error(path.string() + ": cannot open");
  std::ostringstream bytes;
  bytes << file.rdbuf();
  return bytes.str();
}

std::vector<std::string> split(const std::string& line, char separator) {
  std::vector<std::string> fields;
  std::size_t begin = 0;
  while (true) {
    const std::size_t end = line.find(separator, begin);
    fields.push_back(
        line.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
    if (end == std::string::npos) return fields;
    begin = end + 1;
  }
}

}  // namespace

RecordingCalibration parse_calibration(const std::string& json_text) {
  const Json json = JsonParser(json_text).parse();
  const auto* rectified = std::get_if<bool>(&member(json, "rectified").value);
  if (rectified == nullptr || !*rectified) {
    throw std::runtime_error(
        "calibration: the eyes are not rectified; StereoDisparityOp needs a rectified pair");
  }
  RecordingCalibration calibration;
  calibration.camera = text(json, "camera.name");
  if (const std::string serial = text(json, "camera.serial"); !serial.empty()) {
    calibration.camera += " " + serial;
  }
  calibration.width = static_cast<int>(number(json, "left.width"));
  calibration.height = static_cast<int>(number(json, "left.height"));
  if (static_cast<int>(number(json, "right.width")) != calibration.width ||
      static_cast<int>(number(json, "right.height")) != calibration.height) {
    throw std::runtime_error("calibration: the eyes differ in size");
  }
  calibration.fx = number(json, "left.fx");
  calibration.cx = number(json, "left.cx");
  calibration.cy = number(json, "left.cy");
  calibration.right_cx = number(json, "right.cx");
  if (std::abs(number(json, "right.fx") - calibration.fx) > 1e-6 * calibration.fx) {
    throw std::runtime_error("calibration: rectified eyes must share their focal length");
  }
  // right_from_left maps left-eye points into the right eye's frame, so a
  // right eye `baseline` to the right translates them by -baseline along x.
  calibration.baseline = -element(json, "right_from_left.translation", 0);
  if (!(calibration.baseline > 0)) {
    throw std::runtime_error("calibration: the right eye must be to the left eye's right");
  }
  if (json.find("depth") != nullptr) {
    bool aligned = static_cast<int>(number(json, "depth.width")) == calibration.width &&
                   static_cast<int>(number(json, "depth.height")) == calibration.height &&
                   std::abs(number(json, "depth.fx") - calibration.fx) < 1e-3 &&
                   std::abs(number(json, "depth.cx") - calibration.cx) < 1e-3 &&
                   std::abs(number(json, "depth.cy") - calibration.cy) < 1e-3;
    for (std::size_t i = 0; i < 3 && aligned; ++i) {
      aligned = std::abs(element(json, "depth.from_left.translation", i)) < 1e-6;
    }
    if (aligned) {
      calibration.depth_scale = number(json, "depth.scale");
    }
  }
  if (calibration.width < 1 || calibration.height < 1 || !(calibration.fx > 0)) {
    throw std::runtime_error("calibration: needs a positive size and focal length");
  }
  return calibration;
}

Recording read_recording(const std::filesystem::path& folder) {
  Recording recording;
  recording.folder = folder;
  const std::filesystem::path calibration_file = folder / "calibration.json";
  try {
    recording.calibration = parse_calibration(read_file(calibration_file));
  } catch (const std::runtime_error& error) {
    throw std::runtime_error(calibration_file.string() + ": " + error.what());
  }
  const std::filesystem::path index_file = folder / "index.csv";
  std::istringstream index(read_file(index_file));
  std::string line;
  if (!std::getline(index, line)) throw std::runtime_error(index_file.string() + ": empty");
  const std::vector<std::string> header = split(line, ',');
  const auto column = [&](const std::string& name) -> std::optional<std::size_t> {
    for (std::size_t i = 0; i < header.size(); ++i) {
      if (header[i] == name) return i;
    }
    return std::nullopt;
  };
  const auto frame = column("frame");
  const auto left = column("left");
  const auto right = column("right");
  const auto depth = column("depth");
  if (!frame || !left || !right) {
    throw std::runtime_error(index_file.string() + ": needs frame, left and right columns");
  }
  while (std::getline(index, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> fields = split(line, ',');
    if (fields.size() < header.size()) {
      throw std::runtime_error(index_file.string() + ": short row '" + line + "'");
    }
    RecordingFrame entry;
    entry.frame = std::strtoull(fields[*frame].c_str(), nullptr, 10);
    entry.left = folder / fields[*left];
    entry.right = folder / fields[*right];
    if (depth && !fields[*depth].empty()) entry.depth = folder / fields[*depth];
    recording.frames.push_back(std::move(entry));
  }
  if (recording.frames.empty()) throw std::runtime_error(index_file.string() + ": no frames");
  return recording;
}

Pgm parse_pgm(const std::string& bytes, const std::string& name) {
  std::size_t at = 0;
  // The header: P5, width, height and maximum value, separated by white space
  // and comments; then one white space character and the samples.
  const auto token = [&]() -> std::string {
    while (at < bytes.size()) {
      if (bytes[at] == '#') {
        while (at < bytes.size() && bytes[at] != '\n') ++at;
      } else if (std::isspace(static_cast<unsigned char>(bytes[at]))) {
        ++at;
      } else {
        break;
      }
    }
    const std::size_t begin = at;
    while (at < bytes.size() && !std::isspace(static_cast<unsigned char>(bytes[at]))) ++at;
    return bytes.substr(begin, at - begin);
  };
  if (token() != "P5") throw std::runtime_error(name + ": not a binary PGM (P5)");
  Pgm pgm;
  try {
    pgm.width = std::stoi(token());
    pgm.height = std::stoi(token());
    pgm.max_value = std::stoi(token());
  } catch (const std::exception&) {
    throw std::runtime_error(name + ": unreadable PGM header");
  }
  if (pgm.width < 1 || pgm.height < 1 || pgm.width > 16384 || pgm.height > 16384 ||
      pgm.max_value < 1 || pgm.max_value > 65535) {
    throw std::runtime_error(name + ": PGM size or maximum value out of range");
  }
  ++at;  // The single white space character before the samples.
  const std::size_t sample_bytes = pgm.max_value > 255 ? 2 : 1;
  const std::size_t count = static_cast<std::size_t>(pgm.width) * pgm.height;
  if (bytes.size() < at + count * sample_bytes) {
    throw std::runtime_error(name + ": PGM shorter than its header says");
  }
  pgm.pixels.resize(count);
  const auto* data = reinterpret_cast<const unsigned char*>(bytes.data() + at);
  for (std::size_t i = 0; i < count; ++i) {
    pgm.pixels[i] = sample_bytes == 2
                        ? static_cast<std::uint16_t>(data[2 * i] << 8U | data[2 * i + 1])
                        : data[i];
  }
  return pgm;
}

Pgm read_pgm(const std::filesystem::path& path) {
  return parse_pgm(read_file(path), path.string());
}

}  // namespace stereo_disparity_app
