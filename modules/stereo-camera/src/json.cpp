// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "json.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace holoscan::stereo_camera::json {
namespace {

constexpr int kMaxDepth = 64;

class Parser {
 public:
  explicit Parser(std::string_view text) : text_(text) {}

  Value document() {
    skip_space();
    Value value = parse_value(0);
    skip_space();
    if (position_ != text_.size()) {
      fail("unexpected text after the value");
    }
    return value;
  }

 private:
  [[noreturn]] void fail(const std::string& problem) const {
    std::size_t line = 1;
    std::size_t column = 1;
    for (std::size_t i = 0; i < position_ && i < text_.size(); ++i) {
      if (text_[i] == '\n') {
        ++line;
        column = 1;
      } else {
        ++column;
      }
    }
    throw std::invalid_argument("line " + std::to_string(line) + ", column " +
                                std::to_string(column) + ": " + problem);
  }

  [[nodiscard]] bool at_end() const noexcept { return position_ >= text_.size(); }
  [[nodiscard]] char peek() const noexcept { return at_end() ? '\0' : text_[position_]; }

  void skip_space() noexcept {
    while (!at_end() && (peek() == ' ' || peek() == '\t' || peek() == '\n' || peek() == '\r')) {
      ++position_;
    }
  }

  void expect(char c) {
    if (peek() != c) {
      fail(std::string("expected '") + c + "'");
    }
    ++position_;
  }

  void literal(std::string_view word) {
    if (text_.substr(position_, word.size()) != word) {
      fail("unexpected character");
    }
    position_ += word.size();
  }

  Value parse_value(int depth) {
    if (depth > kMaxDepth) {
      fail("nested deeper than 64");
    }
    Value value;
    switch (peek()) {
      case '{':
        parse_object(value, depth);
        break;
      case '[':
        parse_array(value, depth);
        break;
      case '"':
        value.kind = Value::Kind::kString;
        value.string = parse_string();
        break;
      case 't':
        literal("true");
        value.kind = Value::Kind::kBool;
        value.boolean = true;
        break;
      case 'f':
        literal("false");
        value.kind = Value::Kind::kBool;
        break;
      case 'n':
        literal("null");
        break;
      default:
        value.kind = Value::Kind::kNumber;
        value.number = parse_number();
        break;
    }
    return value;
  }

  void parse_object(Value& value, int depth) {
    value.kind = Value::Kind::kObject;
    expect('{');
    skip_space();
    if (peek() == '}') {
      ++position_;
      return;
    }
    while (true) {
      skip_space();
      if (peek() != '"') {
        fail("expected a member name");
      }
      std::string key = parse_string();
      if (value.find(key) != nullptr) {
        fail("duplicate member \"" + key + "\"");
      }
      skip_space();
      expect(':');
      skip_space();
      value.keys.push_back(std::move(key));
      value.items.push_back(parse_value(depth + 1));
      skip_space();
      if (peek() == ',') {
        ++position_;
        continue;
      }
      expect('}');
      return;
    }
  }

  void parse_array(Value& value, int depth) {
    value.kind = Value::Kind::kArray;
    expect('[');
    skip_space();
    if (peek() == ']') {
      ++position_;
      return;
    }
    while (true) {
      skip_space();
      value.items.push_back(parse_value(depth + 1));
      skip_space();
      if (peek() == ',') {
        ++position_;
        continue;
      }
      expect(']');
      return;
    }
  }

  unsigned hex4() {
    if (position_ + 4 > text_.size()) {
      fail("incomplete \\u escape");
    }
    unsigned code = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = text_[position_++];
      code <<= 4U;
      if (c >= '0' && c <= '9') {
        code |= static_cast<unsigned>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        code |= static_cast<unsigned>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        code |= static_cast<unsigned>(c - 'A' + 10);
      } else {
        fail("bad \\u escape");
      }
    }
    return code;
  }

  static void append_utf8(std::string& out, unsigned code) {
    if (code < 0x80U) {
      out += static_cast<char>(code);
    } else if (code < 0x800U) {
      out += static_cast<char>(0xC0U | (code >> 6U));
      out += static_cast<char>(0x80U | (code & 0x3FU));
    } else if (code < 0x10000U) {
      out += static_cast<char>(0xE0U | (code >> 12U));
      out += static_cast<char>(0x80U | ((code >> 6U) & 0x3FU));
      out += static_cast<char>(0x80U | (code & 0x3FU));
    } else {
      out += static_cast<char>(0xF0U | (code >> 18U));
      out += static_cast<char>(0x80U | ((code >> 12U) & 0x3FU));
      out += static_cast<char>(0x80U | ((code >> 6U) & 0x3FU));
      out += static_cast<char>(0x80U | (code & 0x3FU));
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (at_end()) {
        fail("unterminated string");
      }
      const char c = text_[position_++];
      if (c == '"') {
        return out;
      }
      if (static_cast<unsigned char>(c) < 0x20U) {
        --position_;
        fail("control character in a string");
      }
      if (c != '\\') {
        out += c;
        continue;
      }
      if (at_end()) {
        fail("unterminated string");
      }
      const char escape = text_[position_++];
      switch (escape) {
        case '"':
        case '\\':
        case '/':
          out += escape;
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'u': {
          unsigned code = hex4();
          if (code >= 0xD800U && code <= 0xDBFFU) {
            if (text_.substr(position_, 2) != "\\u") {
              fail("unpaired surrogate");
            }
            position_ += 2;
            const unsigned low = hex4();
            if (low < 0xDC00U || low > 0xDFFFU) {
              fail("unpaired surrogate");
            }
            code = 0x10000U + ((code - 0xD800U) << 10U) + (low - 0xDC00U);
          } else if (code >= 0xDC00U && code <= 0xDFFFU) {
            fail("unpaired surrogate");
          }
          append_utf8(out, code);
          break;
        }
        default:
          --position_;
          fail("bad escape");
      }
    }
  }

  double parse_number() {
    // The JSON grammar: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
    const std::size_t start = position_;
    const auto digits = [this]() {
      const std::size_t first = position_;
      while (!at_end() && peek() >= '0' && peek() <= '9') ++position_;
      return position_ - first;
    };
    if (peek() == '-') ++position_;
    if (peek() == '0') {
      ++position_;
    } else if (digits() == 0) {
      position_ = start;
      fail("unexpected character");
    }
    if (peek() == '.') {
      ++position_;
      if (digits() == 0) fail("expected a digit after '.'");
    }
    if (peek() == 'e' || peek() == 'E') {
      ++position_;
      if (peek() == '+' || peek() == '-') ++position_;
      if (digits() == 0) fail("expected a digit in the exponent");
    }
    double value = 0.0;
    const char* first = text_.data() + start;
    const char* last = text_.data() + position_;
    const auto [end, error] = std::from_chars(first, last, value);
    if (error != std::errc() || end != last || !std::isfinite(value)) {
      position_ = start;
      fail("number out of range");
    }
    return value;
  }

  std::string_view text_;
  std::size_t position_{};
};

}  // namespace

const Value* Value::find(std::string_view key) const noexcept {
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (keys[i] == key) {
      return &items[i];
    }
  }
  return nullptr;
}

const char* kind_name(Value::Kind kind) noexcept {
  switch (kind) {
    case Value::Kind::kNull:
      return "null";
    case Value::Kind::kBool:
      return "a boolean";
    case Value::Kind::kNumber:
      return "a number";
    case Value::Kind::kString:
      return "a string";
    case Value::Kind::kArray:
      return "an array";
    case Value::Kind::kObject:
      return "an object";
  }
  return "unknown";
}

Value parse(std::string_view text) { return Parser(text).document(); }

std::string quote(std::string_view text) {
  std::string out = "\"";
  for (const char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20U) {
          static constexpr char kHex[] = "0123456789abcdef";
          out += "\\u00";
          out += kHex[(static_cast<unsigned char>(c) >> 4U) & 0xFU];
          out += kHex[static_cast<unsigned char>(c) & 0xFU];
        } else {
          out += c;
        }
    }
  }
  out += '"';
  return out;
}

std::string number(double value) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument("JSON cannot hold a number that is not finite");
  }
  if (value == 0.0) {
    return "0";  // Also for -0.0, which JSON readers may not keep.
  }
  char buffer[32];
  const auto [end, error] = std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (error != std::errc()) {
    throw std::invalid_argument("cannot format a number");
  }
  return std::string(buffer, end);
}

}  // namespace holoscan::stereo_camera::json
