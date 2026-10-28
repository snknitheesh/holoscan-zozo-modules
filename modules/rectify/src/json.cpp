// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "json.hpp"

#include <cctype>
#include <charconv>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace holoscan::rectify::json {
namespace {

// Nesting allowed, so that a hostile file cannot exhaust the stack.
constexpr int kMaxDepth = 64;

class Parser {
 public:
  explicit Parser(std::string_view text) : text_(text) {}

  Value document() {
    Value value = parse_value(0);
    skip_space();
    if (pos_ != text_.size()) fail("unexpected text after the document");
    return value;
  }

 private:
  [[noreturn]] void fail(const std::string& what) const {
    throw std::invalid_argument("JSON: " + what + " at byte " + std::to_string(pos_));
  }

  void skip_space() {
    while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' ||
                                   text_[pos_] == '\n' || text_[pos_] == '\r')) {
      ++pos_;
    }
  }

  bool consume(char c) {
    skip_space();
    if (pos_ < text_.size() && text_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  void expect(char c) {
    if (!consume(c)) fail(std::string("expected '") + c + "'");
  }

  bool literal(std::string_view word) {
    if (text_.substr(pos_, word.size()) == word) {
      pos_ += word.size();
      return true;
    }
    return false;
  }

  Value parse_value(int depth) {
    if (depth > kMaxDepth) fail("nesting too deep");
    skip_space();
    if (pos_ >= text_.size()) fail("unexpected end");
    const char c = text_[pos_];
    if (c == '{') return Value{parse_object(depth)};
    if (c == '[') return Value{parse_array(depth)};
    if (c == '"') return Value{parse_string()};
    if (literal("true")) return Value{true};
    if (literal("false")) return Value{false};
    if (literal("null")) return Value{nullptr};
    return Value{parse_number()};
  }

  std::shared_ptr<Object> parse_object(int depth) {
    expect('{');
    auto object = std::make_shared<Object>();
    if (consume('}')) return object;
    do {
      skip_space();
      if (pos_ >= text_.size() || text_[pos_] != '"') fail("expected a member name");
      std::string key = parse_string();
      expect(':');
      object->emplace_back(std::move(key), parse_value(depth + 1));
    } while (consume(','));
    expect('}');
    return object;
  }

  std::shared_ptr<Array> parse_array(int depth) {
    expect('[');
    auto array = std::make_shared<Array>();
    if (consume(']')) return array;
    do {
      array->push_back(parse_value(depth + 1));
    } while (consume(','));
    expect(']');
    return array;
  }

  std::string parse_string() {
    ++pos_;  // The opening quote.
    std::string result;
    while (true) {
      if (pos_ >= text_.size()) fail("unterminated string");
      const char c = text_[pos_++];
      if (c == '"') return result;
      if (static_cast<unsigned char>(c) < 0x20) fail("control character in a string");
      if (c != '\\') {
        result.push_back(c);
        continue;
      }
      if (pos_ >= text_.size()) fail("unterminated escape");
      const char e = text_[pos_++];
      switch (e) {
        case '"':
        case '\\':
        case '/':
          result.push_back(e);
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'u': {
          // Calibration files hold ASCII; other code points are kept as '?'.
          if (pos_ + 4 > text_.size()) fail("short \\u escape");
          unsigned code = 0;
          const auto [end, error] =
              std::from_chars(text_.data() + pos_, text_.data() + pos_ + 4, code, 16);
          if (error != std::errc() || end != text_.data() + pos_ + 4) fail("bad \\u escape");
          pos_ += 4;
          result.push_back(code < 0x80 ? static_cast<char>(code) : '?');
          break;
        }
        default:
          fail("unknown escape");
      }
    }
  }

  double parse_number() {
    const std::size_t start = pos_;
    while (pos_ < text_.size() &&
           (std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0 || text_[pos_] == '-' ||
            text_[pos_] == '+' || text_[pos_] == '.' || text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
    }
    if (start == pos_) fail("unexpected character");
    double value = 0.0;
    const auto [end, error] = std::from_chars(text_.data() + start, text_.data() + pos_, value);
    if (error != std::errc() || end != text_.data() + pos_ || !std::isfinite(value)) {
      pos_ = start;
      fail("bad number");
    }
    return value;
  }

  std::string_view text_;
  std::size_t pos_{0};
};

}  // namespace

const Object* Value::object() const noexcept {
  const auto* p = std::get_if<std::shared_ptr<Object>>(&data);
  return p != nullptr ? p->get() : nullptr;
}

const Array* Value::array() const noexcept {
  const auto* p = std::get_if<std::shared_ptr<Array>>(&data);
  return p != nullptr ? p->get() : nullptr;
}

const double* Value::number() const noexcept { return std::get_if<double>(&data); }

const std::string* Value::string() const noexcept { return std::get_if<std::string>(&data); }

const Value* Value::find(std::string_view key) const noexcept {
  const Object* members = object();
  if (members == nullptr) return nullptr;
  for (const auto& [name, value] : *members) {
    if (name == key) return &value;
  }
  return nullptr;
}

Value parse(std::string_view text) { return Parser(text).document(); }

}  // namespace holoscan::rectify::json
