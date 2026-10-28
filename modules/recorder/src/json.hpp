// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A small, strict JSON reader and writer, enough for recording.json without a
// third-party library.

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::recorder::json {

struct Value {
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject };

  Kind kind{Kind::kNull};
  bool boolean{};
  double number{};
  std::string string;
  // The elements of an array, or the values of an object's members.
  std::vector<Value> items;
  // The names of an object's members, in order, one for each of `items`.
  std::vector<std::string> keys;

  // The member named `key` of an object, or null when there is none.
  [[nodiscard]] const Value* find(std::string_view key) const noexcept;
};

// "null", "a boolean", "a number", "a string", "an array" or "an object".
[[nodiscard]] const char* kind_name(Value::Kind kind) noexcept;

// Parses one JSON value, which may be surrounded by white space only. Numbers
// are doubles; strings may hold any escape, \u ones as UTF-8. Duplicate member
// names, nesting deeper than 64 and trailing commas are errors.
// @throws std::invalid_argument "line L, column C: <problem>".
[[nodiscard]] Value parse(std::string_view text);

// `text` as a quoted JSON string.
[[nodiscard]] std::string quote(std::string_view text);
// `value` with the fewest digits that read back as the same double. Finite only.
[[nodiscard]] std::string number(double value);
// `value` as compact JSON text, members in their order.
[[nodiscard]] std::string dump(const Value& value);

}  // namespace holoscan::recorder::json
