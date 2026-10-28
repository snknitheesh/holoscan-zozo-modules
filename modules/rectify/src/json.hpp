// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A small JSON reader, enough for calibration files: objects, arrays, numbers,
// strings, booleans and null. No SDK.

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace holoscan::rectify::json {

struct Value;
using Object = std::vector<std::pair<std::string, Value>>;
using Array = std::vector<Value>;

struct Value {
  std::variant<std::nullptr_t, bool, double, std::string, std::shared_ptr<Array>,
               std::shared_ptr<Object>>
      data;

  [[nodiscard]] const Object* object() const noexcept;
  [[nodiscard]] const Array* array() const noexcept;
  [[nodiscard]] const double* number() const noexcept;
  [[nodiscard]] const std::string* string() const noexcept;
  // The member `key` of an object; nullptr when absent or not an object.
  [[nodiscard]] const Value* find(std::string_view key) const noexcept;
};

// @throws std::invalid_argument naming the byte offset of a syntax error.
[[nodiscard]] Value parse(std::string_view text);

}  // namespace holoscan::rectify::json
