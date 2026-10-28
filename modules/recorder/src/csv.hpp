// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CSV of index.csv (RFC 4180: fields with a comma, a quote or a line break
// are quoted, and quotes in them doubled), and strict number fields.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::recorder::csv {

struct Row {
  std::vector<std::string> fields;
  // The line the row starts on, from 1.
  std::size_t line{};
};

struct Table {
  std::vector<Row> rows;
  // Whether the text ended with a line break, so its last row is complete.
  bool terminated{true};
};

// Parses every row of `text`. A file that does not end with a line break ends
// with a row that may be cut short; `terminated` says so. A blank line is a row
// with one empty field.
// @throws std::invalid_argument "line L: <problem>" for an unterminated quote or
// text after a closing quote.
[[nodiscard]] Table parse(std::string_view text);

// `field`, quoted when it needs to be.
[[nodiscard]] std::string field(std::string_view text);

// Strict number fields: the whole of `text`, in decimal, in range. Each throws
// std::invalid_argument naming what it expected.
[[nodiscard]] std::uint64_t to_u64(std::string_view text);
[[nodiscard]] std::int64_t to_i64(std::string_view text);
[[nodiscard]] std::int32_t to_i32(std::string_view text);
[[nodiscard]] std::uint32_t to_u32(std::string_view text);
[[nodiscard]] float to_float(std::string_view text);
// `value` with the fewest digits that read back as the same float.
[[nodiscard]] std::string from_float(float value);

}  // namespace holoscan::recorder::csv
