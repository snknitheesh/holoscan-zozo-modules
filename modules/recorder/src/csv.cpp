// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "csv.hpp"

#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace holoscan::recorder::csv {
namespace {

template <typename T>
T to_integer(std::string_view text, const char* what) {
  T value{};
  const char* first = text.data();
  const char* last = text.data() + text.size();
  // from_chars takes a leading '-' for signed types only, and never a '+'.
  const auto [end, error] = std::from_chars(first, last, value);
  if (text.empty() || error != std::errc() || end != last) {
    throw std::invalid_argument(std::string("expected ") + what + ", not \"" + std::string(text) +
                                "\"");
  }
  return value;
}

}  // namespace

Table parse(std::string_view text) {
  Table table;
  std::size_t line = 1;
  std::size_t i = 0;
  while (i < text.size()) {
    Row row;
    row.line = line;
    std::string current;
    bool done = false;
    while (!done) {
      if (i < text.size() && text[i] == '"') {
        // A quoted field, which may hold commas, quotes (doubled) and line breaks.
        ++i;
        while (true) {
          if (i >= text.size()) {
            throw std::invalid_argument("line " + std::to_string(row.line) +
                                        ": a quoted field is not closed");
          }
          if (text[i] == '"') {
            if (i + 1 < text.size() && text[i + 1] == '"') {
              current += '"';
              i += 2;
              continue;
            }
            ++i;
            break;
          }
          if (text[i] == '\n') ++line;
          current += text[i++];
        }
        if (i < text.size() && text[i] != ',' && text[i] != '\n' && text[i] != '\r') {
          throw std::invalid_argument("line " + std::to_string(line) +
                                      ": text after a closing quote");
        }
      } else {
        while (i < text.size() && text[i] != ',' && text[i] != '\n' && text[i] != '\r') {
          if (text[i] == '"') {
            throw std::invalid_argument("line " + std::to_string(line) +
                                        ": a quote inside an unquoted field");
          }
          current += text[i++];
        }
      }
      row.fields.push_back(std::move(current));
      current.clear();
      if (i >= text.size()) {
        table.terminated = false;
        done = true;
      } else if (text[i] == ',') {
        ++i;
        if (i >= text.size()) {
          // A row cut short right after a comma.
          row.fields.emplace_back();
          table.terminated = false;
          done = true;
        }
      } else {
        // A line break: "\n" or "\r\n".
        if (text[i] == '\r') {
          ++i;
          if (i >= text.size() || text[i] != '\n') {
            throw std::invalid_argument("line " + std::to_string(line) +
                                        ": a carriage return without a line feed");
          }
        }
        ++i;
        ++line;
        done = true;
      }
    }
    table.rows.push_back(std::move(row));
  }
  return table;
}

std::string field(std::string_view text) {
  if (text.find_first_of(",\"\r\n") == std::string_view::npos) {
    return std::string(text);
  }
  std::string out = "\"";
  for (const char c : text) {
    if (c == '"') out += '"';
    out += c;
  }
  out += '"';
  return out;
}

std::uint64_t to_u64(std::string_view text) {
  return to_integer<std::uint64_t>(text, "an unsigned 64-bit integer");
}

std::int64_t to_i64(std::string_view text) {
  return to_integer<std::int64_t>(text, "a 64-bit integer");
}

std::int32_t to_i32(std::string_view text) {
  return to_integer<std::int32_t>(text, "a 32-bit integer");
}

std::uint32_t to_u32(std::string_view text) {
  return to_integer<std::uint32_t>(text, "an unsigned 32-bit integer");
}

float to_float(std::string_view text) {
  float value{};
  const char* first = text.data();
  const char* last = text.data() + text.size();
  const auto [end, error] = std::from_chars(first, last, value);
  if (text.empty() || error != std::errc() || end != last || !std::isfinite(value)) {
    throw std::invalid_argument("expected a finite number, not \"" + std::string(text) + "\"");
  }
  return value;
}

std::string from_float(float value) {
  char buffer[32];
  const auto [end, error] = std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (error != std::errc()) {
    throw std::invalid_argument("cannot format a number");
  }
  return std::string(buffer, end);
}

}  // namespace holoscan::recorder::csv
