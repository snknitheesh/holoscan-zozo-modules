// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "yaml_reader.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::messages::detail {
namespace {

[[noreturn]] void fail(int line, const std::string& what) {
  throw std::invalid_argument("line " + std::to_string(line) + ": " + what);
}

std::string_view trim(std::string_view text) {
  const auto begin = text.find_first_not_of(" \t\r");
  if (begin == std::string_view::npos) return {};
  const auto end = text.find_last_not_of(" \t\r");
  return text.substr(begin, end - begin + 1);
}

// A line without its comment, its indentation in spaces, and its number.
struct Line {
  int number{};
  int indent{};
  std::string content;
};

// The end of the quoted scalar that starts at text[start], one past its
// closing quote.
std::size_t quoted_end(std::string_view text, std::size_t start, int line) {
  const char quote = text[start];
  for (std::size_t i = start + 1; i < text.size(); ++i) {
    if (quote == '"' && text[i] == '\\') {
      ++i;
    } else if (text[i] == quote) {
      if (quote == '\'' && i + 1 < text.size() && text[i + 1] == '\'') {
        ++i;  // '' is a quote inside single quotes.
      } else {
        return i + 1;
      }
    }
  }
  fail(line, "a quoted scalar must end on the line it starts");
}

// Cuts a comment: a # at the start or after a space, outside quotes.
std::string_view strip_comment(std::string_view text, int line) {
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '"' || c == '\'') {
      // A quote opens a scalar only where one can start; elsewhere (it's) it is text.
      const bool starts_scalar =
          i == 0 || std::string_view(" \t[{,:-").find(text[i - 1]) != std::string_view::npos;
      if (starts_scalar) i = quoted_end(text, i, line) - 1;
    } else if (c == '#' && (i == 0 || text[i - 1] == ' ' || text[i - 1] == '\t')) {
      return text.substr(0, i);
    }
  }
  return text;
}

// Open minus closed brackets outside quotes.
int bracket_balance(std::string_view text, int line) {
  int depth = 0;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if ((c == '"' || c == '\'') &&
        (i == 0 || std::string_view(" \t[{,:").find(text[i - 1]) != std::string_view::npos)) {
      i = quoted_end(text, i, line) - 1;
    } else if (c == '[' || c == '{') {
      ++depth;
    } else if (c == ']' || c == '}') {
      --depth;
    }
  }
  return depth;
}

// The document's lines, without blank lines and comments, a flow collection
// over several lines joined into the line it starts on.
std::vector<Line> logical_lines(std::string_view text) {
  std::vector<Line> lines;
  bool content_seen = false;
  bool document_started = false;
  bool document_ended = false;
  int open = 0;  // Unclosed brackets of the line being joined.
  int number = 0;
  std::size_t position = 0;
  while (position <= text.size()) {
    const std::size_t end = std::min(text.find('\n', position), text.size());
    const std::string_view raw = text.substr(position, end - position);
    position = end + 1;
    ++number;
    const std::string_view content = trim(strip_comment(raw, number));
    if (open > 0) {
      lines.back().content += ' ';
      lines.back().content += content;
      open += bracket_balance(content, number);
      if (open < 0) fail(number, "a ] or } closes nothing");
      continue;
    }
    if (content.empty()) continue;
    if (document_ended) fail(number, "only one document is supported, and it has ended");
    const std::size_t indent = raw.find_first_not_of(' ');
    if (raw[indent] == '\t') fail(number, "indentation must be spaces, not tabs");
    if (indent == 0 && content.front() == '%') {
      if (content_seen || document_started) fail(number, "a directive must come first");
      continue;
    }
    if (indent == 0 && (content == "---" || content.starts_with("--- "))) {
      if (content_seen || document_started) fail(number, "only one document is supported");
      if (content != "---") fail(number, "content after --- must go on its own line");
      document_started = true;
      continue;
    }
    if (indent == 0 && content == "...") {
      document_ended = true;
      continue;
    }
    content_seen = true;
    lines.push_back(Line{number, static_cast<int>(indent), std::string(content)});
    open = bracket_balance(content, number);
    if (open < 0) fail(number, "a ] or } closes nothing");
  }
  if (open > 0) fail(number, "a [ or { is never closed");
  return lines;
}

bool is_sequence_item(std::string_view content) {
  return content == "-" || content.starts_with("- ");
}

// A scalar written without quotes: refuses YAML's indicators this reader
// does not support, rather than taking them as text.
YamlNode plain_scalar(std::string_view text, int line) {
  text = trim(text);
  if (!text.empty() && std::string_view("&*!|>@`%").find(text.front()) != std::string_view::npos) {
    fail(line,
         "'" + std::string(text) + "': anchors, aliases, tags and block scalars are not supported");
  }
  if (text.find(": ") != std::string_view::npos || text.ends_with(":")) {
    fail(line, "'" + std::string(text) + "' is not a value; a mapping on one line needs braces");
  }
  YamlNode node;
  node.scalar = std::string(text);
  node.line = line;
  return node;
}

YamlNode quoted_scalar(std::string_view text, int line) {
  const char quote = text.front();
  YamlNode node;
  node.quoted = true;
  node.line = line;
  for (std::size_t i = 1; i + 1 < text.size(); ++i) {
    char c = text[i];
    if (quote == '\'' && c == '\'') {
      ++i;  // '' is one quote.
    } else if (quote == '"' && c == '\\') {
      c = text[++i];
      switch (c) {
        case 'n':
          c = '\n';
          break;
        case 't':
          c = '\t';
          break;
        case '"':
        case '\\':
        case '/':
          break;
        default:
          fail(line, std::string("the escape \\") + c + " is not supported");
      }
    }
    node.scalar += c;
  }
  return node;
}

// Flow collections: [a, b] and {key: value}, nested.
class FlowParser {
 public:
  FlowParser(std::string_view text, int line) : text_(text), line_(line) {}

  YamlNode value() {
    skip_space();
    if (at_end()) fail(line_, "a value is missing");
    const char c = text_[position_];
    if (c == '[') return sequence();
    if (c == '{') return mapping();
    if (c == '"' || c == '\'') {
      const std::size_t end = quoted_end(text_, position_, line_);
      YamlNode node = quoted_scalar(text_.substr(position_, end - position_), line_);
      position_ = end;
      return node;
    }
    const std::size_t end = text_.find_first_of(",]}", position_);
    const std::string_view scalar = trim(text_.substr(position_, end - position_));
    if (scalar.empty()) fail(line_, "a value is missing");
    position_ = end == std::string_view::npos ? text_.size() : end;
    return plain_scalar(scalar, line_);
  }

  void expect_end() {
    skip_space();
    if (!at_end()) fail(line_, "'" + std::string(text_.substr(position_)) + "' follows the value");
  }

 private:
  YamlNode sequence() {
    YamlNode node;
    node.kind = YamlNode::Kind::kSequence;
    node.line = line_;
    ++position_;  // [
    skip_space();
    if (take(']')) return node;
    while (true) {
      node.items.push_back(value());
      skip_space();
      if (take(']')) return node;
      if (!take(',')) fail(line_, "expected , or ] in a [ ] list");
      skip_space();
      if (take(']')) return node;  // A trailing comma.
    }
  }

  YamlNode mapping() {
    YamlNode node;
    node.kind = YamlNode::Kind::kMapping;
    node.line = line_;
    ++position_;  // {
    skip_space();
    if (take('}')) return node;
    while (true) {
      std::string key = this->key();
      if (node.find(key) != nullptr) fail(line_, "the key '" + key + "' is given twice");
      node.entries.emplace_back(std::move(key), value());
      skip_space();
      if (take('}')) return node;
      if (!take(',')) fail(line_, "expected , or } in a { } mapping");
      skip_space();
      if (take('}')) return node;
    }
  }

  std::string key() {
    skip_space();
    std::string key;
    if (!at_end() && (text_[position_] == '"' || text_[position_] == '\'')) {
      const std::size_t end = quoted_end(text_, position_, line_);
      key = quoted_scalar(text_.substr(position_, end - position_), line_).scalar;
      position_ = end;
      skip_space();
    } else {
      const std::size_t end = text_.find_first_of(":,}", position_);
      key = std::string(trim(text_.substr(position_, end - position_)));
      position_ = end == std::string_view::npos ? text_.size() : end;
    }
    if (key.empty() || !take(':')) fail(line_, "expected key: value in a { } mapping");
    return key;
  }

  void skip_space() {
    while (!at_end() && (text_[position_] == ' ' || text_[position_] == '\t')) ++position_;
  }
  bool take(char c) {
    if (at_end() || text_[position_] != c) return false;
    ++position_;
    return true;
  }
  [[nodiscard]] bool at_end() const { return position_ >= text_.size(); }

  std::string_view text_;
  int line_;
  std::size_t position_{};
};

// A value on the line of its key or sequence dash.
YamlNode inline_value(std::string_view text, int line) {
  text = trim(text);
  if (text.front() == '[' || text.front() == '{' || text.front() == '"' || text.front() == '\'') {
    FlowParser parser(text, line);
    YamlNode node = parser.value();
    parser.expect_end();
    return node;
  }
  return plain_scalar(text, line);
}

class BlockParser {
 public:
  explicit BlockParser(std::vector<Line> lines) : lines_(std::move(lines)) {}

  YamlNode document() {
    if (lines_.empty()) {
      YamlNode empty;
      empty.kind = YamlNode::Kind::kMapping;
      return empty;
    }
    YamlNode root = block(lines_.front().indent);
    if (next_ < lines_.size()) fail(lines_[next_].number, "unexpected indentation");
    return root;
  }

 private:
  YamlNode block(int indent) {
    return is_sequence_item(lines_[next_].content) ? sequence(indent) : mapping(indent);
  }

  // The value of a key or dash with nothing after it: the indented block
  // below it (a sequence may also stand at the key's own indentation), or
  // an empty scalar.
  YamlNode nested(int indent, int line, bool allow_same_indent_sequence) {
    if (next_ < lines_.size()) {
      const Line& below = lines_[next_];
      if (below.indent > indent || (allow_same_indent_sequence && below.indent == indent &&
                                    is_sequence_item(below.content))) {
        return block(below.indent);
      }
    }
    YamlNode empty;
    empty.line = line;
    return empty;
  }

  YamlNode mapping(int indent) {
    YamlNode node;
    node.kind = YamlNode::Kind::kMapping;
    node.line = lines_[next_].number;
    while (next_ < lines_.size() && lines_[next_].indent == indent) {
      const Line& line = lines_[next_];
      if (is_sequence_item(line.content)) fail(line.number, "a list item where a key belongs");
      auto [key, rest] = split_key(line);
      if (node.find(key) != nullptr) fail(line.number, "the key '" + key + "' is given twice");
      ++next_;
      YamlNode value =
          rest.empty() ? nested(indent, line.number, true) : inline_value(rest, line.number);
      node.entries.emplace_back(std::move(key), std::move(value));
    }
    if (next_ < lines_.size() && lines_[next_].indent > indent) {
      fail(lines_[next_].number, "unexpected indentation");
    }
    return node;
  }

  YamlNode sequence(int indent) {
    YamlNode node;
    node.kind = YamlNode::Kind::kSequence;
    node.line = lines_[next_].number;
    while (next_ < lines_.size() && lines_[next_].indent == indent &&
           is_sequence_item(lines_[next_].content)) {
      const Line& line = lines_[next_];
      const std::string_view rest = trim(std::string_view(line.content).substr(1));
      ++next_;
      if (rest.empty()) {
        node.items.push_back(nested(indent, line.number, false));
      } else if (rest.front() != '[' && rest.front() != '{' && rest.front() != '"' &&
                 rest.front() != '\'' &&
                 (rest.find(": ") != std::string_view::npos || rest.ends_with(":"))) {
        fail(line.number, "a mapping inside a list item is not supported");
      } else {
        node.items.push_back(inline_value(rest, line.number));
      }
    }
    if (next_ < lines_.size() && lines_[next_].indent > indent) {
      fail(lines_[next_].number, "unexpected indentation");
    }
    return node;
  }

  // "key: rest", the key unquoted.
  static std::pair<std::string, std::string_view> split_key(const Line& line) {
    const std::string_view content = line.content;
    if (content.front() == '"' || content.front() == '\'') {
      const std::size_t end = quoted_end(content, 0, line.number);
      if (end >= content.size() || content[end] != ':') {
        fail(line.number, "expected 'key: value'");
      }
      return {quoted_scalar(content.substr(0, end), line.number).scalar, content.substr(end + 1)};
    }
    if (content.front() == '[' || content.front() == '{' || content.front() == '?') {
      fail(line.number, "complex keys are not supported");
    }
    std::size_t colon = content.find(':');
    while (colon != std::string_view::npos && colon + 1 < content.size() &&
           content[colon + 1] != ' ') {
      colon = content.find(':', colon + 1);
    }
    if (colon == std::string_view::npos) fail(line.number, "expected 'key: value'");
    const std::string key(trim(content.substr(0, colon)));
    if (key.empty()) fail(line.number, "a key is missing");
    return {key, content.substr(colon + 1)};
  }

  std::vector<Line> lines_;
  std::size_t next_{};
};

}  // namespace

const YamlNode* YamlNode::find(std::string_view key) const noexcept {
  for (const auto& [name, value] : entries) {
    if (name == key) return &value;
  }
  return nullptr;
}

YamlNode parse_yaml(std::string_view text) {
  // A UTF-8 byte order mark, which some editors write, is no content.
  if (text.starts_with("\xEF\xBB\xBF")) text.remove_prefix(3);
  return BlockParser(logical_lines(text)).document();
}

}  // namespace holoscan::messages::detail
