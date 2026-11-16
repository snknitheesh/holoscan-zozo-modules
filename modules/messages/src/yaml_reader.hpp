// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A reader for the part of YAML that camera_info files use, so that the
// module's SDK-free part, which builds and is tested without the Holoscan SDK
// (whose YAML reader, holoscan/config/io.hpp, it therefore cannot use), needs
// no YAML library: block mappings and sequences by indentation, flow
// sequences and mappings (also over several lines), plain and quoted scalars,
// comments, a `%YAML` directive, one `---` document and a leading UTF-8 byte
// order mark. Anything else (anchors, aliases, tags, block scalars, a second
// document, tabs in indentation, a key given twice) is refused with its line
// number, never misread.

#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace holoscan::messages::detail {

struct YamlNode {
  enum class Kind { kScalar, kMapping, kSequence };

  Kind kind{Kind::kScalar};
  // A scalar's text, without its quotes; empty for an empty value.
  std::string scalar;
  bool quoted{};
  // A mapping's entries, in file order.
  std::vector<std::pair<std::string, YamlNode>> entries;
  // A sequence's items.
  std::vector<YamlNode> items;
  // 1-based, for messages.
  int line{};

  // The value of `key` in a mapping, or nullptr.
  [[nodiscard]] const YamlNode* find(std::string_view key) const noexcept;
};

// @throws std::invalid_argument "line N: ..." for text outside the subset.
[[nodiscard]] YamlNode parse_yaml(std::string_view text);

}  // namespace holoscan::messages::detail
