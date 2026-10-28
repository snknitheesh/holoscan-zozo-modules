// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "expectations.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace yolo26_app {
namespace {

std::string_view trim(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r");
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = text.find_last_not_of(" \t\r");
  return text.substr(first, last - first + 1);
}

template <typename T>
bool parse(std::string_view text, T& value) {
  text = trim(text);
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc() && end == text.data() + text.size() && !text.empty();
}

}  // namespace

std::vector<Expectation> read_expectations(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error(path.string() + ": cannot open");
  }
  std::vector<Expectation> result;
  std::string line;
  for (int number = 1; std::getline(in, line); ++number) {
    const std::string_view text = trim(line);
    if (text.empty() || text.front() == '#') {
      continue;
    }
    std::vector<std::string_view> fields;
    std::size_t start = 0;
    while (true) {
      const std::size_t comma = text.find(',', start);
      fields.push_back(text.substr(start, comma == std::string_view::npos ? comma : comma - start));
      if (comma == std::string_view::npos) {
        break;
      }
      start = comma + 1;
    }
    Expectation e;
    const bool ok = fields.size() == 6 && !trim(fields[0]).empty() &&
                    parse(fields[1], e.box.class_id) && parse(fields[2], e.box.x1) &&
                    parse(fields[3], e.box.y1) && parse(fields[4], e.box.x2) &&
                    parse(fields[5], e.box.y2) && e.box.class_id >= 0 && e.box.x2 > e.box.x1 &&
                    e.box.y2 > e.box.y1;
    if (!ok) {
      throw std::runtime_error(path.string() + ":" + std::to_string(number) +
                               ": expected image,class_id,x1,y1,x2,y2 with x2 > x1, y2 > y1");
    }
    e.image = std::string(trim(fields[0]));
    result.push_back(std::move(e));
  }
  return result;
}

std::uint32_t count_matched(std::span<const BoxDetection> detections,
                            std::span<const Expectation> expected, float min_iou) {
  std::vector<bool> used(detections.size(), false);
  std::uint32_t matched = 0;
  for (const Expectation& e : expected) {
    // The best-overlapping unused detection of the class.
    std::size_t best = detections.size();
    float best_iou = min_iou;
    for (std::size_t i = 0; i < detections.size(); ++i) {
      if (used[i] || detections[i].class_id != e.box.class_id) {
        continue;
      }
      const float overlap = iou(detections[i], e.box);
      if (overlap >= best_iou) {
        best = i;
        best_iou = overlap;
      }
    }
    if (best != detections.size()) {
      used[best] = true;
      ++matched;
    }
  }
  return matched;
}

std::vector<std::string> read_labels(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error(path.string() + ": cannot open");
  }
  std::vector<std::string> labels;
  for (std::string line; std::getline(in, line);) {
    labels.emplace_back(trim(line));
  }
  while (!labels.empty() && labels.back().empty()) {
    labels.pop_back();
  }
  if (labels.empty()) {
    throw std::runtime_error(path.string() + ": no class names");
  }
  return labels;
}

}  // namespace yolo26_app
