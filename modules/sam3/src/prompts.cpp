// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>

#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <holoscan/sam3/prompts.hpp>

namespace holoscan::sam3 {
namespace {

static_assert(std::endian::native == std::endian::little,
              "the prompt table is little-endian; this reader copies host bytes");

constexpr char kMagic[8] = {'S', 'A', 'M', '3', 'P', 'R', 'M', '1'};
// Generous bounds that keep a damaged header from asking for gigabytes.
constexpr std::uint32_t kMaxEntries = 65536;
constexpr std::uint32_t kMaxTokens = 4096;
constexpr std::uint32_t kMaxDim = 65536;

class Reader {
 public:
  explicit Reader(const std::filesystem::path& path) : path_(path), in_(path, std::ios::binary) {
    if (!in_) {
      fail("cannot open it");
    }
  }
  void bytes(void* out, std::size_t count) {
    if (!in_.read(static_cast<char*>(out), static_cast<std::streamsize>(count))) {
      fail("it is truncated");
    }
  }
  std::uint32_t u32() {
    std::uint32_t value = 0;
    bytes(&value, sizeof(value));
    return value;
  }
  void floats(std::vector<float>& out, std::size_t count) {
    out.resize(count);
    bytes(out.data(), count * sizeof(float));
  }
  bool at_end() { return in_.peek() == std::ifstream::traits_type::eof(); }
  [[noreturn]] void fail(const std::string& why) const {
    throw std::runtime_error("prompt table " + path_.string() + ": " + why);
  }

 private:
  std::filesystem::path path_;
  std::ifstream in_;
};

bool unit(float value) { return std::isfinite(value) && value >= 0.0F && value <= 1.0F; }

}  // namespace

std::string_view table_text(const Sam3Prompt& prompt) noexcept {
  return prompt.text.empty() ? kVisualPrompt : std::string_view(prompt.text);
}

std::string display_name(const Sam3Prompt& prompt) {
  return prompt.text.empty() ? std::string("box") : prompt.text;
}

void validate_prompt(const Sam3Prompt& prompt) {
  if (prompt.text.empty() && prompt.boxes.empty()) {
    throw std::invalid_argument("a prompt needs a text, boxes or both");
  }
  if (prompt.text.size() > kMaxPromptTextBytes) {
    throw std::invalid_argument("prompt texts are at most " + std::to_string(kMaxPromptTextBytes) +
                                " bytes");
  }
  for (const PromptBox& box : prompt.boxes) {
    const auto& b = box.xyxy;
    if (!unit(b[0]) || !unit(b[1]) || !unit(b[2]) || !unit(b[3]) || !(b[0] < b[2]) ||
        !(b[1] < b[3])) {
      throw std::invalid_argument(
          "prompt boxes are [x1, y1, x2, y2] in fractions of the image, within [0, 1], with "
          "x1 < x2 and y1 < y2");
    }
  }
}

const TextEmbedding* PromptTable::find(std::string_view text) const noexcept {
  for (const TextEmbedding& entry : entries) {
    if (entry.text == text) {
      return &entry;
    }
  }
  return nullptr;
}

PromptTable read_prompt_table(const std::filesystem::path& path) {
  Reader reader(path);
  char magic[sizeof(kMagic)];
  reader.bytes(magic, sizeof(magic));
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    reader.fail("not a prompt table (written by tools/prepare_model.py)");
  }
  const std::uint32_t count = reader.u32();
  const std::uint32_t tokens = reader.u32();
  const std::uint32_t dim = reader.u32();
  if (count > kMaxEntries || tokens < 1 || tokens > kMaxTokens || dim < 1 || dim > kMaxDim) {
    reader.fail("its header is out of range");
  }
  PromptTable table{
      .tokens = static_cast<int>(tokens), .dim = static_cast<int>(dim), .entries = {}};
  std::set<std::string> seen;
  table.entries.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    TextEmbedding entry;
    const std::uint32_t length = reader.u32();
    if (length == 0 || length > kMaxPromptTextBytes) {
      reader.fail("an entry's text is empty or too long");
    }
    entry.text.resize(length);
    reader.bytes(entry.text.data(), length);
    reader.floats(entry.mask, tokens);
    reader.floats(entry.features, static_cast<std::size_t>(tokens) * dim);
    if (!seen.insert(entry.text).second) {
      reader.fail("it has '" + entry.text + "' twice");
    }
    table.entries.push_back(std::move(entry));
  }
  if (!reader.at_end()) {
    reader.fail("it has data after its last entry");
  }
  return table;
}

void write_prompt_table(const std::filesystem::path& path, const PromptTable& table) {
  if (table.tokens < 1 || table.dim < 1 || table.entries.size() > kMaxEntries) {
    throw std::invalid_argument("a prompt table needs positive tokens and dim");
  }
  std::set<std::string> seen;
  std::string bytes(kMagic, sizeof(kMagic));
  const auto put_u32 = [&bytes](std::uint32_t value) {
    bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
  };
  const auto put_floats = [&bytes](const std::vector<float>& values) {
    bytes.append(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
  };
  put_u32(static_cast<std::uint32_t>(table.entries.size()));
  put_u32(static_cast<std::uint32_t>(table.tokens));
  put_u32(static_cast<std::uint32_t>(table.dim));
  for (const TextEmbedding& entry : table.entries) {
    if (entry.text.empty() || entry.text.size() > kMaxPromptTextBytes ||
        !seen.insert(entry.text).second ||
        entry.mask.size() != static_cast<std::size_t>(table.tokens) ||
        entry.features.size() != static_cast<std::size_t>(table.tokens) * table.dim) {
      throw std::invalid_argument("prompt table entry '" + entry.text +
                                  "' is empty, repeated or of the wrong size");
    }
    put_u32(static_cast<std::uint32_t>(entry.text.size()));
    bytes += entry.text;
    put_floats(entry.mask);
    put_floats(entry.features);
  }
  static std::atomic<unsigned> sequence{0};
  const std::filesystem::path temporary =
      path.parent_path() / (path.filename().string() + ".tmp." + std::to_string(::getpid()) + "." +
                            std::to_string(sequence.fetch_add(1)));
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out.flush()) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      throw std::runtime_error("cannot write the prompt table to " + temporary.string());
    }
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    throw std::runtime_error("cannot write the prompt table to " + path.string());
  }
}

DecoderPrompt make_decoder_prompt(const Sam3Prompt& prompt, const PromptTable& table,
                                  int max_boxes) {
  validate_prompt(prompt);
  if (max_boxes < 0 || prompt.boxes.size() > static_cast<std::size_t>(max_boxes)) {
    throw std::invalid_argument("a prompt has " + std::to_string(prompt.boxes.size()) +
                                " boxes; the decoder takes at most " + std::to_string(max_boxes));
  }
  const std::string_view text = table_text(prompt);
  const TextEmbedding* entry = table.find(text);
  if (entry == nullptr) {
    throw std::invalid_argument("prompt '" + std::string(text) +
                                "' is not in the prompt table; add it with "
                                "tools/prepare_model.py --prompts");
  }
  DecoderPrompt result;
  result.text_features = entry->features;
  result.text_mask = entry->mask;
  const auto slots = static_cast<std::size_t>(max_boxes);
  result.boxes.assign(slots * 4U, 0.0F);
  result.box_labels.assign(slots, 0.0F);
  result.box_valid.assign(slots, 0.0F);
  for (std::size_t i = 0; i < prompt.boxes.size(); ++i) {
    const auto& b = prompt.boxes[i].xyxy;
    result.boxes[4 * i] = (b[0] + b[2]) * 0.5F;
    result.boxes[4 * i + 1] = (b[1] + b[3]) * 0.5F;
    result.boxes[4 * i + 2] = b[2] - b[0];
    result.boxes[4 * i + 3] = b[3] - b[1];
    result.box_labels[i] = prompt.boxes[i].positive ? 1.0F : 0.0F;
    result.box_valid[i] = 1.0F;
  }
  return result;
}

void validate_prompts(const std::vector<Sam3Prompt>& prompts, const PromptTable& table) {
  if (prompts.empty() || prompts.size() > static_cast<std::size_t>(kMaxPrompts)) {
    throw std::invalid_argument("1.." + std::to_string(kMaxPrompts) + " prompts are needed");
  }
  for (const Sam3Prompt& prompt : prompts) {
    (void)make_decoder_prompt(prompt, table, static_cast<int>(prompt.boxes.size()));
  }
}

Sam3PromptControl::Sam3PromptControl(PromptTable table) : table_(std::move(table)) {}

void Sam3PromptControl::set(std::vector<Sam3Prompt> prompts) {
  validate_prompts(prompts, table_);
  const std::lock_guard<std::mutex> lock(mutex_);
  prompts_ = std::move(prompts);
  ++generation_;
}

std::pair<std::uint64_t, std::vector<Sam3Prompt>> Sam3PromptControl::latest() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return {generation_, prompts_};
}

}  // namespace holoscan::sam3
