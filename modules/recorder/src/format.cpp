// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/recorder/format.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "csv.hpp"
#include "files.hpp"
#include "json.hpp"

namespace holoscan::recorder {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view kFormatName = "holoscan-recording";
constexpr int kFormatVersion = 1;
constexpr std::string_view kMetadataFile = "recording.json";
constexpr std::string_view kIndexFile = "index.csv";
constexpr std::string_view kDataSuffix = ".bin";
// The index's columns, in order.
constexpr std::array<std::string_view, 22> kColumns = [] {
  return std::array<std::string_view, 22>{"stream",
                                          "sequence",
                                          "timestamp_ns",
                                          "file",
                                          "offset",
                                          "bytes",
                                          "frame_id",
                                          "dtype",
                                          "shape",
                                          "encoding",
                                          "width",
                                          "height",
                                          "color_space",
                                          "significant_bits",
                                          "roi_x",
                                          "roi_y",
                                          "planes",
                                          "exposure_ns",
                                          "gain",
                                          "header_timestamp_ns",
                                          "device_sequence",
                                          "header_frame_id"};
}();
// The first column holding image fields, and the first of the header's.
constexpr std::size_t kFirstImageColumn = 9;
constexpr std::size_t kFirstHeaderColumn = 19;

// A stereo camera recording's index starts with these columns, and may add
// "rgb" and "depth".
constexpr std::array<std::string_view, 5> kStereoColumns = [] {
  return std::array<std::string_view, 5>{"frame", "timestamp_ns", "device_sequence", "left",
                                         "right"};
}();

std::string join_columns() {
  std::string line;
  for (const std::string_view column : kColumns) {
    if (!line.empty()) line += ',';
    line += column;
  }
  return line;
}

std::string shape_text(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) {
    return "scalar";
  }
  std::string text;
  for (const std::int64_t extent : shape) {
    if (!text.empty()) text += 'x';
    text += std::to_string(extent);
  }
  return text;
}

std::vector<std::int64_t> parse_shape(std::string_view text) {
  std::vector<std::int64_t> shape;
  if (text == "scalar") {
    return shape;
  }
  while (true) {
    const std::size_t x = text.find('x');
    const std::int64_t extent = csv::to_i64(text.substr(0, x));
    if (extent < 0) {
      throw std::invalid_argument("a negative extent in the shape");
    }
    shape.push_back(extent);
    if (x == std::string_view::npos) break;
    text.remove_prefix(x + 1);
  }
  if (shape.size() > 32) {
    throw std::invalid_argument("a shape of more than 32 dimensions");
  }
  return shape;
}

std::string planes_text(const std::vector<PlaneLayout>& planes) {
  std::string text;
  for (const PlaneLayout& plane : planes) {
    if (!text.empty()) text += ';';
    text += std::to_string(plane.offset_bytes) + ':' + std::to_string(plane.row_stride_bytes) +
            ':' + std::to_string(plane.size_bytes);
  }
  return text;
}

std::vector<PlaneLayout> parse_planes(std::string_view text) {
  std::vector<PlaneLayout> planes;
  while (!text.empty()) {
    const std::size_t end = text.find(';');
    const std::string_view plane = text.substr(0, end);
    const std::size_t a = plane.find(':');
    const std::size_t b = a == std::string_view::npos ? a : plane.find(':', a + 1);
    if (b == std::string_view::npos) {
      throw std::invalid_argument("planes must be offset:row_stride:size, separated by ';'");
    }
    planes.push_back(PlaneLayout{.offset_bytes = csv::to_u64(plane.substr(0, a)),
                                 .row_stride_bytes = csv::to_u32(plane.substr(a + 1, b - a - 1)),
                                 .size_bytes = csv::to_u64(plane.substr(b + 1))});
    if (end == std::string_view::npos) break;
    text.remove_prefix(end + 1);
    if (text.empty()) {
      throw std::invalid_argument("planes end with ';'");
    }
  }
  return planes;
}

// An SDK enumerator name: capital letters, digits and '_'.
bool valid_enum_name(std::string_view name) {
  return !name.empty() && name.size() <= 32 && std::all_of(name.begin(), name.end(), [](char c) {
    return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
  });
}

// A file name inside the recording: no folders, not "." or "..".
bool plain_file_name(std::string_view name) {
  return !name.empty() && name != "." && name != ".." &&
         name.find_first_of("/\\") == std::string_view::npos && name.find('\0') == name.npos;
}

// A relative path that stays inside the recording.
bool inside_path(std::string_view text) {
  if (text.empty()) return false;
  const fs::path path(text);
  if (path.is_absolute()) return false;
  for (const fs::path& part : path) {
    if (part == "..") return false;
  }
  return true;
}

std::string_view trim(std::string_view text) {
  const auto space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  while (!text.empty() && space(text.front())) text.remove_prefix(1);
  while (!text.empty() && space(text.back())) text.remove_suffix(1);
  return text;
}

// Whether `a * b` overflows, else the product in `out`.
bool multiply(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return true;
  out = a * b;
  return false;
}

// The first bytes of a Netpbm (P5 or P6) file, and what they say.
struct Netpbm {
  int width{};
  int height{};
  int channels{};
  int bytes_per_sample{};
  std::uint64_t data_offset{};

  [[nodiscard]] std::uint64_t data_bytes() const {
    return static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) *
           static_cast<std::uint64_t>(channels) * static_cast<std::uint64_t>(bytes_per_sample);
  }
  friend bool operator==(const Netpbm&, const Netpbm&) = default;
};

Netpbm read_netpbm_header(const detail::File& file, const std::string& what) {
  std::array<char, 512> head{};
  std::size_t have = 0;
  while (have < head.size()) {
    const ssize_t got =
        ::pread(file.fd(), head.data() + have, head.size() - have, static_cast<off_t>(have));
    if (got < 0) {
      if (errno == EINTR) continue;
      throw RecordingError(what + ": cannot read: " + std::strerror(errno));
    }
    if (got == 0) break;
    have += static_cast<std::size_t>(got);
  }
  const std::string_view text(head.data(), have);
  if (text.size() < 2 || text[0] != 'P' || (text[1] != '5' && text[1] != '6')) {
    throw RecordingError(what + ": not a binary PGM (P5) or PPM (P6) image");
  }
  std::size_t position = 2;
  const auto space = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
  };
  const auto number = [&]() -> int {
    // White space and comments, then digits.
    while (position < text.size()) {
      if (space(text[position])) {
        ++position;
      } else if (text[position] == '#') {
        while (position < text.size() && text[position] != '\n') ++position;
      } else {
        break;
      }
    }
    const std::size_t start = position;
    while (position < text.size() && text[position] >= '0' && text[position] <= '9') ++position;
    if (position == start || position - start > 6) {
      throw RecordingError(what + ": a malformed Netpbm header");
    }
    return std::stoi(std::string(text.substr(start, position - start)));
  };
  Netpbm header;
  header.channels = text[1] == '6' ? 3 : 1;
  header.width = number();
  header.height = number();
  const int maxval = number();
  if (position >= text.size() || !space(text[position])) {
    throw RecordingError(what + ": a malformed Netpbm header");
  }
  ++position;
  if (header.width < 1 || header.height < 1 || header.width > 65535 || header.height > 65535 ||
      maxval < 1 || maxval > 65535) {
    throw RecordingError(what + ": an image size or maximum value out of range");
  }
  header.bytes_per_sample = maxval > 255 ? 2 : 1;
  header.data_offset = position;
  return header;
}

std::string netpbm_encoding(const Netpbm& header) {
  if (header.channels == 1) {
    return header.bytes_per_sample == 1 ? "MONO8" : "MONO16";
  }
  return header.bytes_per_sample == 1 ? "RGB8" : "RGB16";
}

std::string describe(const Netpbm& header) {
  return std::to_string(header.width) + "x" + std::to_string(header.height) + " " +
         (header.channels == 1 ? "P5" : "P6") + " " + std::to_string(8 * header.bytes_per_sample) +
         "-bit";
}

}  // namespace

// ---------------------------------------------------------------------------
// Element types, records and names

std::string_view stream_kind_name(StreamKind kind) noexcept {
  return kind == StreamKind::kImage ? "image" : "tensor";
}

std::uint64_t ElementType::bytes() const noexcept {
  return (static_cast<std::uint64_t>(bits) * lanes + 7U) / 8U;
}

std::string to_string(ElementType type) {
  std::string text;
  switch (type.code) {
    case 0:
      text = "int";
      break;
    case 1:
      text = "uint";
      break;
    case 2:
      text = "float";
      break;
    case 4:
      text = "bfloat";
      break;
    case 5:
      text = "complex";
      break;
    case 6:
      text = "bool";
      break;
    default:
      text = "code" + std::to_string(type.code) + "_";
      break;
  }
  text += std::to_string(type.bits);
  if (type.lanes != 1) {
    text += "x" + std::to_string(type.lanes);
  }
  return text;
}

ElementType parse_element_type(std::string_view text) {
  static constexpr std::array<std::pair<std::string_view, std::uint8_t>, 6> kCodes{{
      {"uint", 1},
      {"int", 0},
      {"float", 2},
      {"bfloat", 4},
      {"complex", 5},
      {"bool", 6},
  }};
  ElementType type;
  std::string_view rest;
  bool found = false;
  for (const auto& [name, code] : kCodes) {
    if (text.starts_with(name) && text.size() > name.size() && text[name.size()] >= '0' &&
        text[name.size()] <= '9') {
      type.code = code;
      rest = text.substr(name.size());
      found = true;
      break;
    }
  }
  if (!found && text.starts_with("code")) {
    const std::size_t underscore = text.find('_');
    if (underscore != std::string_view::npos) {
      try {
        const std::uint32_t code = csv::to_u32(text.substr(4, underscore - 4));
        if (code <= 255U) {
          type.code = static_cast<std::uint8_t>(code);
          rest = text.substr(underscore + 1);
          found = true;
        }
      } catch (const std::invalid_argument&) {
      }
    }
  }
  try {
    if (found) {
      const std::size_t x = rest.find('x');
      const std::uint32_t bits = csv::to_u32(rest.substr(0, x));
      const std::uint32_t lanes =
          x == std::string_view::npos ? 1U : csv::to_u32(rest.substr(x + 1));
      if (bits >= 1U && bits <= 255U && lanes >= 1U && lanes <= 65535U) {
        type.bits = static_cast<std::uint8_t>(bits);
        type.lanes = static_cast<std::uint16_t>(lanes);
        return type;
      }
    }
  } catch (const std::invalid_argument&) {
  }
  throw RecordingError("unknown element type \"" + std::string(text) +
                       "\"; expected uint8, float32, int16x4 and the like");
}

std::optional<std::uint64_t> Record::bytes() const noexcept {
  std::uint64_t total = dtype.bytes();
  for (const std::int64_t extent : shape) {
    if (extent < 0 || multiply(total, static_cast<std::uint64_t>(extent), total)) {
      return std::nullopt;
    }
  }
  return total;
}

bool valid_stream_name(std::string_view name) noexcept {
  return !name.empty() && name.size() <= 64 && name.front() != '.' &&
         std::all_of(name.begin(), name.end(), [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                  c == '_' || c == '-' || c == '.';
         });
}

// ---------------------------------------------------------------------------
// RecordingWriter

struct RecordingWriter::Impl {
  fs::path directory;
  std::vector<StreamDeclaration> streams;
  WriterOptions options;
  std::vector<detail::File> data;
  std::vector<std::uint64_t> sizes;
  std::vector<std::uint64_t> counts;
  std::vector<std::optional<std::uint64_t>> last_sequence;
  detail::File index;
  std::uint64_t records{};
  std::uint64_t bytes{};
  bool finished{};

  [[nodiscard]] std::string metadata_json(bool complete,
                                          std::span<const std::uint64_t> dropped) const {
    std::string out = "{\n";
    out += "  \"format\": " + json::quote(kFormatName) + ",\n";
    out += "  \"version\": " + std::to_string(kFormatVersion) + ",\n";
    out += std::string("  \"complete\": ") + (complete ? "true" : "false") + ",\n";
    out += "  \"streams\": [";
    for (std::size_t i = 0; i < streams.size(); ++i) {
      out += i == 0 ? "\n" : ",\n";
      out += "    {\"name\": " + json::quote(streams[i].name) +
             ", \"kind\": " + json::quote(stream_kind_name(streams[i].kind)) +
             ", \"file\": " + json::quote(streams[i].name + std::string(kDataSuffix)) +
             ", \"records\": " + std::to_string(counts[i]) +
             ", \"bytes\": " + std::to_string(sizes[i]) +
             ", \"dropped\": " + std::to_string(i < dropped.size() ? dropped[i] : 0U) + "}";
    }
    out += streams.empty() ? "],\n" : "\n  ],\n";
    out += "  \"calibration\": {";
    bool first = true;
    for (const auto& [name, blob] : options.calibration) {
      out += first ? "\n" : ",\n";
      first = false;
      out += "    " + json::quote(name) + ": " + std::string(trim(blob));
    }
    out += first ? "},\n" : "\n  },\n";
    out += "  \"metadata\": {";
    first = true;
    for (const auto& [key, value] : options.metadata) {
      out += first ? "\n" : ",\n";
      first = false;
      out += "    " + json::quote(key) + ": " + json::quote(value);
    }
    out += first ? "}\n" : "\n  }\n";
    out += "}\n";
    return out;
  }
};

RecordingWriter::RecordingWriter(fs::path directory, std::vector<StreamDeclaration> streams,
                                 WriterOptions options)
    : impl_(std::make_unique<Impl>()) {
  Impl& w = *impl_;
  w.directory = std::move(directory);
  w.streams = std::move(streams);
  w.options = std::move(options);
  if (w.directory.empty()) {
    throw RecordingError("the recording needs a folder");
  }
  if (w.streams.empty()) {
    throw RecordingError("a recording needs at least one stream");
  }
  for (std::size_t i = 0; i < w.streams.size(); ++i) {
    const std::string& name = w.streams[i].name;
    if (!valid_stream_name(name)) {
      throw RecordingError("stream name \"" + name +
                           "\" is not 1 to 64 letters, digits, '_', '-' and '.', not starting "
                           "with '.'");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (w.streams[j].name == name) {
        throw RecordingError("two streams are named \"" + name + "\"");
      }
    }
  }
  for (const auto& [name, blob] : w.options.calibration) {
    if (name.empty()) {
      throw RecordingError("a calibration blob needs a name");
    }
    try {
      (void)json::parse(blob);
    } catch (const std::invalid_argument& error) {
      throw RecordingError("calibration \"" + name + "\" is not JSON: " + error.what());
    }
  }

  // Never write into a folder that holds anything but a recording.
  std::error_code error;
  if (fs::exists(w.directory, error)) {
    if (!fs::is_directory(w.directory, error)) {
      throw RecordingError(w.directory.string() + ": not a folder");
    }
    std::vector<fs::path> previous;
    for (const fs::directory_entry& entry : fs::directory_iterator(w.directory, error)) {
      const std::string name = entry.path().filename().string();
      const bool ours = name == kMetadataFile || name == kIndexFile ||
                        name == std::string(kMetadataFile) + ".tmp" ||
                        (name.ends_with(kDataSuffix) && entry.is_regular_file());
      if (!ours) {
        throw RecordingError(w.directory.string() + " holds " + name +
                             ", which is not part of a recording; record into an empty or new "
                             "folder");
      }
      previous.push_back(entry.path());
    }
    if (error) {
      throw RecordingError(w.directory.string() + ": cannot list: " + error.message());
    }
    if (!previous.empty() && !w.options.overwrite) {
      throw RecordingError(w.directory.string() +
                           " already holds a recording; remove it or set overwrite");
    }
    for (const fs::path& path : previous) {
      fs::remove(path, error);
      if (error) {
        throw RecordingError(path.string() + ": cannot remove: " + error.message());
      }
    }
  } else if (!fs::create_directories(w.directory, error) && error) {
    throw RecordingError(w.directory.string() + ": cannot create: " + error.message());
  }

  const std::size_t n = w.streams.size();
  w.sizes.assign(n, 0);
  w.counts.assign(n, 0);
  w.last_sequence.assign(n, std::nullopt);
  detail::replace_file(w.directory / kMetadataFile, w.metadata_json(false, {}));
  for (const StreamDeclaration& stream : w.streams) {
    w.data.push_back(detail::File::create(w.directory / (stream.name + std::string(kDataSuffix))));
  }
  w.index = detail::File::create(w.directory / kIndexFile);
  const std::string header = join_columns() + "\n";
  w.index.write_all(std::as_bytes(std::span(header)), (w.directory / kIndexFile).string());
}

RecordingWriter::~RecordingWriter() = default;

void RecordingWriter::append(Record record, std::span<const std::byte> data) {
  Impl& w = *impl_;
  if (w.finished) {
    throw RecordingError(w.directory.string() + ": the recording is finished");
  }
  if (record.stream >= w.streams.size()) {
    throw RecordingError("record for stream " + std::to_string(record.stream) + " of " +
                         std::to_string(w.streams.size()));
  }
  const StreamDeclaration& stream = w.streams[record.stream];
  const bool image = stream.kind == StreamKind::kImage;
  if (image != record.image.has_value()) {
    throw RecordingError("stream \"" + stream.name + "\" holds " +
                         std::string(stream_kind_name(stream.kind)) + "s; the record " +
                         (image ? "has no image descriptor" : "has an image descriptor"));
  }
  if (w.last_sequence[record.stream] && record.sequence <= *w.last_sequence[record.stream]) {
    throw RecordingError("stream \"" + stream.name + "\": sequence " +
                         std::to_string(record.sequence) + " does not follow " +
                         std::to_string(*w.last_sequence[record.stream]));
  }
  const std::optional<std::uint64_t> bytes = record.bytes();
  if (!bytes || *bytes != data.size()) {
    throw RecordingError("stream \"" + stream.name + "\": " + std::to_string(data.size()) +
                         " bytes of data for a " + to_string(record.dtype) + " tensor of shape " +
                         shape_text(record.shape));
  }
  if (record.dtype.bits == 0 || record.dtype.lanes == 0) {
    throw RecordingError("stream \"" + stream.name + "\": an element type of no size");
  }
  if (image) {
    const ImageDescriptor& d = *record.image;
    if (!valid_enum_name(d.encoding) || !valid_enum_name(d.color_space)) {
      throw RecordingError("stream \"" + stream.name +
                           "\": the encoding and color space must be SDK enumerator names");
    }
  }
  record.file = stream.name + std::string(kDataSuffix);
  record.offset = w.sizes[record.stream];

  // The data first: an index line never names data that is not there.
  w.data[record.stream].write_all(data, (w.directory / record.file).string());
  w.sizes[record.stream] += data.size();

  std::string line;
  line.reserve(160);
  const auto add = [&line](std::string_view text) {
    line += text;
    line += ',';
  };
  add(stream.name);
  add(std::to_string(record.sequence));
  add(std::to_string(record.timestamp_ns));
  add(record.file);
  add(std::to_string(record.offset));
  add(std::to_string(data.size()));
  add(std::to_string(record.frame_id));
  add(to_string(record.dtype));
  add(shape_text(record.shape));
  if (image) {
    const ImageDescriptor& d = *record.image;
    add(d.encoding);
    add(std::to_string(d.width));
    add(std::to_string(d.height));
    add(d.color_space);
    add(std::to_string(d.significant_bits));
    add(std::to_string(d.roi_offset_x));
    add(std::to_string(d.roi_offset_y));
    add(planes_text(d.plane_layouts));
    add(std::to_string(d.exposure_time_ns));
    add(csv::from_float(d.gain));
    if (d.has_header) {
      add(std::to_string(d.header_capture_timestamp_ns));
      add(std::to_string(d.header_device_sequence));
      // Quoted, so that an empty frame id is told apart from no header.
      line += '"';
      for (const char c : d.header_frame_id) {
        if (c == '"') line += '"';
        line += c;
      }
      line += '"';
    } else {
      line += ",,";
    }
  } else {
    line.append(kColumns.size() - kFirstImageColumn - 1, ',');
  }
  line += '\n';
  w.index.write_all(std::as_bytes(std::span(line)), (w.directory / kIndexFile).string());
  w.last_sequence[record.stream] = record.sequence;
  ++w.counts[record.stream];
  ++w.records;
  w.bytes += data.size();
}

void RecordingWriter::finish(std::span<const std::uint64_t> dropped) {
  Impl& w = *impl_;
  if (w.finished) {
    return;
  }
  for (std::size_t i = 0; i < w.data.size(); ++i) {
    w.data[i].sync((w.directory / (w.streams[i].name + std::string(kDataSuffix))).string());
  }
  w.index.sync((w.directory / kIndexFile).string());
  detail::replace_file(w.directory / kMetadataFile, w.metadata_json(true, dropped));
  w.finished = true;
  for (detail::File& file : w.data) file.close();
  w.index.close();
}

const fs::path& RecordingWriter::directory() const noexcept { return impl_->directory; }
std::uint64_t RecordingWriter::records() const noexcept { return impl_->records; }
std::uint64_t RecordingWriter::bytes() const noexcept { return impl_->bytes; }

// ---------------------------------------------------------------------------
// RecordingReader

struct RecordingReader::Impl {
  RecordingFormat format{RecordingFormat::kNative};
  fs::path directory;
  std::vector<StreamInfo> streams;
  std::vector<Record> records;
  std::map<std::string, std::string> calibration;
  std::map<std::string, std::string> metadata;
  bool complete{};
  // Native: each stream's data file. A stereo camera recording opens each image
  // as it is read.
  std::vector<detail::File> data;
  std::vector<std::uint64_t> data_sizes;
  // Stereo camera: each stream's image header, from its first image.
  std::vector<Netpbm> netpbm;

  void load_native();
  void load_stereo();
  void add_to_stream(const Record& record, std::uint64_t bytes);
};

void RecordingReader::Impl::add_to_stream(const Record& record, std::uint64_t bytes) {
  StreamInfo& info = streams[record.stream];
  if (info.records == 0) {
    info.dtype = record.dtype;
    info.rank = static_cast<std::uint32_t>(record.shape.size());
  } else {
    // A stream whose records differ in element type or rank has neither.
    if (info.dtype && *info.dtype != record.dtype) info.dtype.reset();
    if (info.rank && *info.rank != record.shape.size()) info.rank.reset();
  }
  ++info.records;
  info.max_bytes = std::max(info.max_bytes, bytes);
}

void RecordingReader::Impl::load_native() {
  const fs::path metadata_path = directory / kMetadataFile;
  const std::string where = metadata_path.string();
  json::Value doc;
  try {
    doc = json::parse(detail::read_text(metadata_path));
  } catch (const std::invalid_argument& error) {
    throw RecordingError(where + ": " + error.what());
  }
  const auto member = [&](const json::Value& object, std::string_view key, json::Value::Kind kind,
                          bool required) -> const json::Value* {
    const json::Value* value = object.find(key);
    if (value == nullptr) {
      if (required) throw RecordingError(where + ": needs \"" + std::string(key) + "\"");
      return nullptr;
    }
    if (value->kind != kind) {
      throw RecordingError(where + ": \"" + std::string(key) + "\" must be " +
                           json::kind_name(kind) + ", not " + json::kind_name(value->kind));
    }
    return value;
  };
  if (doc.kind != json::Value::Kind::kObject) {
    throw RecordingError(where + ": must hold an object");
  }
  if (member(doc, "format", json::Value::Kind::kString, true)->string != kFormatName) {
    throw RecordingError(where + ": not a " + std::string(kFormatName) + " (format is \"" +
                         doc.find("format")->string + "\")");
  }
  const double version = member(doc, "version", json::Value::Kind::kNumber, true)->number;
  if (version != kFormatVersion) {
    throw RecordingError(where + ": format version " + json::number(version) +
                         "; this reader reads version " + std::to_string(kFormatVersion));
  }
  if (const json::Value* value = member(doc, "complete", json::Value::Kind::kBool, false)) {
    complete = value->boolean;
  }
  const json::Value* list = member(doc, "streams", json::Value::Kind::kArray, true);
  for (const json::Value& entry : list->items) {
    if (entry.kind != json::Value::Kind::kObject) {
      throw RecordingError(where + ": each of \"streams\" must be an object");
    }
    StreamInfo info;
    info.name = member(entry, "name", json::Value::Kind::kString, true)->string;
    if (!valid_stream_name(info.name)) {
      throw RecordingError(where + ": stream name \"" + info.name + "\" is not valid");
    }
    for (const StreamInfo& other : streams) {
      if (other.name == info.name) {
        throw RecordingError(where + ": two streams are named \"" + info.name + "\"");
      }
    }
    const std::string& kind = member(entry, "kind", json::Value::Kind::kString, true)->string;
    if (kind == "image") {
      info.kind = StreamKind::kImage;
    } else if (kind == "tensor") {
      info.kind = StreamKind::kTensor;
    } else {
      throw RecordingError(where + ": stream \"" + info.name + "\" has kind \"" + kind +
                           "\"; expected image or tensor");
    }
    const std::string& file = member(entry, "file", json::Value::Kind::kString, true)->string;
    if (!plain_file_name(file)) {
      throw RecordingError(where + ": stream \"" + info.name + "\" names file \"" + file +
                           "\", which is not a file of the recording's folder");
    }
    if (const json::Value* value = member(entry, "dropped", json::Value::Kind::kNumber, false)) {
      if (!(value->number >= 0) || value->number > 9.0e15) {
        throw RecordingError(where + ": stream \"" + info.name + "\" has a bad dropped count");
      }
      info.dropped = static_cast<std::uint64_t>(value->number);
    }
    const fs::path path = directory / file;
    data.push_back(detail::File::open_read(path));
    std::error_code error;
    data_sizes.push_back(fs::file_size(path, error));
    if (error) {
      throw RecordingError(path.string() + ": " + error.message());
    }
    streams.push_back(std::move(info));
  }
  // Remember each stream's file name, to check the index against.
  std::vector<std::string> files;
  for (const json::Value& entry : list->items) {
    files.push_back(entry.find("file")->string);
  }
  if (const json::Value* blobs = member(doc, "calibration", json::Value::Kind::kObject, false)) {
    for (std::size_t i = 0; i < blobs->keys.size(); ++i) {
      calibration[blobs->keys[i]] = json::dump(blobs->items[i]);
    }
  }
  if (const json::Value* values = member(doc, "metadata", json::Value::Kind::kObject, false)) {
    for (std::size_t i = 0; i < values->keys.size(); ++i) {
      if (values->items[i].kind != json::Value::Kind::kString) {
        throw RecordingError(where + ": metadata \"" + values->keys[i] + "\" must be a string");
      }
      metadata[values->keys[i]] = values->items[i].string;
    }
  }

  // The index.
  const fs::path index_path = directory / kIndexFile;
  const std::string index_where = index_path.string();
  csv::Table table;
  try {
    table = csv::parse(detail::read_text(index_path));
  } catch (const std::invalid_argument& error) {
    throw RecordingError(index_where + ": " + error.what());
  }
  if (table.rows.empty() ||
      !std::equal(table.rows.front().fields.begin(), table.rows.front().fields.end(),
                  kColumns.begin(), kColumns.end())) {
    throw RecordingError(index_where + ": the first line must be the header " + join_columns());
  }
  std::vector<std::optional<std::uint64_t>> last_sequence(streams.size());
  for (std::size_t r = 1; r < table.rows.size(); ++r) {
    const csv::Row& row = table.rows[r];
    const std::string at = index_where + ": line " + std::to_string(row.line) + ": ";
    if (r + 1 == table.rows.size() && !table.terminated) {
      if (complete) {
        throw RecordingError(at + "the last line is cut short");
      }
      break;  // The recorder stopped while writing it.
    }
    const std::vector<std::string>& f = row.fields;
    if (f.size() != kColumns.size()) {
      throw RecordingError(at + std::to_string(f.size()) + " fields; expected " +
                           std::to_string(kColumns.size()));
    }
    Record record;
    std::uint64_t bytes = 0;
    try {
      const auto stream = std::find_if(streams.begin(), streams.end(),
                                       [&](const StreamInfo& s) { return s.name == f[0]; });
      if (stream == streams.end()) {
        throw std::invalid_argument("stream \"" + f[0] + "\" is not in recording.json");
      }
      record.stream = static_cast<std::uint32_t>(stream - streams.begin());
      record.sequence = csv::to_u64(f[1]);
      record.timestamp_ns = csv::to_i64(f[2]);
      record.file = f[3];
      if (record.file != files[record.stream]) {
        throw std::invalid_argument("file \"" + f[3] + "\" is not stream \"" + f[0] + "\"'s");
      }
      record.offset = csv::to_u64(f[4]);
      bytes = csv::to_u64(f[5]);
      record.frame_id = csv::to_u64(f[6]);
      try {
        record.dtype = parse_element_type(f[7]);
      } catch (const RecordingError& error) {
        throw std::invalid_argument(error.what());
      }
      record.shape = parse_shape(f[8]);
      const std::optional<std::uint64_t> expected = record.bytes();
      if (!expected || *expected != bytes) {
        throw std::invalid_argument("bytes " + f[5] + " do not match a " + f[7] +
                                    " tensor of shape " + f[8]);
      }
      if (record.offset > data_sizes[record.stream] ||
          bytes > data_sizes[record.stream] - record.offset) {
        throw std::invalid_argument("the data, " + f[5] + " bytes at " + f[4] + ", runs past the " +
                                    std::to_string(data_sizes[record.stream]) + " bytes of " +
                                    record.file);
      }
      if (last_sequence[record.stream] && record.sequence <= *last_sequence[record.stream]) {
        throw std::invalid_argument("sequence " + f[1] + " of stream \"" + f[0] +
                                    "\" does not follow " +
                                    std::to_string(*last_sequence[record.stream]));
      }
      if (streams[record.stream].kind == StreamKind::kImage) {
        ImageDescriptor d;
        d.encoding = f[9];
        d.width = csv::to_i32(f[10]);
        d.height = csv::to_i32(f[11]);
        d.color_space = f[12];
        const std::uint32_t bits = csv::to_u32(f[13]);
        if (bits > 255U) throw std::invalid_argument("significant_bits above 255");
        d.significant_bits = static_cast<std::uint8_t>(bits);
        d.roi_offset_x = csv::to_i32(f[14]);
        d.roi_offset_y = csv::to_i32(f[15]);
        d.plane_layouts = parse_planes(f[16]);
        d.exposure_time_ns = csv::to_i64(f[17]);
        d.gain = csv::to_float(f[18]);
        if (!valid_enum_name(d.encoding) || !valid_enum_name(d.color_space)) {
          throw std::invalid_argument("encoding and color_space must be SDK enumerator names");
        }
        if (d.width < 0 || d.height < 0) {
          throw std::invalid_argument("a negative image size");
        }
        d.has_header = !f[kFirstHeaderColumn].empty();
        if (d.has_header) {
          d.header_capture_timestamp_ns = csv::to_i64(f[19]);
          d.header_device_sequence = csv::to_u64(f[20]);
          d.header_frame_id = f[21];
        } else if (!f[20].empty() || !f[21].empty()) {
          throw std::invalid_argument("header fields without a header_timestamp_ns");
        }
        record.image = std::move(d);
      } else {
        for (std::size_t c = kFirstImageColumn; c < f.size(); ++c) {
          if (!f[c].empty()) {
            throw std::invalid_argument("stream \"" + f[0] + "\" holds tensors, but column " +
                                        std::string(kColumns[c]) + " has a value");
          }
        }
      }
    } catch (const std::invalid_argument& error) {
      throw RecordingError(at + error.what());
    }
    last_sequence[record.stream] = record.sequence;
    add_to_stream(record, bytes);
    records.push_back(std::move(record));
  }
}

void RecordingReader::Impl::load_stereo() {
  format = RecordingFormat::kStereoCamera;
  complete = true;
  const fs::path calibration_path = directory / "calibration.json";
  try {
    const json::Value doc = json::parse(detail::read_text(calibration_path));
    if (doc.kind != json::Value::Kind::kObject) {
      throw std::invalid_argument("must hold an object");
    }
    calibration["stereo_camera"] = json::dump(doc);
  } catch (const std::invalid_argument& error) {
    throw RecordingError(calibration_path.string() + ": " + error.what());
  }

  const fs::path index_path = directory / kIndexFile;
  const std::string index_where = index_path.string();
  csv::Table table;
  try {
    table = csv::parse(detail::read_text(index_path));
  } catch (const std::invalid_argument& error) {
    throw RecordingError(index_where + ": " + error.what());
  }
  // The header: frame,timestamp_ns,device_sequence,left,right, then rgb and depth if recorded.
  const std::vector<std::string> header =
      table.rows.empty() ? std::vector<std::string>{} : table.rows.front().fields;
  if (header.size() < kStereoColumns.size() ||
      !std::equal(kStereoColumns.begin(), kStereoColumns.end(), header.begin())) {
    throw RecordingError(index_where +
                         ": the first line must be frame,timestamp_ns,device_sequence,left,right "
                         "and optionally rgb and depth");
  }
  for (std::size_t c = 3; c < header.size(); ++c) {
    if (c >= kStereoColumns.size() && header[c] != "rgb" && header[c] != "depth") {
      throw RecordingError(index_where + ": unknown column \"" + header[c] +
                           "\"; a stereo camera recording has rgb and depth");
    }
    for (std::size_t d = 3; d < c; ++d) {
      if (header[d] == header[c]) {
        throw RecordingError(index_where + ": column \"" + header[c] + "\" appears twice");
      }
    }
    StreamInfo info;
    info.name = header[c];
    streams.push_back(std::move(info));
  }
  netpbm.resize(streams.size());
  std::optional<std::uint64_t> last_frame;
  std::optional<std::int64_t> last_timestamp;
  for (std::size_t r = 1; r < table.rows.size(); ++r) {
    const csv::Row& row = table.rows[r];
    const std::string at = index_where + ": line " + std::to_string(row.line) + ": ";
    const std::vector<std::string>& f = row.fields;
    if (r + 1 == table.rows.size() && !table.terminated) {
      throw RecordingError(at + "the last line is cut short");
    }
    if (f.size() != header.size()) {
      throw RecordingError(at + std::to_string(f.size()) + " fields; expected " +
                           std::to_string(header.size()));
    }
    std::uint64_t frame = 0;
    std::int64_t timestamp = 0;
    std::uint64_t device_sequence = 0;
    try {
      frame = csv::to_u64(f[0]);
      timestamp = csv::to_i64(f[1]);
      device_sequence = csv::to_u64(f[2]);
    } catch (const std::invalid_argument& error) {
      throw RecordingError(at + error.what());
    }
    if (last_frame && frame <= *last_frame) {
      throw RecordingError(at + "frame " + f[0] + " does not follow " +
                           std::to_string(*last_frame));
    }
    if (last_timestamp && timestamp < *last_timestamp) {
      throw RecordingError(at + "timestamp_ns " + f[1] + " goes back in time");
    }
    last_frame = frame;
    last_timestamp = timestamp;
    for (std::size_t s = 0; s < streams.size(); ++s) {
      const std::string& file = f[3 + s];
      if (!inside_path(file)) {
        throw RecordingError(at + "\"" + file + "\" is not a path inside the recording");
      }
      const fs::path path = directory / file;
      std::error_code error;
      if (!fs::is_regular_file(path, error)) {
        throw RecordingError(at + path.string() + " is missing");
      }
      if (streams[s].records == 0) {
        netpbm[s] = read_netpbm_header(detail::File::open_read(path), path.string());
      }
      const Netpbm& image = netpbm[s];
      Record record;
      record.stream = static_cast<std::uint32_t>(s);
      record.sequence = frame;
      record.timestamp_ns = timestamp;
      record.frame_id = frame + 1;
      record.dtype = ElementType{
          .code = 1, .bits = static_cast<std::uint8_t>(8 * image.bytes_per_sample), .lanes = 1};
      record.shape = {image.height, image.width};
      if (image.channels == 3) record.shape.push_back(3);
      ImageDescriptor descriptor;
      descriptor.width = image.width;
      descriptor.height = image.height;
      descriptor.encoding = netpbm_encoding(image);
      descriptor.has_header = true;
      descriptor.header_frame_id = streams[s].name;
      descriptor.header_capture_timestamp_ns = timestamp;
      descriptor.header_device_sequence = device_sequence;
      record.image = std::move(descriptor);
      record.file = file;
      record.offset = image.data_offset;
      add_to_stream(record, image.data_bytes());
      records.push_back(std::move(record));
    }
  }
}

RecordingReader::RecordingReader(const fs::path& directory) : impl_(std::make_unique<Impl>()) {
  impl_->directory = directory;
  std::error_code error;
  if (!fs::is_directory(directory, error)) {
    throw RecordingError(directory.string() + ": no such folder");
  }
  if (fs::exists(directory / kMetadataFile, error)) {
    impl_->load_native();
  } else if (fs::exists(directory / "calibration.json", error) &&
             fs::exists(directory / kIndexFile, error)) {
    impl_->load_stereo();
  } else {
    throw RecordingError(directory.string() +
                         ": not a recording: it has neither recording.json nor a stereo camera "
                         "recording's calibration.json and index.csv");
  }
}

RecordingReader::~RecordingReader() = default;
RecordingReader::RecordingReader(RecordingReader&&) noexcept = default;
RecordingReader& RecordingReader::operator=(RecordingReader&&) noexcept = default;

RecordingFormat RecordingReader::format() const noexcept { return impl_->format; }
const fs::path& RecordingReader::directory() const noexcept { return impl_->directory; }
const std::vector<StreamInfo>& RecordingReader::streams() const noexcept { return impl_->streams; }
const std::vector<Record>& RecordingReader::records() const noexcept { return impl_->records; }
const std::map<std::string, std::string>& RecordingReader::calibration() const noexcept {
  return impl_->calibration;
}
const std::map<std::string, std::string>& RecordingReader::metadata() const noexcept {
  return impl_->metadata;
}
bool RecordingReader::complete() const noexcept { return impl_->complete; }

std::optional<std::uint32_t> RecordingReader::find_stream(std::string_view name) const noexcept {
  for (std::size_t i = 0; i < impl_->streams.size(); ++i) {
    if (impl_->streams[i].name == name) return static_cast<std::uint32_t>(i);
  }
  return std::nullopt;
}

void RecordingReader::read(const Record& record, std::span<std::byte> out) const {
  const Impl& r = *impl_;
  const std::optional<std::uint64_t> bytes = record.bytes();
  if (record.stream >= r.streams.size() || !bytes || *bytes != out.size()) {
    throw RecordingError(r.directory.string() + ": read() of a record that is not this " +
                         "recording's, or into a buffer of the wrong size");
  }
  const std::string what = (r.directory / record.file).string();
  if (r.format == RecordingFormat::kNative) {
    r.data[record.stream].read_at(record.offset, out, what);
    return;
  }
  // A stereo camera image: its header must say what the stream's first one did.
  const detail::File file = detail::File::open_read(r.directory / record.file);
  const Netpbm header = read_netpbm_header(file, what);
  const Netpbm& expected = r.netpbm[record.stream];
  if (header.width != expected.width || header.height != expected.height ||
      header.channels != expected.channels ||
      header.bytes_per_sample != expected.bytes_per_sample) {
    throw RecordingError(what + ": a " + describe(header) + " image in a stream of " +
                         describe(expected) + " images");
  }
  file.read_at(header.data_offset, out, what);
  if (header.bytes_per_sample == 2 && std::endian::native == std::endian::little) {
    // Netpbm stores 16-bit samples big-endian.
    for (std::size_t i = 0; i + 1 < out.size(); i += 2) {
      std::swap(out[i], out[i + 1]);
    }
  }
}

}  // namespace holoscan::recorder
