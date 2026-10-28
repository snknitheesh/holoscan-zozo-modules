// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_camera/recording.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <holoscan/stereo_camera/netpbm.hpp>

namespace holoscan::stereo_camera {
namespace {

// The columns every index starts with.
constexpr std::string_view kRequiredHeader = "frame,timestamp_ns,device_sequence,left,right";

std::vector<std::string_view> split(std::string_view line) {
  std::vector<std::string_view> fields;
  std::size_t start = 0;
  while (true) {
    const std::size_t comma = line.find(',', start);
    fields.push_back(line.substr(start, comma == std::string_view::npos ? comma : comma - start));
    if (comma == std::string_view::npos) {
      return fields;
    }
    start = comma + 1;
  }
}

template <typename T>
bool parse_integer(std::string_view text, T& value) {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return !text.empty() && error == std::errc() && end == text.data() + text.size();
}

// A path relative to the recording that stays inside it.
bool safe_relative(std::string_view text) {
  if (text.empty()) {
    return false;
  }
  const std::filesystem::path path(text);
  if (path.is_absolute() || path.has_root_name()) {
    return false;
  }
  for (const auto& part : path) {
    if (part == "..") return false;
  }
  return true;
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::invalid_argument(path.string() + ": cannot open");
  }
  std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (file.bad()) {
    throw std::invalid_argument(path.string() + ": cannot read");
  }
  return text;
}

StreamLayout layout_of(const Image& image) {
  return StreamLayout{.width = image.width, .height = image.height, .format = image.format};
}

std::string describe(const StreamLayout& layout) {
  return std::to_string(layout.width) + "x" + std::to_string(layout.height) + " " +
         std::string(pixel_format_name(layout.format));
}

// Reads one stream's image of a frame and checks it against the stream's layout.
void read_stream(const std::filesystem::path& path, const StreamLayout& expected, Image& image) {
  read_netpbm(path, image);
  if (layout_of(image) != expected) {
    throw std::invalid_argument(path.string() + ": is " + describe(layout_of(image)) +
                                ", the recording's first frame " + describe(expected));
  }
}

std::string frame_name(std::uint64_t index, PixelFormat format) {
  std::string name = std::to_string(index);
  if (name.size() < 6) name.insert(0, 6 - name.size(), '0');
  return name + std::string(netpbm_extension(format));
}

bool same_size(const ImageView& view, const CameraModel& camera) {
  return view.width == camera.width && view.height == camera.height;
}

}  // namespace

std::vector<RecordedFrame> parse_index(std::string_view text, std::string_view what) {
  std::vector<RecordedFrame> frames;
  std::vector<std::string_view> columns;
  std::size_t line_number = 0;
  std::size_t start = 0;
  const auto fail = [&](const std::string& problem) {
    throw std::invalid_argument(std::string(what) + ":" + std::to_string(line_number) + ": " +
                                problem);
  };
  int rgb_column = -1;
  int depth_column = -1;
  while (start < text.size()) {
    std::size_t end = text.find('\n', start);
    if (end == std::string_view::npos) end = text.size();
    std::string_view line = text.substr(start, end - start);
    start = end + 1;
    ++line_number;
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    if (line.empty()) continue;
    const std::vector<std::string_view> fields = split(line);
    if (columns.empty()) {
      columns = fields;
      if (!line.starts_with(kRequiredHeader) ||
          (line.size() > kRequiredHeader.size() && line[kRequiredHeader.size()] != ',')) {
        fail("the header must start with " + std::string(kRequiredHeader));
      }
      for (std::size_t i = 5; i < columns.size(); ++i) {
        if (columns[i] == "rgb" && rgb_column < 0 && depth_column < 0) {
          rgb_column = static_cast<int>(i);
        } else if (columns[i] == "depth" && depth_column < 0) {
          depth_column = static_cast<int>(i);
        } else {
          fail("unknown or repeated column \"" + std::string(columns[i]) +
               "\"; after right come rgb and depth, in that order");
        }
      }
      continue;
    }
    if (fields.size() != columns.size()) {
      fail("expected " + std::to_string(columns.size()) + " fields, found " +
           std::to_string(fields.size()));
    }
    RecordedFrame frame;
    if (!parse_integer(fields[0], frame.index)) fail("frame is not a whole number");
    if (!parse_integer(fields[1], frame.timestamp_ns)) fail("timestamp_ns is not a whole number");
    if (!parse_integer(fields[2], frame.device_sequence)) {
      fail("device_sequence is not a whole number");
    }
    if (frame.index != frames.size()) {
      fail("frame " + std::to_string(frame.index) + " out of order; expected " +
           std::to_string(frames.size()));
    }
    if (!frames.empty() && frame.timestamp_ns <= frames.back().timestamp_ns) {
      fail("timestamp_ns does not increase");
    }
    for (std::size_t i = 3; i < fields.size(); ++i) {
      if (!safe_relative(fields[i])) {
        fail("\"" + std::string(fields[i]) + "\" is not a relative path inside the recording");
      }
    }
    frame.left = std::filesystem::path(fields[3]);
    frame.right = std::filesystem::path(fields[4]);
    if (rgb_column >= 0) frame.rgb = std::filesystem::path(fields[rgb_column]);
    if (depth_column >= 0) frame.depth = std::filesystem::path(fields[depth_column]);
    frames.push_back(std::move(frame));
  }
  if (columns.empty()) {
    line_number = 1;
    fail("empty index");
  }
  if (frames.empty()) {
    fail("no frames");
  }
  return frames;
}

RecordingReader::RecordingReader(std::filesystem::path directory)
    : directory_(std::move(directory)) {
  std::error_code error;
  if (!std::filesystem::is_directory(directory_, error)) {
    throw std::invalid_argument(directory_.string() + ": not a recording directory");
  }
  calibration_ = read_calibration(directory_ / kCalibrationFile);
  const std::filesystem::path index = directory_ / kIndexFile;
  frames_ = parse_index(read_text(index), index.string());
  const RecordedFrame& first = frames_.front();
  if (first.rgb && !calibration_.rgb) {
    throw std::invalid_argument(index.string() + ": has rgb images, but " + kCalibrationFile +
                                " has no rgb camera");
  }
  if (first.depth && !calibration_.depth) {
    throw std::invalid_argument(index.string() + ": has depth images, but " + kCalibrationFile +
                                " has no depth camera");
  }
  // The first frame's images set each stream's layout, which must match the
  // calibration; every later frame must match them.
  Image image;
  const auto first_layout = [&](const std::filesystem::path& file, const CameraModel& camera,
                                std::initializer_list<PixelFormat> formats) {
    const std::filesystem::path path = directory_ / file;
    read_netpbm(path, image);
    if (image.width != camera.width || image.height != camera.height) {
      throw std::invalid_argument(path.string() + ": is " + std::to_string(image.width) + "x" +
                                  std::to_string(image.height) + ", the calibration says " +
                                  std::to_string(camera.width) + "x" +
                                  std::to_string(camera.height));
    }
    if (std::find(formats.begin(), formats.end(), image.format) == formats.end()) {
      throw std::invalid_argument(path.string() + ": " +
                                  std::string(pixel_format_name(image.format)) +
                                  " is not a format this stream can have");
    }
    return layout_of(image);
  };
  left_ = first_layout(first.left, calibration_.left, {PixelFormat::kMono8, PixelFormat::kRgb8});
  right_ = first_layout(first.right, calibration_.right, {left_.format});
  if (first.rgb) rgb_ = first_layout(*first.rgb, calibration_.rgb->model, {PixelFormat::kRgb8});
  if (first.depth) {
    depth_ = first_layout(*first.depth, calibration_.depth->model, {PixelFormat::kMono16});
  }
}

std::int64_t RecordingReader::median_period_ns() const noexcept {
  if (frames_.size() < 2) {
    return 0;
  }
  std::vector<std::int64_t> periods;
  periods.reserve(frames_.size() - 1);
  for (std::size_t i = 1; i < frames_.size(); ++i) {
    periods.push_back(frames_[i].timestamp_ns - frames_[i - 1].timestamp_ns);
  }
  const auto middle = periods.begin() + static_cast<std::ptrdiff_t>(periods.size() / 2);
  std::nth_element(periods.begin(), middle, periods.end());
  return *middle;
}

void RecordingReader::read(std::size_t index, Image& left, Image& right, Image* rgb,
                           Image* depth) const {
  if (index >= frames_.size()) {
    throw std::out_of_range("frame " + std::to_string(index) + " is past the recording's " +
                            std::to_string(frames_.size()) + " frames");
  }
  const RecordedFrame& frame = frames_[index];
  read_stream(directory_ / frame.left, left_, left);
  read_stream(directory_ / frame.right, right_, right);
  if (rgb != nullptr && frame.rgb) read_stream(directory_ / *frame.rgb, *rgb_, *rgb);
  if (depth != nullptr && frame.depth) read_stream(directory_ / *frame.depth, *depth_, *depth);
}

RecordingWriter::RecordingWriter(std::filesystem::path directory,
                                 const StereoCalibration& calibration, bool rgb, bool depth)
    : directory_(std::move(directory)), calibration_(calibration), rgb_(rgb), depth_(depth) {
  validate(calibration_);
  if (rgb_ && !calibration_.rgb) {
    throw std::invalid_argument("recording rgb images needs an rgb camera in the calibration");
  }
  if (depth_ && !calibration_.depth) {
    throw std::invalid_argument("recording depth images needs a depth camera in the calibration");
  }
  std::error_code error;
  if (std::filesystem::exists(directory_, error) &&
      !(std::filesystem::is_directory(directory_, error) &&
        std::filesystem::is_empty(directory_, error))) {
    throw std::invalid_argument(directory_.string() + ": exists and is not an empty directory");
  }
  for (const char* sub : {"left", "right", "rgb", "depth"}) {
    if ((std::string_view(sub) == "rgb" && !rgb_) ||
        (std::string_view(sub) == "depth" && !depth_)) {
      continue;
    }
    std::filesystem::create_directories(directory_ / sub, error);
    if (error) {
      throw std::runtime_error((directory_ / sub).string() + ": " + error.message());
    }
  }
  write_calibration(directory_ / kCalibrationFile, calibration_);
}

RecordingWriter::~RecordingWriter() {
  try {
    close();
  } catch (...) {  // NOLINT(bugprone-empty-catch): a destructor cannot report it.
  }
}

void RecordingWriter::write(const StereoFrameView& frame) {
  if (closed_) {
    throw std::invalid_argument("the recording is closed");
  }
  if (!same_size(frame.left, calibration_.left) || !same_size(frame.right, calibration_.right)) {
    throw std::invalid_argument("the eyes must have the calibration's size");
  }
  const auto eye_format = [](PixelFormat format) {
    return format == PixelFormat::kMono8 || format == PixelFormat::kRgb8 ||
           format == PixelFormat::kBgr8;
  };
  if (!eye_format(frame.left.format) || frame.right.format != frame.left.format) {
    throw std::invalid_argument("the eyes must both be mono8, rgb8 or bgr8");
  }
  if (frame.rgb.has_value() != rgb_ || frame.depth.has_value() != depth_) {
    throw std::invalid_argument("the frame must have rgb and depth images as the writer was made");
  }
  if (frame.rgb &&
      (!same_size(*frame.rgb, calibration_.rgb->model) || channels(frame.rgb->format) != 3)) {
    throw std::invalid_argument("the rgb image must be rgb8 or bgr8 of the calibration's size");
  }
  if (frame.depth && (!same_size(*frame.depth, calibration_.depth->model) ||
                      frame.depth->format != PixelFormat::kMono16)) {
    throw std::invalid_argument("the depth image must be mono16 of the calibration's size");
  }
  if (last_timestamp_ && frame.timestamp_ns <= *last_timestamp_) {
    throw std::invalid_argument("timestamps must increase");
  }
  const std::uint64_t index = lines_.size();
  std::string line = std::to_string(index) + "," + std::to_string(frame.timestamp_ns) + "," +
                     std::to_string(frame.device_sequence);
  const auto store = [&](const char* folder, const ImageView& view) {
    const std::string relative = std::string(folder) + "/" + frame_name(index, view.format);
    write_netpbm(directory_ / relative, view);
    line += "," + relative;
  };
  store("left", frame.left);
  store("right", frame.right);
  if (frame.rgb) store("rgb", *frame.rgb);
  if (frame.depth) store("depth", *frame.depth);
  lines_.push_back(std::move(line));
  last_timestamp_ = frame.timestamp_ns;
}

void RecordingWriter::close() {
  if (closed_) {
    return;
  }
  closed_ = true;
  std::string text = "frame,timestamp_ns,device_sequence,left,right";
  if (rgb_) text += ",rgb";
  if (depth_) text += ",depth";
  text += "\n";
  for (const std::string& line : lines_) {
    text += line + "\n";
  }
  const std::filesystem::path path = directory_ / kIndexFile;
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file << text;
  file.close();
  if (!file) {
    throw std::runtime_error(path.string() + ": cannot write");
  }
}

}  // namespace holoscan::stereo_camera
