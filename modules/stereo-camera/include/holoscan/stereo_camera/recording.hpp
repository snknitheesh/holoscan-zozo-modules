// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A recording of a stereo camera: a directory with the camera's calibration,
// an index of its frames, and each frame's images as Netpbm files.
//
//   calibration.json       The calibration (calibration.hpp)
//   index.csv              One line per frame, after a header line:
//                            frame,timestamp_ns,device_sequence,left,right[,rgb][,depth]
//   left/000000.pgm ...    8-bit PGM, or 8-bit PPM for a color pair
//   right/000000.pgm ...   Like left
//   rgb/000000.ppm ...     8-bit PPM, when recorded
//   depth/000000.pgm ...   16-bit PGM in the calibration's depth units, when recorded
//
// The file columns hold paths relative to the directory. Frames are numbered
// 0, 1, 2, ... in order, and their timestamps increase. The reader checks all
// of it, and needs nothing but this library: no camera driver. No SDK.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/stereo_camera/calibration.hpp>
#include <holoscan/stereo_camera/image.hpp>

namespace holoscan::stereo_camera {

inline constexpr const char* kCalibrationFile = "calibration.json";
inline constexpr const char* kIndexFile = "index.csv";

// One line of index.csv.
struct RecordedFrame {
  std::uint64_t index{};
  std::int64_t timestamp_ns{};
  std::uint64_t device_sequence{};
  // Relative to the recording's directory.
  std::filesystem::path left;
  std::filesystem::path right;
  std::optional<std::filesystem::path> rgb;
  std::optional<std::filesystem::path> depth;
};

// The size and pixel format of one stream.
struct StreamLayout {
  int width{};
  int height{};
  PixelFormat format{PixelFormat::kMono8};

  friend bool operator==(const StreamLayout&, const StreamLayout&) = default;
};

// Parses the text of index.csv. Blank lines and a final newline are allowed.
// @throws std::invalid_argument naming `what` and the line of the first problem:
//         an unknown or missing column, a number that does not parse, a frame out
//         of order, a timestamp that does not increase, an empty or absolute path
//         or one with "..", or no frame at all.
[[nodiscard]] std::vector<RecordedFrame> parse_index(std::string_view text,
                                                     std::string_view what = kIndexFile);

// Reads a recording. The constructor reads the calibration and the index, and
// the headers of the first frame's files, to learn each stream's layout; it
// reads no pixels.
class RecordingReader {
 public:
  // @throws std::invalid_argument naming the file and the problem: a missing or
  //         malformed calibration or index, an rgb or depth column without that
  //         camera in the calibration (or the reverse), or first images whose
  //         sizes differ from the calibration's.
  explicit RecordingReader(std::filesystem::path directory);

  [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }
  [[nodiscard]] const StereoCalibration& calibration() const noexcept { return calibration_; }
  [[nodiscard]] const std::vector<RecordedFrame>& frames() const noexcept { return frames_; }
  [[nodiscard]] std::size_t size() const noexcept { return frames_.size(); }
  [[nodiscard]] bool has_rgb() const noexcept { return rgb_.has_value(); }
  [[nodiscard]] bool has_depth() const noexcept { return depth_.has_value(); }
  [[nodiscard]] const StreamLayout& left_layout() const noexcept { return left_; }
  [[nodiscard]] const StreamLayout& right_layout() const noexcept { return right_; }
  [[nodiscard]] const std::optional<StreamLayout>& rgb_layout() const noexcept { return rgb_; }
  [[nodiscard]] const std::optional<StreamLayout>& depth_layout() const noexcept { return depth_; }
  // The median time between frames, from the timestamps; 0 for one frame.
  [[nodiscard]] std::int64_t median_period_ns() const noexcept;

  // Decodes frame `index`'s images into the given ones, reusing their memory;
  // `rgb` and `depth` may be null to skip those streams.
  // @throws std::out_of_range for an index past the end.
  // @throws std::invalid_argument naming the file, for one that is missing, does
  //         not decode, or differs in size or format from the first frame's.
  void read(std::size_t index, Image& left, Image& right, Image* rgb, Image* depth) const;

 private:
  std::filesystem::path directory_;
  StereoCalibration calibration_;
  std::vector<RecordedFrame> frames_;
  StreamLayout left_;
  StreamLayout right_;
  std::optional<StreamLayout> rgb_;
  std::optional<StreamLayout> depth_;
};

// Writes a recording, frame by frame. The constructor creates the directory,
// which must not exist or be empty, and writes the calibration; each write()
// adds a frame's files and its index line. The index is written in full by
// close() (or the destructor), so an interrupted recording has none and does
// not load.
class RecordingWriter {
 public:
  // @throws std::invalid_argument if the calibration is not valid or the
  //         directory is not empty, std::runtime_error if it cannot be written.
  RecordingWriter(std::filesystem::path directory, const StereoCalibration& calibration, bool rgb,
                  bool depth);
  ~RecordingWriter();
  RecordingWriter(const RecordingWriter&) = delete;
  RecordingWriter& operator=(const RecordingWriter&) = delete;

  // Writes `frame` as the next one: its images, numbered from 0, and the source's
  // timestamp and device sequence. The frame must have rgb and depth exactly when
  // the writer was made for them, the eyes the calibration's size, and a timestamp
  // after the previous frame's.
  // @throws std::invalid_argument for such a frame, std::runtime_error if a file
  //         cannot be written.
  void write(const StereoFrameView& frame);
  // Writes index.csv. Further writes are errors.
  // @throws std::runtime_error if it cannot be written.
  void close();

  [[nodiscard]] std::size_t frames() const noexcept { return lines_.size(); }

 private:
  std::filesystem::path directory_;
  StereoCalibration calibration_;
  bool rgb_{};
  bool depth_{};
  bool closed_{};
  std::optional<std::int64_t> last_timestamp_;
  std::vector<std::string> lines_;
};

}  // namespace holoscan::stereo_camera
