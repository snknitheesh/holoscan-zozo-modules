// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The recording format, without the Holoscan SDK: RecordingWriter writes a
// recording, RecordingReader reads one, or imports a stereo camera recording.
// RecorderOp and ReplayerOp use them; so can any tool. The format is described
// in the module's README.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::recorder {

// A recording that cannot be written, or read: the message names the file, and
// the line of an index, and what is wrong.
class RecordingError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// What a stream holds.
enum class StreamKind : std::uint8_t {
  kImage,   // schema::ImageT: a tensor and its image descriptor.
  kTensor,  // A bare holoscan::Tensor.
};

[[nodiscard]] std::string_view stream_kind_name(StreamKind kind) noexcept;

// An element type, as DLPack has it: `code` is a DLDataTypeCode (0 int, 1 uint,
// 2 float, 4 bfloat, 5 complex, 6 bool), `bits` per lane, `lanes` per element.
struct ElementType {
  std::uint8_t code{1};
  std::uint8_t bits{8};
  std::uint16_t lanes{1};

  // Bytes per element: bits times lanes, rounded up to whole bytes.
  [[nodiscard]] std::uint64_t bytes() const noexcept;
  friend bool operator==(const ElementType&, const ElementType&) = default;
};

// "uint8", "float32", "int16x4" (with lanes), "bool8", "bfloat16", "complex64".
[[nodiscard]] std::string to_string(ElementType type);
// The reverse of to_string(). @throws RecordingError for any other text.
[[nodiscard]] ElementType parse_element_type(std::string_view text);

// One plane of a multi-planar image, as schema::ImagePlaneLayout has it.
struct PlaneLayout {
  std::uint64_t offset_bytes{};
  std::uint32_t row_stride_bytes{};
  std::uint64_t size_bytes{};
  friend bool operator==(const PlaneLayout&, const PlaneLayout&) = default;
};

// What schema::ImageT carries besides its tensor. `encoding` and `color_space`
// are the SDK's names for schema::ImageEncoding and schema::ColorSpace values.
struct ImageDescriptor {
  std::int32_t width{};
  std::int32_t height{};
  std::string encoding{"CUSTOM"};
  std::string color_space{"UNSPECIFIED"};
  std::uint8_t significant_bits{};
  std::int32_t roi_offset_x{};
  std::int32_t roi_offset_y{};
  std::vector<PlaneLayout> plane_layouts;
  std::int64_t exposure_time_ns{};
  float gain{};
  // The image's header (schema::HeaderT), when it had one; its `stamp` is not kept.
  bool has_header{};
  std::string header_frame_id;
  std::int64_t header_capture_timestamp_ns{};
  std::uint64_t header_device_sequence{};
  friend bool operator==(const ImageDescriptor&, const ImageDescriptor&) = default;
};

// One recorded sample: a line of the index.
struct Record {
  // The stream, an index into the recording's streams.
  std::uint32_t stream{};
  // The sample's number on its stream, counting every sample the recorder was
  // given, so a sample it dropped leaves a gap.
  std::uint64_t sequence{};
  // The capture time, in nanoseconds of the clock that stamped it.
  std::int64_t timestamp_ns{};
  // The sample's frame id, or 0 when it had none.
  std::uint64_t frame_id{};
  ElementType dtype;
  // The tensor's shape; its data is stored packed, row-major.
  std::vector<std::int64_t> shape;
  // For an image stream.
  std::optional<ImageDescriptor> image;
  // Where the data is: the file, relative to the recording, and the byte offset
  // in it. Assigned by RecordingWriter::append().
  std::string file;
  std::uint64_t offset{};

  // The size of the packed data: the elements of `shape` times dtype.bytes(), or
  // nothing when that overflows or a dimension is negative.
  [[nodiscard]] std::optional<std::uint64_t> bytes() const noexcept;
};

// The layout of a recording on disk.
enum class RecordingFormat : std::uint8_t {
  kNative,        // recording.json, index.csv and one data file per stream.
  kStereoCamera,  // A stereo camera recording, imported read-only.
};

struct StreamInfo {
  std::string name;
  StreamKind kind{StreamKind::kImage};
  // Its records, and the largest of them.
  std::uint64_t records{};
  std::uint64_t max_bytes{};
  // The element type and rank every record has, when it has any.
  std::optional<ElementType> dtype;
  std::optional<std::uint32_t> rank;
  // Samples the recorder dropped, as recording.json says.
  std::uint64_t dropped{};
};

// A stream to write. Names are 1 to 64 letters, digits, '_', '-' and '.', not
// starting with '.', and unique; a stream's data goes to "<name>.bin".
struct StreamDeclaration {
  std::string name;
  StreamKind kind{StreamKind::kImage};
};

struct WriterOptions {
  // Write into a folder that already holds a recording, replacing it. A folder
  // that holds anything else is never written into.
  bool overwrite{false};
  // Calibration blobs, each a JSON value (usually an object), by name. Stored
  // verbatim in recording.json under "calibration".
  std::map<std::string, std::string> calibration;
  // Free-form text, stored in recording.json under "metadata".
  std::map<std::string, std::string> metadata;
};

// [A-Za-z0-9_.-]{1,64}, not starting with '.'.
[[nodiscard]] bool valid_stream_name(std::string_view name) noexcept;

// Writes a recording: recording.json at once, marked incomplete; each record's
// data to its stream's file and its line to index.csv, flushed, data first, so
// that a recording cut short still reads up to its last full line; and
// recording.json again, complete, at finish(). Not thread-safe: one thread
// writes.
class RecordingWriter {
 public:
  // Creates `directory` (and its parents) if needed. @throws RecordingError if
  // the streams are not valid, a calibration blob is not JSON, or the folder
  // cannot be written (or holds files and `overwrite` does not apply).
  RecordingWriter(std::filesystem::path directory, std::vector<StreamDeclaration> streams,
                  WriterOptions options = {});
  // Closes the files; a recording not finished stays marked incomplete.
  ~RecordingWriter();
  RecordingWriter(const RecordingWriter&) = delete;
  RecordingWriter& operator=(const RecordingWriter&) = delete;

  // Appends `record`, whose data is `data`, packed, exactly record.bytes() long.
  // Its `file` and `offset` are assigned. @throws RecordingError if the record
  // does not fit its stream (image fields on a tensor stream or none on an image
  // stream, a sequence not above the stream's last) or a write fails.
  void append(Record record, std::span<const std::byte> data);

  // Marks the recording complete, with the samples dropped from each stream (in
  // the order of the streams; empty for none). Later appends fail.
  // @throws RecordingError if a write fails.
  void finish(std::span<const std::uint64_t> dropped = {});

  [[nodiscard]] const std::filesystem::path& directory() const noexcept;
  // Records and data bytes written so far.
  [[nodiscard]] std::uint64_t records() const noexcept;
  [[nodiscard]] std::uint64_t bytes() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Reads a recording: either format, told apart by its files. Opening checks the
// whole index and that every record's data is there; read() then only reads.
// read() may be called from any thread.
class RecordingReader {
 public:
  // @throws RecordingError naming the file and what is wrong with it.
  explicit RecordingReader(const std::filesystem::path& directory);
  ~RecordingReader();
  RecordingReader(RecordingReader&&) noexcept;
  RecordingReader& operator=(RecordingReader&&) noexcept;

  [[nodiscard]] RecordingFormat format() const noexcept;
  [[nodiscard]] const std::filesystem::path& directory() const noexcept;
  [[nodiscard]] const std::vector<StreamInfo>& streams() const noexcept;
  // The index of the stream named `name`.
  [[nodiscard]] std::optional<std::uint32_t> find_stream(std::string_view name) const noexcept;
  // Every record, in index order: the order they were written in, or frame by
  // frame for a stereo camera recording.
  [[nodiscard]] const std::vector<Record>& records() const noexcept;
  // The calibration blobs as JSON text, by name. A stereo camera recording has
  // one, "stereo_camera": its calibration.json.
  [[nodiscard]] const std::map<std::string, std::string>& calibration() const noexcept;
  [[nodiscard]] const std::map<std::string, std::string>& metadata() const noexcept;
  // Whether the recorder finished it. An unfinished recording reads up to its
  // last complete record.
  [[nodiscard]] bool complete() const noexcept;

  // Reads `record`'s data, packed, into `out`, which must be exactly
  // record.bytes() long. 16-bit stereo camera images come out in host byte
  // order. @throws RecordingError if the data cannot be read or is not what the
  // index says.
  void read(const Record& record, std::span<std::byte> out) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::recorder
