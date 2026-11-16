// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The four stereo camera recordings in $HOLOSCAN_TEST_DATA/stereo-camera/, read
// with RecordingReader: their streams, sizes, timestamps and calibration, and
// every pixel of every frame against the image files read here independently.
// Skipped (exit 77) without HOLOSCAN_TEST_DATA.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/recorder/format.hpp>

#include "host_support.hpp"
#include "test_support.hpp"

namespace holoscan::recorder::test {
namespace {

constexpr int kSkip = 77;
std::uint64_t g_bytes = 0;

struct Clip {
  const char* name;
  int width;
  int height;
};
constexpr std::array<Clip, 4> kClips = [] {
  return std::array<Clip, 4>{
      Clip{"d585_720p_emitter_on", 1280, 720}, Clip{"d585_720p_emitter_off", 1280, 720},
      Clip{"d585_480p_emitter_on", 848, 480}, Clip{"d585_480p_emitter_off", 848, 480}};
}();

// The pixels of a Netpbm file, 16-bit samples turned to host order.
std::vector<std::byte> pixels(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  int width = 0;
  int height = 0;
  int maxval = 0;
  file >> magic >> width >> height >> maxval;
  file.get();
  const std::size_t sample = maxval > 255 ? 2U : 1U;
  std::vector<std::byte> out(static_cast<std::size_t>(width) * height * (magic == "P6" ? 3U : 1U) *
                             sample);
  file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  if (sample == 2U) {
    for (std::size_t i = 0; i + 1 < out.size(); i += 2) std::swap(out[i], out[i + 1]);
  }
  return out;
}

void read_clip(const std::filesystem::path& folder, const Clip& clip) {
  const RecordingReader reader(folder);
  VERIFY(reader.format() == RecordingFormat::kStereoCamera);
  VERIFY(reader.complete());
  VERIFY(reader.streams().size() == 4U);
  const std::vector<std::string> names{"left", "right", "rgb", "depth"};
  const std::vector<std::string> encodings{"MONO8", "MONO8", "RGB8", "MONO16"};
  for (std::size_t s = 0; s < reader.streams().size() && s < names.size(); ++s) {
    const StreamInfo& info = reader.streams()[s];
    VERIFY(info.name == names[s]);
    VERIFY(info.kind == StreamKind::kImage);
    VERIFY(info.records == 60U);
  }
  const std::string calibration = reader.calibration().at("stereo_camera");
  VERIFY(contains(calibration, "\"rectified\":true"));
  VERIFY(contains(calibration, "\"serial\":\"541522320091\""));

  // The index, as the file has it.
  std::ifstream index(folder / "index.csv");
  std::string line;
  std::getline(index, line);
  std::vector<std::vector<std::string>> rows;
  while (std::getline(index, line)) {
    std::stringstream parts(line);
    std::vector<std::string> row;
    std::string field;
    while (std::getline(parts, field, ',')) row.push_back(field);
    rows.push_back(std::move(row));
  }
  VERIFY(reader.records().size() == rows.size() * 4U);
  std::vector<std::byte> data;
  for (const Record& record : reader.records()) {
    const std::size_t frame = record.sequence;
    const std::vector<std::string>& row = rows.at(frame);
    const std::string what =
        std::string(clip.name) + " " + names[record.stream] + " frame " + std::to_string(frame);
    VERIFY_MSG(record.timestamp_ns == std::stoll(row[1]), what + ": timestamp");
    VERIFY_MSG(record.frame_id == frame + 1, what + ": frame id");
    VERIFY_MSG(record.image && record.image->encoding == encodings[record.stream],
               what + ": encoding");
    VERIFY_MSG(
        record.image && record.image->width == clip.width && record.image->height == clip.height,
        what + ": size");
    VERIFY_MSG(record.image && record.image->header_device_sequence == std::stoull(row[2]),
               what + ": device sequence");
    data.resize(*record.bytes());
    reader.read(record, data);
    VERIFY_MSG(data == pixels(folder / row[3 + record.stream]), what + ": pixels");
    g_bytes += data.size();
  }
}

}  // namespace
}  // namespace holoscan::recorder::test

int main() {
  const char* data = std::getenv("HOLOSCAN_TEST_DATA");
  const std::filesystem::path root =
      data == nullptr ? std::filesystem::path() : std::filesystem::path(data) / "stereo-camera";
  if (data == nullptr || !std::filesystem::is_directory(root)) {
    std::cout << "SKIP: set HOLOSCAN_TEST_DATA to a folder with stereo-camera/ recordings\n";
    return holoscan::recorder::test::kSkip;
  }
  namespace t = holoscan::recorder::test;
  std::vector<t::TestCase> cases;
  for (const t::Clip& clip : t::kClips) {
    cases.push_back(t::TestCase{std::string("reads ") + clip.name,
                                [root, clip] { t::read_clip(root / clip.name, clip); }});
  }
  const int status = t::run(cases);
  std::cout << "compared " << (t::g_bytes >> 20U) << " MiB\n";
  return status;
}
