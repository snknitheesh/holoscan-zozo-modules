// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The stereo camera recordings in $HOLOSCAN_TEST_DATA/stereo-camera/, played by
// ReplayerOp into device memory: every frame of every stream arrives, with the
// recorded capture time, and its pixels are the image file's, read here
// independently of the module. Skipped (exit 77) without HOLOSCAN_TEST_DATA.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/recorder/format.hpp>
#include <holoscan/recorder/replayer_op.hpp>

#include "graph_support.hpp"
#include "test_support.hpp"

namespace holoscan::recorder::test {
namespace {

constexpr int kSkip = 77;
constexpr std::array<const char*, 4> kClips = [] {
  return std::array<const char*, 4>{"d585_720p_emitter_on", "d585_720p_emitter_off",
                                    "d585_480p_emitter_on", "d585_480p_emitter_off"};
}();

// A Netpbm image as its file has it, 16-bit samples turned to host order.
struct Image {
  int width{};
  int height{};
  int channels{};
  std::vector<std::uint8_t> bytes;
};

Image read_netpbm(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  int maxval = 0;
  Image image;
  file >> magic >> image.width >> image.height >> maxval;
  file.get();
  image.channels = magic == "P6" ? 3 : 1;
  const int sample = maxval > 255 ? 2 : 1;
  image.bytes.resize(static_cast<std::size_t>(image.width) * image.height * image.channels *
                     sample);
  file.read(reinterpret_cast<char*>(image.bytes.data()),
            static_cast<std::streamsize>(image.bytes.size()));
  if (!file) throw std::runtime_error(path.string() + ": short read");
  if (sample == 2) {
    for (std::size_t i = 0; i + 1 < image.bytes.size(); i += 2) {
      std::swap(image.bytes[i], image.bytes[i + 1]);
    }
  }
  return image;
}

struct Row {
  std::int64_t timestamp_ns{};
  std::vector<std::string> files;
};

std::vector<Row> read_index(const std::filesystem::path& clip, std::vector<std::string>& columns) {
  std::ifstream file(clip / "index.csv");
  std::string line;
  std::getline(file, line);
  std::stringstream header(line);
  std::string field;
  for (int i = 0; std::getline(header, field, ','); ++i) {
    if (i >= 3) columns.push_back(field);
  }
  std::vector<Row> rows;
  while (std::getline(file, line)) {
    std::stringstream parts(line);
    Row row;
    for (int i = 0; std::getline(parts, field, ','); ++i) {
      if (i == 1) row.timestamp_ns = std::stoll(field);
      if (i >= 3) row.files.push_back(field);
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

void play_clip(const std::filesystem::path& clip) {
  std::vector<std::string> columns;
  const std::vector<Row> rows = read_index(clip, columns);
  VERIFY_MSG(rows.size() == 60U, clip.string() + ": 60 frames");
  VERIFY(columns == (std::vector<std::string>{"left", "right", "rgb", "depth"}));
  std::vector<Stream> streams;
  for (const std::string& name : columns) streams.push_back(Stream{.name = name});

  holoscan::Graph graph{"real_data"};
  auto stats = std::make_shared<ReplayerStats>();
  auto replayer = graph.op<ReplayerOp>(
      "replayer",
      ReplayerConfig{.directory = clip.string(), .memory = ReplayMemory::kDevice, .stats = stats});
  auto collected = std::make_shared<Collected>();
  auto sink = graph.op<CollectSinkOp>("sink", SinkConfig{.streams = streams,
                                                         .expected = rows.size() * streams.size(),
                                                         .collected = collected});
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  for (const std::string& name : columns) {
    graph.add_flow(replayer->image(name), sink->image(name), lossless);
  }
  VERIFY(replayer->recording().calibration().count("stereo_camera") == 1U);
  holoscan::CompileOptions options;
  replayer->bind_device(options, "replayer");
  const RunOutcome outcome = run_graph(graph, std::move(options), std::chrono::seconds{300});
  VERIFY_MSG(outcome.compiled && outcome.completed, outcome.diagnostics);
  VERIFY(stats->emitted == rows.size() * streams.size());

  for (std::size_t s = 0; s < columns.size(); ++s) {
    const std::vector<Received>& samples = collected->streams[columns[s]];
    VERIFY_MSG(samples.size() == rows.size(), columns[s] + ": every frame");
    for (std::size_t f = 0; f < samples.size() && f < rows.size(); ++f) {
      const Received& got = samples[f];
      const std::string what =
          clip.filename().string() + " " + columns[s] + " frame " + std::to_string(f);
      VERIFY_MSG(got.capture_ns == rows[f].timestamp_ns, what + ": capture time");
      VERIFY_MSG(got.frame_id == f + 1, what + ": frame id");
      VERIFY_MSG(got.device == kDLCUDA, what + ": device memory");
      // Every frame's pixels would read 1.1 GB; the first, a middle and the last do.
      if (f == 0 || f == 30 || f + 1 == rows.size()) {
        const Image image = read_netpbm(clip / rows[f].files[s]);
        VERIFY_MSG(got.bytes == image.bytes, what + ": pixels");
        VERIFY_MSG(
            got.image && got.image->width == image.width && got.image->height == image.height,
            what + ": size");
        const schema::ImageEncoding expected = columns[s] == "depth" ? schema::ImageEncoding_MONO16
                                               : columns[s] == "rgb" ? schema::ImageEncoding_RGB8
                                                                     : schema::ImageEncoding_MONO8;
        VERIFY_MSG(got.image && got.image->encoding == expected, what + ": encoding");
      }
    }
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
  using holoscan::recorder::test::TestCase;
  namespace t = holoscan::recorder::test;
  std::vector<TestCase> cases;
  for (const char* clip : t::kClips) {
    cases.push_back(
        TestCase{std::string("replays ") + clip, [root, clip] { t::play_clip(root / clip); }});
  }
  return t::run(cases);
}
