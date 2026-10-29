// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Recordings: Netpbm images in every format, the index, and a synthetic
// recording written and read back pixel for pixel; the synthetic camera's known
// disparity; and malformed recordings, each rejected with a message that names
// the file and the problem.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/stereo_camera/netpbm.hpp>
#include <holoscan/stereo_camera/recording.hpp>
#include <holoscan/stereo_camera/synthetic.hpp>

#include "host_support.hpp"
#include "test_support.hpp"

namespace holoscan::stereo_camera {
namespace {

using test::contains;
using test::invalid_message;

constexpr int kWidth = 96;
constexpr int kHeight = 72;

void check_netpbm(const std::filesystem::path& work) {
  const auto folder = test::fresh_folder(work, "netpbm");
  // Every format, with padded rows in the source view.
  for (const PixelFormat format :
       {PixelFormat::kMono8, PixelFormat::kMono16, PixelFormat::kRgb8, PixelFormat::kBgr8}) {
    const int width = 7;
    const int height = 5;
    const std::size_t row = static_cast<std::size_t>(width) * bytes_per_pixel(format);
    const std::size_t pitch = row + 5;
    std::vector<std::uint8_t> padded(pitch * height, 0xEE);
    for (int y = 0; y < height; ++y) {
      for (std::size_t x = 0; x < row; ++x) {
        padded[y * pitch + x] = static_cast<std::uint8_t>(31 * y + 7 * x + 1);
      }
    }
    const ImageView view{
        .data = padded.data(), .width = width, .height = height, .pitch = pitch, .format = format};
    const auto path = folder / ("image" + std::string(netpbm_extension(format)));
    write_netpbm(path, view);
    Image back;
    read_netpbm(path, back);
    if (format == PixelFormat::kBgr8) {
      // Written as RGB: read back with the channels swapped.
      VERIFY(back.format == PixelFormat::kRgb8);
      bool swapped = true;
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const std::uint8_t* in = padded.data() + y * pitch + 3 * x;
          const std::uint8_t* out = back.pixels.data() + (y * width + x) * 3;
          swapped &= out[0] == in[2] && out[1] == in[1] && out[2] == in[0];
        }
      }
      VERIFY(swapped);
    } else {
      VERIFY_MSG(same_pixels(view, back.view()),
                 std::string(pixel_format_name(format)) + " did not round-trip");
    }
  }
  // 16-bit samples are big-endian in the file.
  const std::uint16_t value = 0x1234;
  write_netpbm(folder / "one.pgm", ImageView{.data = reinterpret_cast<const std::uint8_t*>(&value),
                                             .width = 1,
                                             .height = 1,
                                             .pitch = 2,
                                             .format = PixelFormat::kMono16});
  const std::string bytes = test::read_text(folder / "one.pgm");
  VERIFY(bytes == std::string("P5\n1 1\n65535\n") + "\x12\x34");
  // Headers with comments decode.
  Image image;
  decode_netpbm(std::string("P5 # gray\n2 # wide\n1\n255\n") + "\x01\x02", image);
  VERIFY(image.width == 2 && image.height == 1 && image.pixels[1] == 2);
  const std::vector<std::pair<std::string, std::string>> bad{
      {"P2\n1 1\n255\n1", "not a binary PGM (P5) or PPM (P6)"},
      {"P5\n1 1\n1023\n\x01\x02", "only 8-bit or 16-bit PGM"},
      {"P6\n1 1\n65535\n", "only 8-bit PPM"},
      {"P5\n2 2\n255\n\x01", "pixel data is truncated: 1 of 4 bytes"},
      {"P5\n0 2\n255\n", "width and height must be 1..16384"},
      {"P5\n2\n", "header has no height"},
      {"P5\n2 2\n", "header has no maxval"},
      {"P5\n2 2\n255", "header does not end with white space"},
  };
  for (const auto& [text, expected] : bad) {
    const std::string message = invalid_message([&] { decode_netpbm(text, image, "x.pgm"); });
    VERIFY_MSG(contains(message, "x.pgm: " + expected), "\"" + message + "\" for " + text);
  }
}

void check_index() {
  const std::vector<RecordedFrame> frames = parse_index(
      "frame,timestamp_ns,device_sequence,left,right,depth\r\n"
      "0,100,7,left/0.pgm,right/0.pgm,depth/0.pgm\r\n"
      "\n"
      "1,200,8,left/1.pgm,right/1.pgm,depth/1.pgm");
  VERIFY(frames.size() == 2);
  VERIFY(frames[1].index == 1 && frames[1].timestamp_ns == 200 && frames[1].device_sequence == 8);
  VERIFY(frames[1].left == "left/1.pgm" && !frames[1].rgb && frames[1].depth == "depth/1.pgm");
  const std::string header = "frame,timestamp_ns,device_sequence,left,right\n";
  const std::vector<std::pair<std::string, std::string>> bad{
      {"", "index.csv:1: empty index"},
      {header, "index.csv:1: no frames"},
      {"frame,timestamp,device_sequence,left,right\n0,1,0,a,b\n",
       "index.csv:1: the header must start with"},
      {"frame,timestamp_ns,device_sequence,left,right,depth,rgb\n",
       "index.csv:1: unknown or repeated column \"rgb\""},
      {"frame,timestamp_ns,device_sequence,left,right,colour\n",
       "unknown or repeated column \"colour\""},
      {header + "0,1,0,a\n", "index.csv:2: expected 5 fields, found 4"},
      {header + "1,1,0,a,b\n", "index.csv:2: frame 1 out of order; expected 0"},
      {header + "0,1,0,a,b\n1,1,0,a,b\n", "index.csv:3: timestamp_ns does not increase"},
      {header + "0,x,0,a,b\n", "timestamp_ns is not a whole number"},
      {header + "0,1,-1,a,b\n", "device_sequence is not a whole number"},
      {header + "0,1,0,/etc/passwd,b\n", "\"/etc/passwd\" is not a relative path"},
      {header + "0,1,0,a,../b\n", "\"../b\" is not a relative path"},
      {header + "0,1,0,a,\n", "\"\" is not a relative path"},
  };
  for (const auto& [text, expected] : bad) {
    const std::string message = invalid_message([&] { (void)parse_index(text); });
    VERIFY_MSG(contains(message, expected), "\"" + message + "\" for:\n" + text);
  }
}

void check_synthetic_scene() {
  for (std::uint64_t index : {0U, 5U, 40U}) {
    const SyntheticScene scene(kWidth, kHeight, index);
    int visible = 0;
    int mismatched = 0;
    int occluded = 0;
    for (int v = 0; v < kHeight; ++v) {
      for (int u = 0; u < kWidth; ++u) {
        const int d = scene.disparity(u, v);
        VERIFY(d == scene.background_disparity() || d == scene.foreground_disparity());
        if (scene.visible_in_right(u, v)) {
          ++visible;
          mismatched += scene.left(u, v) != scene.right(u - d, v);
        } else {
          occluded += u - d >= 0;
        }
      }
    }
    VERIFY_MSG(mismatched == 0, std::to_string(mismatched) + " visible points differ");
    VERIFY(visible > kWidth * kHeight / 2);
    VERIFY_MSG(occluded > 0, "the rectangle hides some of the background from the right eye");
  }
  VERIFY(test::throws<std::invalid_argument>([] { SyntheticScene(32, 64, 0); }));
}

void check_synthetic_round_trip(const std::filesystem::path& work) {
  const auto folder = work / "synthetic_recording";
  std::filesystem::remove_all(folder);
  constexpr std::uint32_t kFrames = 4;
  write_synthetic_recording(folder, kWidth, kHeight, kFrames, true, true);
  const RecordingReader reader(folder);
  VERIFY(reader.size() == kFrames && reader.has_rgb() && reader.has_depth());
  VERIFY((reader.left_layout() == StreamLayout{kWidth, kHeight, PixelFormat::kMono8}));
  VERIFY((reader.depth_layout() == StreamLayout{kWidth, kHeight, PixelFormat::kMono16}));
  VERIFY(reader.median_period_ns() == 33'333'333);
  VERIFY(reader.calibration().rectified && reader.calibration().camera.name == "synthetic");
  Image left;
  Image right;
  Image rgb;
  Image depth;
  Image truth_left;
  Image truth_right;
  Image truth_rgb;
  Image truth_depth;
  for (std::uint32_t i = 0; i < kFrames; ++i) {
    reader.read(i, left, right, &rgb, &depth);
    render_synthetic_frame(kWidth, kHeight, i, truth_left, truth_right, &truth_rgb, &truth_depth);
    VERIFY(same_pixels(left.view(), truth_left.view()));
    VERIFY(same_pixels(right.view(), truth_right.view()));
    VERIFY(same_pixels(rgb.view(), truth_rgb.view()));
    VERIFY(same_pixels(depth.view(), truth_depth.view()));
    VERIFY(reader.frames()[i].timestamp_ns == synthetic_timestamp_ns(i));
    VERIFY(reader.frames()[i].device_sequence == i + 1U);
  }
  // Depth is fx * baseline / disparity, in millimetres.
  const SyntheticScene scene(kWidth, kHeight, kFrames - 1);
  const double factor = reader.calibration().depth_from_disparity();
  int wrong = 0;
  for (int v = 0; v < kHeight; ++v) {
    for (int u = 0; u < kWidth; ++u) {
      std::uint16_t mm = 0;
      std::memcpy(&mm, depth.pixels.data() + 2 * (v * kWidth + u), 2);
      wrong += mm != std::lround(1000.0 * factor / scene.disparity(u, v));
    }
  }
  VERIFY(wrong == 0);
  VERIFY(test::throws<std::out_of_range>(
      [&] { reader.read(kFrames, left, right, nullptr, nullptr); }));
  // Without rgb and depth.
  const auto plain = work / "synthetic_plain";
  std::filesystem::remove_all(plain);
  write_synthetic_recording(plain, kWidth, kHeight, 2, false, false);
  const RecordingReader plain_reader(plain);
  VERIFY(!plain_reader.has_rgb() && !plain_reader.has_depth());
  VERIFY(test::read_text(plain / kIndexFile)
             .starts_with(
                 "frame,timestamp_ns,device_sequence,left,right\n0,1000000000,1,left/000000.pgm,"));
}

void check_writer_rules(const std::filesystem::path& work) {
  const auto folder = test::fresh_folder(work, "writer");
  test::write_text(folder / "occupied", "");
  const StereoCalibration calibration = synthetic_calibration(kWidth, kHeight, false, false);
  VERIFY(contains(invalid_message([&] { RecordingWriter(folder, calibration, false, false); }),
                  "exists and is not an empty directory"));
  VERIFY(contains(invalid_message([&] { RecordingWriter(folder / "a", calibration, true, false); }),
                  "needs an rgb camera in the calibration"));
  RecordingWriter writer(folder / "b", calibration, false, false);
  Image left;
  Image right;
  render_synthetic_frame(kWidth, kHeight, 0, left, right, nullptr, nullptr);
  StereoFrameView frame{.left = left.view(), .right = right.view(), .timestamp_ns = 10};
  writer.write(frame);
  VERIFY(contains(invalid_message([&] { writer.write(frame); }), "timestamps must increase"));
  frame.timestamp_ns = 20;
  frame.depth = left.view();
  VERIFY(contains(invalid_message([&] { writer.write(frame); }), "rgb and depth images"));
  frame.depth.reset();
  Image small;
  small.resize(8, 8, PixelFormat::kMono8);
  frame.left = small.view();
  VERIFY(contains(invalid_message([&] { writer.write(frame); }), "the calibration's size"));
  frame.left = left.view();
  writer.write(frame);
  writer.close();
  VERIFY(contains(invalid_message([&] { writer.write(frame); }), "the recording is closed"));
  VERIFY(RecordingReader(folder / "b").size() == 2);
}

// A copy of a fresh synthetic recording, broken by `damage`.
template <typename Damage>
std::string broken(const std::filesystem::path& work, Damage&& damage) {
  const auto folder = work / "broken";
  std::filesystem::remove_all(folder);
  write_synthetic_recording(folder, kWidth, kHeight, 3, false, true);
  damage(folder);
  return invalid_message([&] {
    const RecordingReader reader(folder);
    Image left;
    Image right;
    Image depth;
    for (std::size_t i = 0; i < reader.size(); ++i) reader.read(i, left, right, nullptr, &depth);
  });
}

void check_malformed_recordings(const std::filesystem::path& work) {
  using std::filesystem::path;
  const auto expect = [](const std::string& message, const std::string& part) {
    VERIFY_MSG(contains(message, part), "expected \"" + part + "\", got \"" + message + "\"");
  };
  expect(invalid_message([&] { RecordingReader(work / "nowhere"); }),
         "nowhere: not a recording directory");
  expect(broken(work, [](const path& f) { std::filesystem::remove(f / kCalibrationFile); }),
         "calibration.json: cannot open");
  expect(broken(work, [](const path& f) { std::filesystem::remove(f / kIndexFile); }),
         "index.csv: cannot open");
  expect(broken(work, [](const path& f) { test::write_text(f / kCalibrationFile, "{"); }),
         "calibration.json: line 1, column 2");
  expect(broken(work,
                [](const path& f) {
                  // The depth camera removed from the calibration.
                  test::write_text(f / kCalibrationFile, format_calibration(synthetic_calibration(
                                                             kWidth, kHeight, false, false)));
                }),
         "has depth images, but calibration.json has no depth camera");
  expect(broken(work,
                [](const path& f) {
                  test::write_text(f / kCalibrationFile, format_calibration(synthetic_calibration(
                                                             kWidth + 2, kHeight, false, true)));
                }),
         "000000.pgm: is 96x72, the calibration says 98x72");
  expect(broken(work, [](const path& f) { std::filesystem::remove(f / "right/000001.pgm"); }),
         "right/000001.pgm: cannot open");
  expect(broken(work,
                [](const path& f) {
                  test::write_text(f / "left/000002.pgm", "P5\n96 72\n255\n\x01\x02");
                }),
         "left/000002.pgm: pixel data is truncated");
  expect(broken(work,
                [](const path& f) {
                  // A later frame of another size.
                  Image small;
                  small.resize(8, 8, PixelFormat::kMono8);
                  write_netpbm(f / "left/000001.pgm", small.view());
                }),
         "left/000001.pgm: is 8x8 mono8, the recording's first frame 96x72 mono8");
  expect(broken(work,
                [](const path& f) {
                  // A depth image that is 8-bit.
                  Image gray;
                  gray.resize(kWidth, kHeight, PixelFormat::kMono8);
                  write_netpbm(f / "depth/000000.pgm", gray.view());
                }),
         "depth/000000.pgm: mono8 is not a format this stream can have");
  expect(broken(work,
                [](const path& f) {
                  // Eyes of different formats.
                  Image color;
                  color.resize(kWidth, kHeight, PixelFormat::kRgb8);
                  write_netpbm(f / "right/000000.ppm", color.view());
                  std::string index = test::read_text(f / kIndexFile);
                  index.replace(index.find("right/000000.pgm"), 16, "right/000000.ppm");
                  test::write_text(f / kIndexFile, index);
                }),
         "right/000000.ppm: rgb8 is not a format this stream can have");
  expect(broken(work,
                [](const path& f) {
                  std::string index = test::read_text(f / kIndexFile);
                  index.replace(index.find("1,1033333333"), 12, "1,999");
                  test::write_text(f / kIndexFile, index);
                }),
         "index.csv:3: timestamp_ns does not increase");
}

}  // namespace
}  // namespace holoscan::stereo_camera

int main(int argc, char** argv) {
  namespace sc = holoscan::stereo_camera;
  const std::filesystem::path work = argc > 1 ? argv[1] : std::filesystem::current_path();
  return sc::test::run({
      {"Netpbm images", [&] { sc::check_netpbm(work); }},
      {"index.csv", sc::check_index},
      {"synthetic scene disparity", sc::check_synthetic_scene},
      {"synthetic recording round trip", [&] { sc::check_synthetic_round_trip(work); }},
      {"writer rules", [&] { sc::check_writer_rules(work); }},
      {"malformed recordings", [&] { sc::check_malformed_recordings(work); }},
  });
}
