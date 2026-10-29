// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's recording reader on recordings it writes into the folder given as
// its argument: index.csv with its columns in any order, 8-bit PGM eyes with
// comments, and the files it refuses.

#include "recording.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
int failures = 0;

void check(bool condition, const std::string& what) {
  if (!condition) {
    ++failures;
    std::cerr << "check failed: " << what << '\n';
  }
}

bool refuses(const std::function<void()>& body) {
  try {
    body();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void write(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary) << text;
}

std::string pgm(int width, int height, std::uint8_t first) {
  std::string text =
      "P5\n# a comment\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
  for (int i = 0; i < width * height; ++i) text.push_back(static_cast<char>(first + i));
  return text;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: recording_test <scratch folder>\n";
    return 2;
  }
  const fs::path root = argv[1];
  fs::remove_all(root);
  using rectify_app::read_index;
  using rectify_app::read_pgm;
  try {
    // A recording with rgb and depth columns, which the reader ignores.
    const fs::path clip = root / "clip";
    write(clip / "index.csv",
          "frame,timestamp_ns,device_sequence,left,right,rgb,depth\r\n"
          "0,100,31,left/000000.pgm,right/000000.pgm,rgb/0.ppm,depth/0.pgm\r\n"
          "1,133,32,left/000001.pgm,right/000001.pgm,rgb/1.ppm,depth/1.pgm\r\n");
    write(clip / "left/000000.pgm", pgm(4, 3, 10));
    const auto frames = read_index(clip);
    check(frames.size() == 2, "two frames");
    check(frames[1].timestamp_ns == 133 && frames[1].device_sequence == 32, "frame 1's fields");
    check(frames[0].left == clip / "left/000000.pgm", "left path");
    const auto image = read_pgm(clip / "left/000000.pgm");
    check(image.width == 4 && image.height == 3 && image.pixels.size() == 12, "PGM size");
    check(image.pixels[0] == 10 && image.pixels[11] == 21, "PGM pixels");

    // Columns in another order.
    const fs::path other = root / "other";
    write(other / "index.csv",
          "right,left,frame,device_sequence,timestamp_ns\nr.pgm,l.pgm,0,0,5\n");
    const auto reordered = read_index(other);
    check(reordered.size() == 1 && reordered[0].left == other / "l.pgm" &&
              reordered[0].timestamp_ns == 5,
          "reordered columns");

    // What the reader refuses.
    write(root / "escape/index.csv",
          "frame,timestamp_ns,device_sequence,left,right\n0,1,0,../x.pgm,r.pgm\n");
    check(refuses([&] { (void)read_index(root / "escape"); }), "a path out of the recording");
    write(root / "gap/index.csv",
          "frame,timestamp_ns,device_sequence,left,right\n1,1,0,l.pgm,r.pgm\n");
    check(refuses([&] { (void)read_index(root / "gap"); }), "frames not counting from 0");
    write(root / "nocolumn/index.csv", "frame,timestamp_ns,left,right\n0,1,l.pgm,r.pgm\n");
    check(refuses([&] { (void)read_index(root / "nocolumn"); }), "a missing column");
    check(refuses([&] { (void)read_index(root / "missing"); }), "a missing index");
    write(root / "wide.pgm", "P5\n2 2\n65535\n12345678");
    check(refuses([&] { (void)read_pgm(root / "wide.pgm"); }), "a 16-bit PGM");
    write(root / "short.pgm", "P5\n4 4\n255\nabc");
    check(refuses([&] { (void)read_pgm(root / "short.pgm"); }), "a truncated PGM");
  } catch (const std::exception& error) {
    ++failures;
    std::cerr << "unexpected exception: " << error.what() << '\n';
  }
  std::cout << (failures == 0 ? "all passed" : std::to_string(failures) + " check(s) failed")
            << '\n';
  return failures == 0 ? 0 : 1;
}
