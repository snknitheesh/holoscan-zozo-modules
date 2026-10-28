// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's host-side pieces, without a GPU: matching detections to
// expectations, the expectation and label files, image file headers, PPM
// reading and writing, encodings to RGB and box drawing.
//
// Usage: yolo26_files_test DIR (a scratch folder for the files)

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "expectations.hpp"
#include "image_files.hpp"
#include "test_support.hpp"

namespace yolo26_app {
namespace {

std::filesystem::path g_directory;

std::filesystem::path write(const std::string& name, const std::string& content) {
  const auto path = g_directory / name;
  std::ofstream(path, std::ios::binary) << content;
  return path;
}

BoxDetection box(float x1, float y1, float x2, float y2, int class_id = 0, float score = 0.9F) {
  return BoxDetection{x1, y1, x2, y2, score, class_id};
}

void check_count_matched() {
  const std::vector<Expectation> expected{{"a.jpg", box(0, 0, 100, 100, 15)},
                                          {"a.jpg", box(200, 0, 300, 100, 15)}};
  // One detection cannot match two expectations.
  VERIFY(count_matched(std::vector{box(0, 0, 100, 100, 15)}, expected, 0.8F) == 1);
  VERIFY(count_matched(std::vector{box(5, 5, 100, 100, 15), box(200, 0, 290, 100, 15)}, expected,
                       0.8F) == 2);
  VERIFY(count_matched(std::vector{box(0, 0, 100, 100, 16)}, expected, 0.8F) == 0);
  VERIFY(count_matched(std::vector{box(0, 0, 50, 100, 15)}, expected, 0.8F) == 0);
}

void check_expectation_file() {
  const auto good = write("good.csv",
                          "# image,class_id,x1,y1,x2,y2\n"
                          "\n"
                          "a.jpg,15,344,24.5,640,374.8\r\n"
                          " b.ppm , 0 , 1 , 2 , 3 , 4 \n");
  const auto e = read_expectations(good);
  VERIFY(e.size() == 2);
  if (e.size() == 2) {
    VERIFY(e[0].image == "a.jpg" && e[0].box.class_id == 15 && e[0].box.x1 == 344.0F &&
           e[0].box.y2 == 374.8F);
    VERIFY(e[1].image == "b.ppm" && e[1].box.x2 == 3.0F);
  }
  for (const char* bad : {"a.jpg,15,1,2,3\n", "a.jpg,x,1,2,3,4\n", "a.jpg,1,3,2,1,4\n",
                          ",1,1,2,3,4\n", "a.jpg,-1,1,2,3,4\n", "a.jpg,1,1,2,3,4,5\n"}) {
    const auto path = write("bad.csv", std::string("# header\n") + bad);
    VERIFY_MSG(test::throws<std::runtime_error>([&] { (void)read_expectations(path); }),
               std::string("rejects ") + bad);
  }
  VERIFY(test::throws<std::runtime_error>(
      [] { (void)read_expectations(g_directory / "missing.csv"); }));
}

void check_labels() {
  const auto labels = read_labels(write("labels.txt", "person\nbicycle\r\ntraffic light\n\n"));
  VERIFY(labels == (std::vector<std::string>{"person", "bicycle", "traffic light"}));
  VERIFY(test::throws<std::runtime_error>([] { (void)read_labels(write("empty.txt", "\n")); }));
}

// A minimal JPEG header: SOI, an APP0 segment, then a frame header.
std::string jpeg_header(unsigned char frame_marker, int width, int height, int precision = 8,
                        int components = 3) {
  std::string bytes = "\xFF\xD8";
  bytes += std::string("\xFF\xE0\x00\x04JF", 6);
  bytes += '\xFF';
  bytes += static_cast<char>(frame_marker);
  bytes += std::string("\x00\x11", 2);
  bytes += static_cast<char>(precision);
  bytes += static_cast<char>(height >> 8);
  bytes += static_cast<char>(height & 0xFF);
  bytes += static_cast<char>(width >> 8);
  bytes += static_cast<char>(width & 0xFF);
  bytes += static_cast<char>(components);
  bytes += std::string(9, '\0');
  return bytes;
}

void check_image_headers() {
  ImageFileInfo info = read_image_info(write("baseline.jpg", jpeg_header(0xC0, 810, 1080)));
  VERIFY(info.kind == ImageFileKind::kJpeg && info.width == 810 && info.height == 1080);
  info = read_image_info(write("progressive.JPEG", jpeg_header(0xC2, 640, 480, 8, 1)));
  VERIFY(info.width == 640 && info.height == 480);
  info = read_image_info(write("comment.ppm", "P6\n# a comment\n 33 \t17\n255\n"));
  VERIFY(info.kind == ImageFileKind::kPpm && info.width == 33 && info.height == 17);
  const std::vector<std::pair<std::string, std::string>> bad{
      {"arithmetic.jpg", jpeg_header(0xC9, 10, 10)},
      {"twelve_bit.jpg", jpeg_header(0xC0, 10, 10, 12)},
      {"cmyk.jpg", jpeg_header(0xC0, 10, 10, 8, 4)},
      {"zero.jpg", jpeg_header(0xC0, 0, 10)},
      {"truncated.jpg", jpeg_header(0xC0, 10, 10).substr(0, 9)},
      {"scan_first.jpg", std::string("\xFF\xD8\xFF\xDA\x00\x02", 6)},
      {"png.jpg", "\x89PNG\r\n"},
      {"ascii.ppm", "P3\n2 2\n255\n"},
      {"sixteen_bit.ppm", "P6\n2 2\n65535\n"},
      {"huge.ppm", "P6\n20000 2\n255\n"},
      {"image.png", "P6\n2 2\n255\n"},
  };
  for (const auto& [name, content] : bad) {
    const auto path = write(name, content);
    VERIFY_MSG(test::throws<std::runtime_error>([&] { (void)read_image_info(path); }),
               "rejects " + name);
  }
}

void check_listing() {
  const auto folder = g_directory / "listing";
  std::filesystem::remove_all(folder);
  std::filesystem::create_directories(folder / "sub.jpg");
  for (const char* name : {"b.JPG", "a.ppm", "c.jpeg", "d.png", "e.txt"}) {
    std::ofstream(folder / name) << "x";
  }
  const auto files = list_image_files(folder);
  VERIFY(files == (std::vector<std::filesystem::path>{folder / "a.ppm", folder / "b.JPG",
                                                      folder / "c.jpeg"}));
  VERIFY(list_image_files(folder / "a.ppm") == std::vector{folder / "a.ppm"});
  VERIFY(test::throws<std::runtime_error>([&] { (void)list_image_files(folder / "d.png"); }));
  VERIFY(test::throws<std::runtime_error>([&] { (void)list_image_files(folder / "missing"); }));
  std::filesystem::create_directories(folder / "empty");
  VERIFY(test::throws<std::runtime_error>([&] { (void)list_image_files(folder / "empty"); }));
}

void check_ppm_round_trip() {
  std::vector<std::uint8_t> rgb(5 * 3 * 3);
  for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>(i * 5);
  const auto path = g_directory / "round_trip.ppm";
  write_ppm(path, 5, 3, rgb);
  int width = 0;
  int height = 0;
  VERIFY(read_ppm(path, width, height) == rgb && width == 5 && height == 3);
  const auto truncated = write("truncated.ppm", "P6\n5 3\n255\n" + std::string(10, 'x'));
  VERIFY(test::throws<std::runtime_error>([&] { (void)read_ppm(truncated, width, height); }));
  VERIFY(test::throws<std::runtime_error>([&] { write_ppm(path, 5, 4, rgb); }));
}

void check_to_rgb() {
  // One pixel (10, 20, 30) in each layout, rows padded to 8 bytes.
  const std::array<std::array<std::uint8_t, 8>, 4> pixels{{{10, 20, 30, 0, 0, 0, 0, 0},
                                                           {30, 20, 10, 0, 0, 0, 0, 0},
                                                           {10, 20, 30, 255, 0, 0, 0, 0},
                                                           {30, 20, 10, 255, 0, 0, 0, 0}}};
  const std::array<PixelFormat, 4> formats{PixelFormat::kRgb8, PixelFormat::kBgr8,
                                           PixelFormat::kRgba8, PixelFormat::kBgra8};
  for (std::size_t i = 0; i < formats.size(); ++i) {
    VERIFY(to_rgb(pixels[i].data(), 8, 1, 1, formats[i]) ==
           (std::vector<std::uint8_t>{10, 20, 30}));
  }
}

void check_draw_boxes() {
  constexpr int kW = 20;
  constexpr int kH = 10;
  std::vector<std::uint8_t> rgb(kW * kH * 3, 0);
  const std::vector<BoxDetection> boxes{box(2, 2, 12, 8, 0), box(15, -5, 40, 40, 1)};
  draw_boxes(rgb, kW, kH, boxes, 1);
  const auto lit = [&](int x, int y) { return rgb[(y * kW + x) * 3] | rgb[(y * kW + x) * 3 + 1]; };
  VERIFY(lit(2, 2) && lit(11, 7) && lit(6, 2) && lit(2, 5));  // Edges of the first box.
  VERIFY(!lit(6, 5) && !lit(1, 1) && !lit(12, 8));            // Inside it, and outside.
  VERIFY(lit(15, 0) && lit(19, 9) && !lit(17, 5));            // The second, clipped.
}

}  // namespace
}  // namespace yolo26_app

int main(int argc, char** argv) {
  namespace yd = yolo26_app;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " DIR\n";
    return 2;
  }
  yd::g_directory = argv[1];
  std::filesystem::create_directories(yd::g_directory);
  return yd::test::run({
      {"expectation matching", yd::check_count_matched},
      {"expectation file", yd::check_expectation_file},
      {"labels file", yd::check_labels},
      {"image headers", yd::check_image_headers},
      {"image listing", yd::check_listing},
      {"PPM round trip", yd::check_ppm_round_trip},
      {"encodings to RGB", yd::check_to_rgb},
      {"box drawing", yd::check_draw_boxes},
  });
}
