// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The module's host-only pieces, without the SDK, CUDA or a GPU: the letterbox
// geometry, as the Ultralytics LetterBox computes it, for wide and tall images
// with even and odd padding; the stub model file; the synthetic scenes; and the
// matching of detections to their truth.
//
// Usage: yolo26_host_test DIR (a scratch folder for the files)

#include <algorithm>
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

#include <holoscan/yolo26/letterbox.hpp>
#include <holoscan/yolo26/stub_model.hpp>
#include <holoscan/yolo26/synthetic_scene.hpp>

#include "test_support.hpp"

namespace holoscan::yolo26 {
namespace {

std::filesystem::path g_directory;

BoxDetection box(float x1, float y1, float x2, float y2, int class_id = 0, float score = 0.9F) {
  return BoxDetection{x1, y1, x2, y2, score, class_id};
}

void check_geometry(int width, int height, int size, int resized_width, int resized_height,
                    int pad_left, int pad_top) {
  const Letterbox lb = make_letterbox(width, height, size);
  const std::string what = std::to_string(width) + "x" + std::to_string(height) + " in " +
                           std::to_string(size) + ": " + std::to_string(lb.resized_width) + "x" +
                           std::to_string(lb.resized_height) + " at (" +
                           std::to_string(lb.pad_left) + ", " + std::to_string(lb.pad_top) + ")";
  VERIFY_MSG(lb.resized_width == resized_width && lb.resized_height == resized_height &&
                 lb.pad_left == pad_left && lb.pad_top == pad_top,
             what);
  VERIFY(lb.scale == static_cast<float>(std::min(static_cast<double>(size) / width,
                                                 static_cast<double>(size) / height)));
}

void check_geometries() {
  check_geometry(640, 480, 640, 640, 480, 0, 80);    // Wide, even padding.
  check_geometry(810, 1080, 640, 480, 640, 80, 0);   // Tall, even padding.
  check_geometry(1000, 333, 640, 640, 213, 0, 213);  // Odd padding: 213 above, 214 below.
  check_geometry(333, 1000, 640, 213, 640, 213, 0);  // And 213 left, 214 right.
  // 721 * 0.5 = 360.5 rounds half to even, to 360, as Python's round() does.
  check_geometry(1280, 721, 640, 640, 360, 0, 140);
  check_geometry(1280, 723, 640, 640, 362, 0, 139);  // 361.5 rounds to 362.
  check_geometry(100, 37, 640, 640, 237, 0, 201);    // Upscaled.
  check_geometry(1, 1, 32, 32, 32, 0, 0);
  check_geometry(16384, 1, 4096, 4096, 1, 0, 2047);  // A side never rounds to 0.
  for (const auto& [w, h, s] : {std::array<int, 3>{0, 10, 640}, std::array<int, 3>{10, 0, 640},
                                std::array<int, 3>{16385, 10, 640}, std::array<int, 3>{10, 10, 31},
                                std::array<int, 3>{10, 10, 4097}}) {
    VERIFY_MSG(
        test::throws<std::invalid_argument>([&] { (void)make_letterbox(w, h, s); }),
        "rejects " + std::to_string(w) + "x" + std::to_string(h) + " in " + std::to_string(s));
  }
}

void check_iou() {
  VERIFY(iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == 1.0F);
  VERIFY(iou(box(0, 0, 10, 10), box(10, 0, 20, 10)) == 0.0F);
  VERIFY(iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) == 50.0F / 150.0F);
  VERIFY(iou(box(0, 0, 0, 10), box(0, 0, 0, 10)) == 0.0F);
  VERIFY(edge_error(box(0, 0, 10, 10), box(1, -2, 10.5F, 10)) == 2.0F);
}

void check_match_truth() {
  const std::vector<BoxDetection> truth{box(0, 0, 10, 10, 0), box(20, 20, 40, 40, 1, 0.8F)};
  // Exact within the tolerance.
  TruthMatch m = match_truth(std::vector{box(0.5F, 0, 10, 9.5F, 0), box(20, 21, 40, 40, 1, 0.8F)},
                             truth, 1.0F, 1e-6F);
  VERIFY(m.missed == 0 && m.extra == 0 && m.max_edge_error == 1.0F);
  // An edge too far off: one missed, one extra, and how far.
  m = match_truth(std::vector{box(0, 0, 13, 10, 0), box(20, 20, 40, 40, 1, 0.8F)}, truth, 1.0F,
                  1e-6F);
  VERIFY(m.missed == 1 && m.extra == 1 && m.max_edge_error == 3.0F);
  // The wrong class, the wrong score, a duplicate.
  m = match_truth(std::vector{box(0, 0, 10, 10, 1), box(20, 20, 40, 40, 1, 0.7F)}, truth, 1.0F,
                  1e-6F);
  VERIFY(m.missed == 2 && m.extra == 2);
  m = match_truth(
      std::vector{box(0, 0, 10, 10, 0), box(0, 0, 10, 10, 0), box(20, 20, 40, 40, 1, 0.8F)}, truth,
      1.0F, 1e-6F);
  VERIFY(m.missed == 0 && m.extra == 1);
  m = match_truth(std::vector<BoxDetection>{}, truth, 1.0F, 1e-6F);
  VERIFY(m.missed == 2 && m.extra == 0);
}

void check_stub_model_file() {
  VERIFY(stub_model_onnx(640) == stub_model_onnx(640));
  VERIFY(stub_model_onnx(640) != stub_model_onnx(320));
  VERIFY(test::throws<std::invalid_argument>([] { (void)stub_model_onnx(16); }));
  const auto folder = g_directory / "stub";
  std::filesystem::remove_all(folder);
  std::filesystem::create_directories(folder);
  const auto path = write_stub_model(folder, 320);
  VERIFY(path == folder / "yolo26_stub_320.onnx");
  // An unchanged file is left alone, so an engine cached beside it stays
  // valid; a changed one is replaced.
  const auto written = std::filesystem::last_write_time(path);
  VERIFY(write_stub_model(folder, 320) == path &&
         std::filesystem::last_write_time(path) == written);
  std::ofstream(path, std::ios::binary) << "corrupt";
  (void)write_stub_model(folder, 320);
  std::ifstream in(path, std::ios::binary);
  const std::string content{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  VERIFY(content == stub_model_onnx(320));
  VERIFY(std::distance(std::filesystem::directory_iterator(folder),
                       std::filesystem::directory_iterator{}) == 1);
}

void check_scene() {
  for (const auto& [w, h] : {std::pair{64, 48}, std::pair{1280, 720}, std::pair{333, 1000}}) {
    for (std::uint32_t frame = 0; frame < 60; ++frame) {
      const SyntheticScene scene = make_scene(w, h, frame);
      for (const SceneRect& r : scene.rects) {
        VERIFY(r.x0 >= 0 && r.y0 >= 0 && r.x1 <= w && r.y1 <= h && r.x1 > r.x0 && r.y1 > r.y0);
      }
      // The detected rectangles are apart, so neither covers the other.
      VERIFY(scene.rects[0].x1 <= scene.rects[1].x0);
      const auto truth = scene_truth(scene);
      VERIFY(truth.size() == 2 && truth[0].class_id == 0 && truth[1].class_id == 1);
      VERIFY(truth[0].x1 == static_cast<float>(scene.rects[0].x0) &&
             truth[1].y2 == static_cast<float>(scene.rects[1].y1));
    }
  }
  VERIFY(make_scene(640, 480, 3).rects[0].x0 != make_scene(640, 480, 4).rects[0].x0);
  VERIFY(test::throws<std::invalid_argument>([] { (void)make_scene(63, 48, 0); }));
  VERIFY(test::throws<std::invalid_argument>([] { (void)make_scene(64, 4097, 0); }));
  // The colors land in the right channels and the rectangles in the right
  // places; alpha is opaque.
  const SyntheticScene scene = make_scene(200, 100, 0);
  std::vector<std::uint8_t> bgra(200 * 100 * 4);
  render_scene(scene, PixelFormat::kBgra8, bgra.data(), 200 * 4);
  const SceneRect& red = scene.rects[0];
  const std::uint8_t* inside = bgra.data() + (red.y0 * 200 + red.x0) * 4;
  VERIFY(inside[0] == red.rgb[2] && inside[1] == red.rgb[1] && inside[2] == red.rgb[0] &&
         inside[3] == 255);
  const std::uint8_t* corner = bgra.data();
  VERIFY(corner[0] == 40 && corner[1] == 40 && corner[2] == 40);
}

}  // namespace
}  // namespace holoscan::yolo26

int main(int argc, char** argv) {
  namespace yd = holoscan::yolo26;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " DIR\n";
    return 2;
  }
  yd::g_directory = argv[1];
  std::filesystem::create_directories(yd::g_directory);
  return yd::test::run({
      {"letterbox geometry", yd::check_geometries},
      {"iou and edge error", yd::check_iou},
      {"truth matching", yd::check_match_truth},
      {"stub model file", yd::check_stub_model_file},
      {"synthetic scene", yd::check_scene},
  });
}
