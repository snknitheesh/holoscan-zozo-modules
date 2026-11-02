// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Both backends through the public factory: the cuda backend finds the same markers as the cpu
// backend, with the same corners, and refines the same chessboard corners, on clean, blurred,
// noisy, occluded, mirrored and cluttered images; both against ground truth; a frame over the
// candidate capacity reports it; what auto resolves to; and each backend's time at 1280x720.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/fiducials/charuco.hpp>
#include <holoscan/fiducials/detector_backend.hpp>
#include <holoscan/fiducials/pose.hpp>
#include <holoscan/fiducials/synthetic_scene.hpp>

#include "test_support.hpp"

namespace holoscan::fiducials {
namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;

// A gray image on the host and its copy on the device.
struct TestImage {
  std::vector<std::uint8_t> host;
  std::uint8_t* device{};
  int width{};
  int height{};

  TestImage(std::vector<std::uint8_t> pixels, int w, int h)
      : host(std::move(pixels)), width(w), height(h) {
    VERIFY(cudaMalloc(reinterpret_cast<void**>(&device), host.size()) == cudaSuccess);
    VERIFY(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
  }
  ~TestImage() { cudaFree(device); }
  TestImage(const TestImage&) = delete;
  TestImage& operator=(const TestImage&) = delete;

  [[nodiscard]] GrayImageView view(MemoryLocation where) const {
    return {where == MemoryLocation::kHost ? host.data() : device, width, height,
            static_cast<std::size_t>(width), where};
  }
};

GrayImageView view_for(const TestImage& image, const DetectorBackend& backend) {
  return image.view(backend.input_location());
}

struct Parity {
  int frames{};
  int markers{};
  int mismatched_frames{};  // Different ids, counts or overflow.
  double max_corner_diff{};
  int saddles{};
  int mismatched_saddles{};
  double max_saddle_diff{};
};

// Both backends on `image`: markers, then the chessboard corners of `board` when given.
void compare(const TestImage& image, const DetectorOptions& options,
             const std::optional<CharucoBoard>& board, cudaStream_t stream, Parity& parity) {
  auto cpu = create_backend(Backend::kCpu, options, image.width, image.height);
  auto cuda = create_backend(Backend::kCuda, options, image.width, image.height);
  const FrameMarkers a = cpu->detect(view_for(image, *cpu), stream);
  const FrameMarkers b = cuda->detect(view_for(image, *cuda), stream);
  ++parity.frames;
  parity.markers += static_cast<int>(a.markers.size());
  bool same = a.markers.size() == b.markers.size() &&
              a.overflow.candidates == b.overflow.candidates &&
              a.overflow.markers == b.overflow.markers;
  for (std::size_t i = 0; same && i < a.markers.size(); ++i) {
    same = a.markers[i].id == b.markers[i].id && a.markers[i].hamming == b.markers[i].hamming;
    for (int c = 0; c < 4; ++c) {
      parity.max_corner_diff =
          std::max(parity.max_corner_diff,
                   std::hypot(a.markers[i].corners[c][0] - b.markers[i].corners[c][0],
                              a.markers[i].corners[c][1] - b.markers[i].corners[c][1]));
    }
  }
  parity.mismatched_frames += !same;
  if (!board || !same) return;
  std::vector<SaddleSeed> seeds;
  const auto capture = [&](std::span<const SaddleSeed> s) {
    seeds.assign(s.begin(), s.end());
    return cpu->refine_saddles(view_for(image, *cpu), s, stream);
  };
  (void)observe_charuco(*board, a.markers, capture, image.width, image.height, {}, std::nullopt);
  const auto on_cpu = cpu->refine_saddles(view_for(image, *cpu), seeds, stream);
  const auto on_cuda = cuda->refine_saddles(view_for(image, *cuda), seeds, stream);
  VERIFY(on_cpu.size() == seeds.size() && on_cuda.size() == seeds.size());
  for (std::size_t i = 0; i < seeds.size() && i < on_cpu.size() && i < on_cuda.size(); ++i) {
    ++parity.saddles;
    if (on_cpu[i].has_value() != on_cuda[i].has_value()) {
      ++parity.mismatched_saddles;
    } else if (on_cpu[i]) {
      parity.max_saddle_diff = std::max(
          parity.max_saddle_diff,
          std::hypot((*on_cpu[i])[0] - (*on_cuda[i])[0], (*on_cpu[i])[1] - (*on_cuda[i])[1]));
    }
  }
}

void check_parity() {
  cudaStream_t stream = nullptr;
  VERIFY(cudaStreamCreate(&stream) == cudaSuccess);
  Parity parity;
  for (const char* name : {"DICT_4X4_50", "DICT_6X6_250", "DICT_7X7_1000", "DICT_APRILTAG_36h11"}) {
    for (std::uint32_t frame = 0; frame < 4; ++frame) {
      SyntheticScene scene = make_marker_scene(name, frame * 17, kWidth, kHeight);
      scene.blur_sigma = 0.5 * (frame % 3);
      scene.noise_sigma = 2.0 + frame;
      scene.mirrored = frame == 3;
      if (frame == 2) {
        const auto c = marker_truth(scene, scene.markers[0]);
        scene.occluders.push_back({{c[0], c[1], {c[1][0], c[2][1]}, {c[0][0], c[2][1]}}, 120});
      }
      const TestImage image(render_scene(scene, 1), kWidth, kHeight);
      compare(image, DetectorOptions{.dictionary = name}, std::nullopt, stream, parity);
    }
  }
  for (const CharucoBoardConfig& config :
       {CharucoBoardConfig{}, CharucoBoardConfig{.squares_x = 9,
                                                 .squares_y = 6,
                                                 .square_length = 0.03,
                                                 .marker_length = 0.022,
                                                 .dictionary = "DICT_5X5_100"}}) {
    const CharucoBoard board(config);
    for (std::uint32_t frame = 0; frame < 4; ++frame) {
      SyntheticScene scene = make_board_scene(config, frame * 13, kWidth, kHeight);
      scene.blur_sigma = 0.4 * frame;
      const TestImage image(render_scene(scene, 1), kWidth, kHeight);
      compare(image, DetectorOptions{.dictionary = config.dictionary}, board, stream, parity);
    }
  }
  cudaStreamDestroy(stream);
  std::cout << "  parity: " << parity.frames << " frames, " << parity.markers << " markers, "
            << parity.mismatched_frames << " frames differ, corners differ by up to "
            << parity.max_corner_diff << " px; " << parity.saddles << " chessboard corners, "
            << parity.mismatched_saddles << " differ, by up to " << parity.max_saddle_diff
            << " px\n";
  VERIFY(parity.markers > 100 && parity.saddles > 100);
  VERIFY(parity.mismatched_frames == 0 && parity.mismatched_saddles == 0);
  VERIFY(parity.max_corner_diff < 1e-3 && parity.max_saddle_diff < 1e-3);
}

// The cuda backend against ground truth, markers and a board.
void check_cuda_accuracy() {
  auto cuda = create_backend(Backend::kCuda, DetectorOptions{.dictionary = "DICT_5X5_100"}, kWidth,
                             kHeight);
  int missed = 0;
  int extra = 0;
  double max_corner = 0.0;
  double max_translation = 0.0;
  for (std::uint32_t frame = 0; frame < 6; ++frame) {
    const SyntheticScene scene = make_marker_scene("DICT_5X5_100", frame * 6, kWidth, kHeight);
    const TestImage image(render_scene(scene, 1), kWidth, kHeight);
    std::vector<Marker> markers = cuda->detect(view_for(image, *cuda), nullptr).markers;
    for (Marker& m : markers) m.pose = marker_pose(m, scene.intrinsics, 0.1);
    const TruthCounts counts = match_markers(scene, markers, [&](const MarkerMatch& match) {
      for (const double e : match.corner_errors) max_corner = std::max(max_corner, e);
      if (match.translation_error) {
        max_translation = std::max(max_translation, *match.translation_error);
      }
    });
    missed += counts.missed;
    extra += counts.extra;
  }
  std::cout << "  cuda markers: missed " << missed << ", extra " << extra << ", corner max "
            << max_corner << " px, translation max " << 100 * max_translation << "%\n";
  VERIFY(missed == 0 && extra == 0 && max_corner < 0.3 && max_translation < 0.005);

  const CharucoBoardConfig config{};
  const CharucoBoard board(config);
  auto board_backend = create_backend(Backend::kCuda, DetectorOptions{}, kWidth, kHeight);
  const SyntheticScene scene = make_board_scene(config, 5, kWidth, kHeight);
  const TestImage image(render_scene(scene, 1), kWidth, kHeight);
  const GrayImageView view = view_for(image, *board_backend);
  const FrameMarkers found = board_backend->detect(view, nullptr);
  const CharucoObservation observation = observe_charuco(
      board, found.markers,
      [&](std::span<const SaddleSeed> seeds) {
        return board_backend->refine_saddles(view, seeds, nullptr);
      },
      kWidth, kHeight, {}, scene.intrinsics);
  const auto truth = board_truth(scene);
  VERIFY(observation.corners.size() == truth.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < observation.corners.size() && i < truth.size(); ++i) {
    VERIFY(observation.corners[i].id == truth[i].id);
    worst = std::max(worst, std::hypot(observation.corners[i].image[0] - truth[i].image[0],
                                       observation.corners[i].image[1] - truth[i].image[1]));
  }
  std::cout << "  cuda board: " << observation.corners.size() << " corners, max " << worst
            << " px\n";
  VERIFY(worst < 0.3 && observation.pose.has_value());
}

// Thousands of small dark squares: more candidates than the capacity, reported as dropped.
void check_overflow() {
  std::vector<std::uint8_t> dots(static_cast<std::size_t>(kWidth) * kHeight, 220);
  for (int y = 2; y + 6 < kHeight; y += 9) {
    for (int x = 2; x + 6 < kWidth; x += 9) {
      for (int dy = 0; dy < 6; ++dy) {
        for (int dx = 0; dx < 6; ++dx)
          dots[static_cast<std::size_t>(y + dy) * kWidth + x + dx] = 30;
      }
    }
  }
  const TestImage image(std::move(dots), kWidth, kHeight);
  for (const Backend kind : available_backends()) {
    auto backend = create_backend(kind, DetectorOptions{.min_marker_px = 4}, kWidth, kHeight);
    const FrameMarkers found = backend->detect(view_for(image, *backend), nullptr);
    std::cout << "  " << backend->name() << ": " << found.overflow.candidates
              << " candidates dropped\n";
    VERIFY(found.overflow.truncated() && found.markers.empty());
  }
}

// On a CI runner, auto is cuda: every lane has a GPU.
void check_auto() {
  const Backend chosen = resolve_backend(Backend::kAuto);
  VERIFY(chosen == Backend::kCuda || chosen == Backend::kCpu);
  if (std::getenv("HOLOSCAN_CI_PLATFORM") != nullptr) VERIFY(chosen == Backend::kCuda);
  VERIFY(create_backend(Backend::kAuto, DetectorOptions{}, 640, 480)->name() ==
         backend_name(chosen));
}

// Milliseconds per frame, the mean over `runs` after one warm-up.
template <typename Body>
double time_ms(int runs, const Body& body) {
  body();
  cudaDeviceSynchronize();
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) body();
  cudaDeviceSynchronize();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
             .count() /
         runs;
}

void report_timings() {
  cudaStream_t stream = nullptr;
  VERIFY(cudaStreamCreate(&stream) == cudaSuccess);
  const TestImage markers(render_scene(make_marker_scene("DICT_4X4_50", 0, kWidth, kHeight), 1),
                          kWidth, kHeight);
  const CharucoBoardConfig config{.squares_x = 9,
                                  .squares_y = 6,
                                  .square_length = 0.03,
                                  .marker_length = 0.022,
                                  .dictionary = "DICT_5X5_100"};
  const CharucoBoard board(config);
  const TestImage boards(render_scene(make_board_scene(config, 0, kWidth, kHeight), 1), kWidth,
                         kHeight);
  std::cout << std::fixed << std::setprecision(2);
  for (const Backend kind : available_backends()) {
    auto aruco = create_backend(kind, DetectorOptions{}, kWidth, kHeight);
    auto charuco =
        create_backend(kind, DetectorOptions{.dictionary = config.dictionary}, kWidth, kHeight);
    constexpr int kRuns = 20;
    const double aruco_ms =
        time_ms(kRuns, [&] { (void)aruco->detect(view_for(markers, *aruco), stream); });
    const double charuco_ms = time_ms(kRuns, [&] {
      const GrayImageView view = view_for(boards, *charuco);
      const FrameMarkers found = charuco->detect(view, stream);
      (void)observe_charuco(
          board, found.markers,
          [&](std::span<const SaddleSeed> seeds) {
            return charuco->refine_saddles(view, seeds, stream);
          },
          kWidth, kHeight, {}, std::nullopt);
    });
    std::cout << "  " << aruco->name() << " at 1280x720: six markers " << aruco_ms
              << " ms, 9x6 ChArUco board (27 markers, 40 corners) " << charuco_ms << " ms\n";
  }
  cudaStreamDestroy(stream);
}

}  // namespace
}  // namespace holoscan::fiducials

int main() {
  namespace f = holoscan::fiducials;
  return f::test::run({
      {"cuda and cpu agree", f::check_parity},
      {"cuda against ground truth", f::check_cuda_accuracy},
      {"candidate overflow is reported", f::check_overflow},
      {"auto backend", f::check_auto},
      {"timings at 1280x720", f::report_timings},
  });
}
