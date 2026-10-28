// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Every compiled backend against the synthetic scene's ground truth: each tag
// found with its id, corners in order and in place, poses close, and nothing
// else reported. Also checks the CUDA backend against the CPU reference on tags
// cut by the image edge, that every backend reports the work it drops when a
// frame is over one of its capacities, and what the auto backend resolves to.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/apriltag/detector_backend.hpp>
#include <holoscan/apriltag/pose.hpp>
#include <holoscan/apriltag/synthetic_scene.hpp>

#include "gray_converter.hpp"
#include "test_support.hpp"

namespace holoscan::apriltag {
namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr std::uint32_t kFrames = 24;

struct Errors {
  int frames{};
  int expected{};
  int found{};
  int extra{};
  int wrong_order{};
  double max_corner_px{};
  double sum_corner_px{};
  int corners{};
  double max_translation_rel{};
  double max_rotation_deg{};
  double detect_ms{};
  int overflowed{};  // Frames that reported any overflow.
};

double rotation_error_deg(const std::array<double, 9>& a, const std::array<double, 9>& b) {
  // Angle of a^T b.
  double trace = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      trace += a[k * 3 + i] * b[k * 3 + i];
    }
  }
  return std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / M_PI;
}

FrameDetections run_backend(DetectorBackend& backend, const std::vector<std::uint8_t>& gray,
                            GrayConverter& converter, cudaStream_t stream,
                            std::uint8_t* device_copy, bool device_input) {
  ImageView view{gray.data(), kWidth, kHeight, kWidth, PixelFormat::kMono8, MemoryLocation::kHost};
  if (device_input) {
    cudaMemcpy(device_copy, gray.data(), gray.size(), cudaMemcpyHostToDevice);
    view.data = device_copy;
    view.location = MemoryLocation::kDevice;
  }
  return backend.detect(converter.convert(view, backend.input_location(), stream), stream);
}

Errors evaluate(Backend kind, const std::string& family, bool device_input, int max_hamming) {
  DetectorOptions options;
  options.family = family;
  options.max_hamming = max_hamming;
  auto backend = create_backend(kind, options, kWidth, kHeight);
  GrayConverter converter;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  std::uint8_t* device_copy = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&device_copy), static_cast<std::size_t>(kWidth) * kHeight);

  Errors errors;
  for (std::uint32_t frame = 0; frame < kFrames; ++frame) {
    const SyntheticScene scene = make_scene(family, frame, kWidth, kHeight);
    const auto gray = render_scene(scene, 1);
    const auto start = std::chrono::steady_clock::now();
    FrameDetections result =
        run_backend(*backend, gray, converter, stream, device_copy, device_input);
    std::vector<Detection>& detections = result.tags;
    const Overflow& overflow = result.overflow;
    if (overflow.truncated() || overflow.detections != 0) ++errors.overflowed;
    errors.detect_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    ++errors.frames;
    errors.expected += static_cast<int>(scene.tags.size());
    for (Detection& detection : detections) {
      detection.pose = estimate_pose(detection, scene.intrinsics, scene.tag_size);
    }
    errors.extra +=
        match_truth(scene, detections, [&errors](const TruthMatch& match) {
          ++errors.found;
          double worst = 0.0;
          for (const double e : match.corner_errors) {
            worst = std::max(worst, e);
            errors.sum_corner_px += e;
            ++errors.corners;
          }
          if (worst > 5.0) ++errors.wrong_order;
          errors.max_corner_px = std::max(errors.max_corner_px, worst);
          if (!match.translation_error) {
            errors.max_translation_rel = HUGE_VAL;
            return;
          }
          errors.max_translation_rel =
              std::max(errors.max_translation_rel, *match.translation_error);
          errors.max_rotation_deg =
              std::max(errors.max_rotation_deg,
                       rotation_error_deg(match.detection.pose->rotation, match.tag.pose.rotation));
        }).extra;
  }
  cudaFree(device_copy);
  cudaStreamDestroy(stream);
  std::printf(
      "  %-8s %-17s hamming %d %-6s found %d/%d extra %d | corner mean %.3f max %.3f px | "
      "translation max %.2f%% | rotation max %.2f deg | %.2f ms/frame\n",
      std::string(backend_name(kind)).c_str(), family.c_str(), max_hamming,
      device_input ? "device" : "host", errors.found, errors.expected, errors.extra,
      errors.corners ? errors.sum_corner_px / errors.corners : 0.0, errors.max_corner_px,
      100.0 * errors.max_translation_rel, errors.max_rotation_deg,
      errors.detect_ms / errors.frames);
  return errors;
}

void check_backend(Backend kind, const std::string& family, bool device_input, int max_hamming) {
  const Errors errors = evaluate(kind, family, device_input, max_hamming);
  VERIFY_MSG(errors.found == errors.expected, "every tag is found");
  VERIFY_MSG(errors.extra == 0, "nothing else is reported");
  VERIFY_MSG(errors.overflowed == 0, "no frame is over a capacity");
  VERIFY_MSG(errors.wrong_order == 0, "corners are in the AprilTag order");
  VERIFY_MSG(errors.max_corner_px < 1.0, "corners are within a pixel");
  VERIFY_MSG(errors.max_translation_rel < 0.02, "translation is within 2% of the distance");
  VERIFY_MSG(errors.max_rotation_deg < 3.0, "rotation is within 3 degrees");
}

// The largest error of `detection`'s corners against `truth`.
double worst_corner_error(const Detection& detection, const SceneTag& truth,
                          const SyntheticScene& scene) {
  const auto corners = project_corners(truth, scene.intrinsics, scene.tag_size);
  double worst = 0.0;
  for (int i = 0; i < 4; ++i) {
    worst = std::max(worst, std::hypot(detection.corners[i][0] - corners[i][0],
                                       detection.corners[i][1] - corners[i][1]));
  }
  return worst;
}

// Whether every corner of `truth` is at least `margin` pixels inside the image.
bool fully_visible(const SceneTag& truth, const SyntheticScene& scene, double margin) {
  const auto corners = project_corners(truth, scene.intrinsics, scene.tag_size);
  return std::all_of(corners.begin(), corners.end(), [&](const Point2& c) {
    return c[0] >= margin && c[1] >= margin && c[0] <= kWidth - 1 - margin &&
           c[1] <= kHeight - 1 - margin;
  });
}

// Tags pushed toward the image edges until some are cut off. A clipped tag's
// corners are poorly constrained for every detector (the reference CPU
// backend's err by several pixels), so: fully visible tags must be within a
// pixel; no backend may report a tag that is not there; and the CUDA backend
// must find every tag the reference finds, no less accurately.
void check_edge_tags(Backend kind) {
  DetectorOptions options;
  auto reference = create_backend(Backend::kCpu, options, kWidth, kHeight);
  auto backend = create_backend(kind, options, kWidth, kHeight);
  GrayConverter reference_converter;
  GrayConverter converter;
  int reference_tags = 0;
  int found_tags = 0;
  for (std::uint32_t frame = 0; frame < kFrames; ++frame) {
    SyntheticScene scene = make_scene("tag36h11", frame, kWidth, kHeight);
    for (SceneTag& tag : scene.tags) {
      tag.pose.translation[0] *= 2.1;
      tag.pose.translation[1] *= 1.6;
    }
    const auto gray = render_scene(scene, 1);
    const ImageView view{gray.data(),          kWidth, kHeight, kWidth, PixelFormat::kMono8,
                         MemoryLocation::kHost};
    const auto expected =
        reference
            ->detect(reference_converter.convert(view, MemoryLocation::kHost, nullptr), nullptr)
            .tags;
    const auto found =
        backend->detect(converter.convert(view, backend->input_location(), nullptr), nullptr).tags;
    reference_tags += static_cast<int>(expected.size());
    found_tags += static_cast<int>(found.size());
    const std::string where = "frame " + std::to_string(frame) + ": tag ";
    for (const Detection& got : found) {
      const auto truth = std::find_if(scene.tags.begin(), scene.tags.end(),
                                      [&](const SceneTag& t) { return t.id == got.id; });
      VERIFY_MSG(truth != scene.tags.end(),
                 where + std::to_string(got.id) + " is not in the scene");
      if (truth == scene.tags.end()) continue;
      const double error = worst_corner_error(got, *truth, scene);
      if (fully_visible(*truth, scene, 2.0)) {
        VERIFY_MSG(error < 1.0, where + std::to_string(got.id) + " corners off by " +
                                    std::to_string(error) + " px");
      }
      const auto reference_match = std::find_if(expected.begin(), expected.end(),
                                                [&](const Detection& d) { return d.id == got.id; });
      if (kind == Backend::kCuda && reference_match != expected.end()) {
        const double reference_error = worst_corner_error(*reference_match, *truth, scene);
        VERIFY_MSG(error <= reference_error + 1.0,
                   where + std::to_string(got.id) + " corners off by " + std::to_string(error) +
                       " px, the reference by " + std::to_string(reference_error) + " px");
      }
    }
    if (kind == Backend::kCuda) {
      for (const Detection& want : expected) {
        VERIFY_MSG(std::any_of(found.begin(), found.end(),
                               [&](const Detection& d) { return d.id == want.id; }),
                   where + std::to_string(want.id) + " found by cpu is missing");
      }
    }
  }
  std::printf("  %-8s edge scene: cpu found %d, %s found %d of %u\n",
              std::string(backend_name(kind)).c_str(), reference_tags,
              std::string(backend_name(kind)).c_str(), found_tags, 4 * kFrames);
  // The scene must actually cut tags: fewer than all of them are decodable.
  VERIFY_MSG(reference_tags > 0 && reference_tags < static_cast<int>(4 * kFrames),
             "the edge scene clips some tags");
}

// Vertical black and white stripes one decimated pixel wide: every pixel
// borders the next stripe on its right and both lower diagonals, three boundary
// points per pixel, more than the CUDA backend's buffers hold (two), so the
// excess is dropped and reported. Nothing is a tag; a scene detected afterwards
// still is, with nothing dropped.
void check_dense_edges() {
  DetectorOptions options;
  auto backend = create_backend(Backend::kCuda, options, kWidth, kHeight);
  GrayConverter converter;
  std::vector<std::uint8_t> stripes(static_cast<std::size_t>(kWidth) * kHeight);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      stripes[static_cast<std::size_t>(y) * kWidth + x] = (x / options.decimate) % 2 ? 255 : 0;
    }
  }
  for (int pass = 0; pass < 2; ++pass) {
    const FrameDetections found =
        run_backend(*backend, stripes, converter, nullptr, nullptr, /*device_input=*/false);
    std::printf("  cuda     stripes: %zu tags, dropped %u points\n", found.tags.size(),
                found.overflow.points);
    VERIFY_MSG(found.tags.empty(), std::to_string(found.tags.size()) + " tags found in stripes");
    VERIFY_MSG(found.overflow.points > 0 && found.overflow.truncated(),
               "the dropped boundary points are reported");
  }
  const SyntheticScene scene = make_scene("tag36h11", 0, kWidth, kHeight);
  const FrameDetections found =
      run_backend(*backend, render_scene(scene, 1), converter, nullptr, nullptr, false);
  VERIFY_MSG(found.tags.size() == scene.tags.size(),
             std::to_string(found.tags.size()) + " tags found after the stripes");
  VERIFY_MSG(!found.overflow.truncated(), "nothing is dropped after the stripes");
}

// A checkerboard of `square`-pixel squares: every black square is a quad the
// CUDA backend fits, more than it can fit (small squares) or decode (larger
// ones). Nothing is a tag, and the dropped work is reported.
Overflow checkerboard_overflow(int square) {
  DetectorOptions options;
  auto backend = create_backend(Backend::kCuda, options, kWidth, kHeight);
  GrayConverter converter;
  std::vector<std::uint8_t> board(static_cast<std::size_t>(kWidth) * kHeight);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      board[static_cast<std::size_t>(y) * kWidth + x] = ((x / square + y / square) % 2) ? 230 : 20;
    }
  }
  const FrameDetections found = run_backend(*backend, board, converter, nullptr, nullptr, false);
  std::printf(
      "  cuda     %d px checkerboard: %zu tags, dropped %u points, %u candidates, %u quads\n",
      square, found.tags.size(), found.overflow.points, found.overflow.candidates,
      found.overflow.quads);
  VERIFY_MSG(found.tags.empty(),
             std::to_string(found.tags.size()) + " tags found in a checkerboard");
  VERIFY_MSG(found.overflow.truncated(), "the checkerboard's dropped work is reported");
  return found.overflow;
}

void check_quad_overflow() {
  VERIFY_MSG(checkerboard_overflow(16).quads > 0, "quads past the decoding capacity are counted");
}

void check_candidate_overflow() {
  VERIFY_MSG(checkerboard_overflow(10).candidates > 0,
             "quad candidates past the fitting capacity are counted");
}

// More tags than a frame reports: a grid of fronto-parallel tags. Each backend
// reports the first kMaxDetectionsPerFrame, all of them real, and counts the rest.
void check_detection_overflow(Backend kind) {
  constexpr int kColumns = 12;
  constexpr int kRows = 7;
  SyntheticScene scene = make_scene("tag36h11", 0, kWidth, kHeight);
  scene.tags.clear();
  const double depth = 2.0;
  const double pitch_x = kWidth / scene.intrinsics.fx * depth / kColumns;
  const double pitch_y = kHeight / scene.intrinsics.fy * depth / kRows;
  for (int row = 0; row < kRows; ++row) {
    for (int column = 0; column < kColumns; ++column) {
      SceneTag tag;
      tag.id = row * kColumns + column;
      tag.pose.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      tag.pose.translation = {(column - (kColumns - 1) / 2.0) * pitch_x,
                              (row - (kRows - 1) / 2.0) * pitch_y, depth};
      scene.tags.push_back(tag);
    }
  }
  scene.tag_size = 0.6 * std::min(pitch_x, pitch_y);
  DetectorOptions options;
  auto backend = create_backend(kind, options, kWidth, kHeight);
  GrayConverter converter;
  const FrameDetections found =
      run_backend(*backend, render_scene(scene, 1), converter, nullptr, nullptr, false);
  std::printf("  %-8s %d-tag grid: reported %zu, dropped %u\n",
              std::string(backend_name(kind)).c_str(), kColumns * kRows, found.tags.size(),
              found.overflow.detections);
  const auto excess = static_cast<std::uint32_t>(kColumns * kRows - kMaxDetectionsPerFrame);
  VERIFY_MSG(found.tags.size() == kMaxDetectionsPerFrame,
             std::to_string(found.tags.size()) + " tags reported");
  // VPI discards what does not fit its array without counting it.
  VERIFY_MSG(
      kind == Backend::kVpi ? found.overflow.detections >= 1 : found.overflow.detections == excess,
      std::to_string(found.overflow.detections) + " tags counted as dropped");
  VERIFY_MSG(!found.overflow.truncated(), "nothing before decoding is dropped");
  const TruthCounts counts = match_truth(scene, found.tags, [](const TruthMatch&) {});
  VERIFY_MSG(counts.extra == 0, std::to_string(counts.extra) + " reported tags are not real");
}

// On the Jetson CI runner, the vpi backend must run on the PVA, not fall back to
// VPI's CPU path, for an image the PVA accepts.
void check_vpi_on_pva() {
  const auto backend = create_backend(Backend::kVpi, DetectorOptions{}, kWidth, kHeight);
  VERIFY_MSG(backend->name() == "vpi-pva",
             "the vpi backend runs as " + std::string(backend->name()));
}

// "auto" names Backend::kAuto, which resolves to a compiled backend on this
// platform. On the CI runners: vpi on the Jetson's PVA, cuda on x86_64.
void check_auto(const char* platform) {
  VERIFY(parse_backend("auto") == Backend::kAuto && backend_name(Backend::kAuto) == "auto");
  const PlatformFacts facts = probe_platform();
  const Backend resolved = resolve_backend(Backend::kAuto);
  VERIFY(resolved == resolve_auto(facts));
  const auto available = available_backends();
  VERIFY(std::find(available.begin(), available.end(), Backend::kAuto) == available.end());
  VERIFY(std::find(available.begin(), available.end(), resolved) != available.end());
  const auto backend = create_backend(Backend::kAuto, DetectorOptions{}, kWidth, kHeight);
  const std::string_view expected = platform == nullptr                      ? backend->name()
                                    : std::string_view(platform) == "jetson" ? "vpi-pva"
                                    : std::string_view(platform) == "x86_64" ? "cuda"
                                                                             : backend->name();
  VERIFY_MSG(backend->name() == expected, "auto resolves to " + std::string(backend->name()) +
                                              ", expected " + std::string(expected));
  std::printf("  auto     resolves to %s (vpi compiled %d, PVA %d, CUDA device %d)\n",
              std::string(backend->name()).c_str(), facts.vpi_compiled, facts.has_pva,
              facts.cuda_device);
}

}  // namespace
}  // namespace holoscan::apriltag

int main() {
  using holoscan::apriltag::Backend;
  using holoscan::apriltag::test::TestCase;
  std::vector<TestCase> cases;
  for (Backend backend : holoscan::apriltag::available_backends()) {
    const std::string name(holoscan::apriltag::backend_name(backend));
    // tagCircle21h7 has bits outside its border, at negative coordinates.
    for (const char* family : {"tag36h11", "tagStandard41h12", "tag25h9", "tagCircle21h7"}) {
      for (bool device_input : {false, true}) {
        cases.push_back(
            {name + " " + family + (device_input ? " device input" : " host input"),
             [=] { holoscan::apriltag::check_backend(backend, family, device_input, 1); }});
      }
    }
    // Two corrected bits: the largest decode table.
    cases.push_back({name + " tag25h9 max-hamming 2",
                     [=] { holoscan::apriltag::check_backend(backend, "tag25h9", false, 2); }});
    cases.push_back({name + " more tags than a frame reports",
                     [=] { holoscan::apriltag::check_detection_overflow(backend); }});
  }
  for (Backend backend : holoscan::apriltag::available_backends()) {
    if (backend == Backend::kCpu) continue;
    cases.push_back(
        {std::string(holoscan::apriltag::backend_name(backend)) + " tags cut by the edge",
         [=] { holoscan::apriltag::check_edge_tags(backend); }});
  }
  cases.push_back({"cuda more edges than its buffers hold", holoscan::apriltag::check_dense_edges});
  cases.push_back({"cuda more quads than it decodes", holoscan::apriltag::check_quad_overflow});
  cases.push_back(
      {"cuda more quad candidates than it fits", holoscan::apriltag::check_candidate_overflow});
  const char* platform = std::getenv("HOLOSCAN_CI_PLATFORM");
  cases.push_back(
      {"auto resolves for the platform", [=] { holoscan::apriltag::check_auto(platform); }});
  if (platform != nullptr && std::string_view(platform) == "jetson") {
    cases.push_back({"vpi runs on the PVA", holoscan::apriltag::check_vpi_on_pva});
  }
  return holoscan::apriltag::test::run(cases);
}
