// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The GPU Stitcher on the synthetic world: bird's-eye views and panoramas
// against their directly rendered truth (PSNR and checkerboard corners), the
// tables and the blend against the CPU reference, exposure compensation
// recovering known gains, seam visibility, determinism, input formats, and
// the edge cases: a camera that sees nothing, cameras that see the same.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/stitching/remap.hpp>
#include <holoscan/stitching/stitcher.hpp>
#include <holoscan/stitching/synthetic_world.hpp>

#include "gpu_support.hpp"
#include "test_support.hpp"

namespace holoscan::stitching {
namespace {

using test::render_frame;
using test::RigFrame;
using test::stitch_frame;
using test::truth_image;

const SyntheticWorld kWorld{};
// The demo's sizes: cameras at 1280x720, a 1024x1024 bird's-eye view and a
// 4096x1024 panorama.
constexpr int kCameraWidth = 1280;
constexpr int kCameraHeight = 720;

// Corners only where the cameras resolve at least a quarter of a camera pixel
// per output pixel (resolution_map()).
Comparison compare(const std::vector<std::uint8_t>& stitched,
                   const std::vector<std::uint8_t>& truth, const OutputGeometry& g,
                   const RigPose& pose, const std::vector<float>& resolution) {
  const std::vector<Point2> corners = checker_corners(kWorld, g, pose);
  const std::size_t pitch = static_cast<std::size_t>(g.width) * 4U;
  return compare_with_truth(stitched.data(), pitch, truth.data(), pitch, g.width, g.height, corners,
                            2, resolution);
}

void print(const std::string& what, const Comparison& c) {
  std::cout << std::fixed << std::setprecision(3) << "  " << what << ": psnr=" << c.psnr_db
            << " dB gain=" << c.gain << " coverage=" << c.coverage
            << " pixels=" << c.compared_pixels << " corners=" << c.corners
            << " max_corner=" << c.max_corner_error << " px mean_corner=" << c.mean_corner_error
            << " px\n";
}

// Stitches the surround rig at `pose` into `g` and checks it against the truth.
Comparison stitch_and_compare(const CameraRig& rig, const OutputGeometry& g, const RigPose& pose,
                              const BlendSettings& settings = {}) {
  Stitcher stitcher(rig, g, settings);
  const auto frame = render_frame(kWorld, rig, pose, PixelFormat::kRgba8);
  return compare(stitch_frame(stitcher, frame), truth_image(kWorld, g, pose), g, pose,
                 resolution_map(stitcher.download_tables()));
}

void check_against_truth(const std::string& what, const Comparison& c, double min_psnr,
                         double min_coverage, int min_corners) {
  print(what, c);
  VERIFY_MSG(c.psnr_db >= min_psnr, what + ": PSNR " + std::to_string(c.psnr_db));
  VERIFY_MSG(std::abs(c.gain - 1.0) < 0.02, what + ": gain " + std::to_string(c.gain));
  VERIFY_MSG(c.coverage >= min_coverage, what + ": coverage " + std::to_string(c.coverage));
  VERIFY_MSG(c.corners >= min_corners, what + ": corners " + std::to_string(c.corners));
  VERIFY_MSG(c.max_corner_error <= 1.0,
             what + ": a corner is " + std::to_string(c.max_corner_error) + " px off");
}

void birds_eye_matches_truth() {
  const CameraRig rig = surround_rig(kCameraWidth, kCameraHeight);
  const OutputGeometry g = ground_view(1024, 1024, 5.5);
  for (const RigPose pose : {RigPose{}, moving_pose(37), moving_pose(250)}) {
    check_against_truth("birds_eye yaw " + std::to_string(pose.yaw_deg),
                        stitch_and_compare(rig, g, pose), 27.0, 0.95, 200);
  }
}

void cylindrical_panorama_matches_truth() {
  const CameraRig rig = surround_rig(kCameraWidth, kCameraHeight);
  const OutputGeometry g = wall_panorama(kWorld, Projection::kCylindrical, 4096, 1024);
  for (const RigPose pose : {RigPose{}, moving_pose(123)}) {
    check_against_truth("cylindrical surround yaw " + std::to_string(pose.yaw_deg),
                        stitch_and_compare(rig, g, pose), 30.0, 0.95, 400);
  }
}

void spherical_panorama_matches_truth() {
  // Six pinholes sharing (nearly) a centre, as panorama rigs do; the scene
  // assumed on the sphere of the wall's radius.
  const CameraRig rig = ring_rig(6, kCameraWidth, kCameraHeight, 80.0, 0.05);
  const OutputGeometry g = wall_panorama(kWorld, Projection::kSpherical, 4096, 1024);
  check_against_truth("spherical ring", stitch_and_compare(rig, g, RigPose{}), 35.0, 0.9, 400);
  // A concentric rig at infinity: only the cameras' rotations count.
  const CameraRig concentric = ring_rig(5, kCameraWidth, kCameraHeight, 90.0, 0.0);
  OutputGeometry infinite = wall_panorama(kWorld, Projection::kCylindrical, 4096, 1024);
  infinite.panorama.radius = 0.0;
  check_against_truth("cylindrical concentric at infinity",
                      stitch_and_compare(concentric, infinite, moving_pose(60)), 33.0, 0.99, 400);
}

void partial_panorama() {
  // A 120-degree panorama looking left, which no camera fully covers.
  const CameraRig rig = ring_rig(4, kCameraWidth, kCameraHeight, 100.0, 0.05);
  OutputGeometry g = wall_panorama(kWorld, Projection::kCylindrical, 1024, 512);
  g.panorama.azimuth_center_deg = 90.0;
  g.panorama.horizontal_fov_deg = 120.0;
  check_against_truth("cylindrical 120 degrees", stitch_and_compare(rig, g, RigPose{}), 30.0, 0.95,
                      40);
}

// The largest difference of two tables' entries, where both see the pixel,
// and whether they see the same pixels.
void check_tables(const RemapTables& gpu, const RemapTables& cpu) {
  VERIFY(gpu.cameras.size() == cpu.cameras.size());
  double worst_uv = 0.0;
  double worst_margin = 0.0;
  std::size_t disagree = 0;
  std::size_t seen = 0;
  for (std::size_t c = 0; c < gpu.cameras.size() && c < cpu.cameras.size(); ++c) {
    const CameraTable& a = gpu.cameras[c];
    const CameraTable& b = cpu.cameras[c];
    VERIFY_MSG(a.x0 == b.x0 && a.y0 == b.y0 && a.width == b.width && a.height == b.height,
               "camera " + std::to_string(c) + ": the regions of interest differ");
    if (a.margin.size() != b.margin.size()) continue;
    for (std::size_t i = 0; i < a.margin.size(); ++i) {
      const bool sa = a.margin[i] > 0.0F;
      const bool sb = b.margin[i] > 0.0F;
      seen += sa;
      if (sa != sb) {
        // Only where the margin is a rounding away from 0.
        ++disagree;
        VERIFY(std::max(a.margin[i], b.margin[i]) < 1e-3F);
        continue;
      }
      if (!sa) continue;
      worst_uv = std::max({worst_uv, std::abs(double{a.uv[2 * i]} - b.uv[2 * i]),
                           std::abs(double{a.uv[2 * i + 1]} - b.uv[2 * i + 1])});
      worst_margin = std::max(worst_margin, std::abs(double{a.margin[i]} - b.margin[i]));
    }
  }
  std::cout << "  tables: entries=" << seen << " max_uv_difference=" << worst_uv
            << " px max_margin_difference=" << worst_margin << " px disagreeing=" << disagree
            << '\n';
  VERIFY(seen > 0);
  VERIFY(worst_uv < 1e-3);
  VERIFY(worst_margin < 1e-3);
  VERIFY(disagree <= seen / 100000 + 2);
}

void remap_matches_cpu_reference() {
  // The fisheye surround rig and the pinhole ring, every projection, small.
  struct Case {
    std::string name;
    CameraRig rig;
    OutputGeometry geometry;
    BlendSettings settings;
  };
  BlendSettings hard;
  hard.mode = BlendMode::kHardSeam;
  hard.output_format = PixelFormat::kRgb8;
  hard.background = {10, 20, 30};
  BlendSettings feathered;
  feathered.feather_width = 25.0F;
  feathered.exposure_compensation = true;
  OutputGeometry spherical = wall_panorama(kWorld, Projection::kSpherical, 700, 180);
  spherical.panorama.radius = 0.0;
  const std::vector<Case> cases{
      {"birds_eye surround", surround_rig(320, 180), ground_view(301, 257, 5.0), BlendSettings{}},
      {"cylindrical surround", surround_rig(320, 180),
       wall_panorama(kWorld, Projection::kCylindrical, 800, 160), hard},
      {"spherical ring at infinity", ring_rig(5, 320, 180, 90.0, 0.1), spherical, feathered},
  };
  const std::vector<float> gains{1.0F, 0.85F, 1.15F, 0.95F, 1.05F};
  for (const Case& test_case : cases) {
    std::cout << "  " << test_case.name << '\n';
    Stitcher stitcher(test_case.rig, test_case.geometry, test_case.settings);
    const RemapTables gpu_tables = stitcher.download_tables();
    const RemapTables cpu_tables = build_remap_tables(test_case.rig, test_case.geometry);
    check_tables(gpu_tables, cpu_tables);
    const auto frame = render_frame(
        kWorld, test_case.rig, moving_pose(77), PixelFormat::kRgb8,
        std::vector<float>(gains.begin(), gains.begin() + test_case.rig.cameras.size()));
    const std::vector<std::uint8_t> got = stitch_frame(stitcher, frame);
    // The GPU's gains equal the reference's from the same tables.
    std::vector<float> want_gains;
    if (test_case.settings.exposure_compensation) {
      want_gains = reference_gains(reference_overlap_stats(gpu_tables, frame.host_images),
                                   test_case.settings, {});
      const std::vector<float> got_gains = stitcher.gains(nullptr);
      VERIFY_MSG(got_gains == want_gains, test_case.name + ": the gains differ");
    }
    const OutputGeometry& g = test_case.geometry;
    const std::size_t pitch =
        static_cast<std::size_t>(g.width) * channels(test_case.settings.output_format);
    // Byte for byte from the GPU's own tables.
    std::vector<std::uint8_t> want(pitch * g.height);
    reference_stitch(gpu_tables, frame.host_images, test_case.settings, want_gains, want.data(),
                     pitch);
    VERIFY_MSG(got == want, test_case.name + ": the GPU blend differs from the reference");
    // From the CPU's tables, within a level where an entry rounds differently.
    std::vector<std::uint8_t> from_cpu(pitch * g.height);
    reference_stitch(cpu_tables, frame.host_images, test_case.settings, want_gains, from_cpu.data(),
                     pitch);
    std::size_t differing = 0;
    int worst = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      const int d = std::abs(static_cast<int>(got[i]) - static_cast<int>(from_cpu[i]));
      differing += d != 0;
      worst = std::max(worst, d);
    }
    std::cout << "    from CPU tables: differing bytes=" << differing << '/' << got.size()
              << " largest=" << worst << '\n';
    VERIFY(differing <= got.size() / 1000);
    VERIFY(worst <= 2);
  }
}

// How visible the seams are: the mean luminance gradient of the stitch along
// the hard seams (where the camera with the largest margin changes between
// neighbouring pixels), at those whose truth is flat within 3 pixels, so that
// the only edge there is one the stitch made.
double seam_visibility(const RemapTables& tables, const std::vector<std::uint8_t>& stitched,
                       const std::vector<std::uint8_t>& truth) {
  const int w = tables.output_width;
  const int h = tables.output_height;
  std::vector<int> owner(static_cast<std::size_t>(w) * h, -1);
  std::vector<float> best(owner.size(), 0.0F);
  for (std::size_t c = 0; c < tables.cameras.size(); ++c) {
    const CameraTable& t = tables.cameras[c];
    for (int dy = 0; dy < t.height; ++dy) {
      for (int dx = 0; dx < t.width; ++dx) {
        const float m = t.margin[static_cast<std::size_t>(dy) * t.width + dx];
        const std::size_t i = static_cast<std::size_t>(t.y0 + dy) * w + (t.x0 + dx) % w;
        if (m > best[i]) {
          owner[i] = static_cast<int>(c);
          best[i] = m;
        }
      }
    }
  }
  const auto luma = [w](const std::vector<std::uint8_t>& image, int x, int y) {
    const std::uint8_t* p = image.data() + (static_cast<std::size_t>(y) * w + x) * 4U;
    return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  };
  const auto gradient = [&luma](const std::vector<std::uint8_t>& image, int x, int y) {
    return std::hypot(luma(image, x + 1, y) - luma(image, x - 1, y),
                      luma(image, x, y + 1) - luma(image, x, y - 1));
  };
  constexpr int kFlat = 3;
  double sum = 0.0;
  std::size_t count = 0;
  for (int y = kFlat + 1; y < h - kFlat - 1; ++y) {
    for (int x = kFlat + 1; x < w - kFlat - 1; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * w + x;
      if (owner[i] < 0 || (owner[i] == owner[i + 1] && owner[i] == owner[i + w])) continue;
      bool flat = true;
      for (int dy = -kFlat; dy <= kFlat && flat; ++dy) {
        for (int dx = -kFlat; dx <= kFlat && flat; ++dx) {
          const std::size_t j = (static_cast<std::size_t>(y + dy) * w + x + dx) * 4U;
          flat = truth[j + 3] == 255 && stitched[j + 3] == 255 &&
                 gradient(truth, x + dx, y + dy) < 2.0;
        }
      }
      if (!flat) continue;
      sum += gradient(stitched, x, y);
      ++count;
    }
  }
  return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

void exposure_compensation_and_seams() {
  const CameraRig rig = surround_rig(kCameraWidth, kCameraHeight);
  const std::vector<float> exposure{1.0F, 0.8F, 1.2F, 0.9F};
  const RigPose pose = moving_pose(20);
  for (const OutputGeometry& g : {ground_view(1024, 1024, 5.5),
                                  wall_panorama(kWorld, Projection::kCylindrical, 4096, 1024)}) {
    const std::string name(projection_name(g.projection));
    const auto frame = render_frame(kWorld, rig, pose, PixelFormat::kRgba8, exposure);
    const auto truth = truth_image(kWorld, g, pose);
    BlendSettings hard;
    hard.mode = BlendMode::kHardSeam;
    BlendSettings feather;
    BlendSettings compensated;
    compensated.exposure_compensation = true;
    BlendSettings compensated_hard = hard;
    compensated_hard.exposure_compensation = true;
    double visibility[4];
    int k = 0;
    std::vector<float> gains;
    for (const BlendSettings& settings : {hard, feather, compensated_hard, compensated}) {
      Stitcher stitcher(rig, g, settings);
      const auto stitched = stitch_frame(stitcher, frame);
      visibility[k++] = seam_visibility(stitcher.download_tables(), stitched, truth);
      if (settings.exposure_compensation) gains = stitcher.gains(nullptr);
    }
    std::cout << std::fixed << std::setprecision(3) << "  " << name
              << ": seam visibility hard=" << visibility[0] << " feather=" << visibility[1]
              << " hard+exposure=" << visibility[2] << " feather+exposure=" << visibility[3]
              << '\n';
    // The gains undo the exposures: gain * exposure is the same for all.
    double lo = 1e9;
    double hi = 0.0;
    std::cout << "  " << name << ": gains";
    for (std::size_t c = 0; c < gains.size(); ++c) {
      const double product = gains[c] * exposure[c];
      lo = std::min(lo, product);
      hi = std::max(hi, product);
      std::cout << ' ' << gains[c] << " (x" << exposure[c] << " = " << product << ')';
    }
    std::cout << " spread=" << (hi - lo) / lo << '\n';
    // The panorama's overlaps include ground the cameras see with parallax.
    const double tolerance = g.projection == Projection::kBirdsEye ? 0.01 : 0.03;
    VERIFY_MSG((hi - lo) / lo < tolerance, name + ": the gains do not undo the exposures");
    VERIFY(visibility[1] < visibility[0]);
    VERIFY(visibility[2] < 0.5 * visibility[0]);
    VERIFY(visibility[3] < visibility[1]);
    VERIFY(visibility[3] <= visibility[2]);
  }
}

void exposure_without_differences() {
  // Equal exposures give gains of 1, within the overlaps' noise.
  const CameraRig rig = surround_rig(kCameraWidth, kCameraHeight);
  BlendSettings settings;
  settings.exposure_compensation = true;
  Stitcher stitcher(rig, ground_view(1024, 1024, 5.5), settings);
  (void)stitch_frame(stitcher, render_frame(kWorld, rig, RigPose{}, PixelFormat::kRgba8));
  for (const float g : stitcher.gains(nullptr)) {
    VERIFY_MSG(std::abs(g - 1.0F) < 0.02F, "gain " + std::to_string(g));
  }
  // Smoothing moves the gains part of the way from the previous frame's.
  settings.exposure_smoothing = 0.75F;
  Stitcher smoothed(rig, ground_view(256, 256, 5.5), settings);
  const auto frame = render_frame(kWorld, rig, RigPose{}, PixelFormat::kRgba8,
                                  std::vector<float>{1.0F, 0.6F, 1.0F, 1.0F});
  (void)stitch_frame(smoothed, frame);
  const float first = smoothed.gains(nullptr)[1];
  (void)stitch_frame(smoothed, frame);
  const float second = smoothed.gains(nullptr)[1];
  smoothed.reset_gains(nullptr);
  VERIFY(smoothed.gains(nullptr)[1] == 1.0F);
  std::cout << "  smoothed gain of the dark camera: " << first << " then " << second << '\n';
  VERIFY(first > 1.0F && second > first);
}

void deterministic() {
  const CameraRig rig = surround_rig(kCameraWidth, kCameraHeight);
  BlendSettings settings;
  settings.exposure_compensation = true;
  const auto frame = render_frame(kWorld, rig, moving_pose(5), PixelFormat::kRgb8,
                                  std::vector<float>{1.1F, 0.9F, 1.0F, 0.95F});
  for (const OutputGeometry& g : {ground_view(1024, 1024, 5.5),
                                  wall_panorama(kWorld, Projection::kCylindrical, 4096, 1024)}) {
    Stitcher a(rig, g, settings);
    Stitcher b(rig, g, settings);
    const auto first = stitch_frame(a, frame);
    const auto second = stitch_frame(b, frame);
    a.reset_gains(nullptr);
    const auto third = stitch_frame(a, frame);
    VERIFY(first == second);
    VERIFY(first == third);
    VERIFY(a.gains(nullptr) == b.gains(nullptr));
    // The renders too.
    const auto again = render_frame(kWorld, rig, moving_pose(5), PixelFormat::kRgb8,
                                    std::vector<float>{1.1F, 0.9F, 1.0F, 0.95F});
    VERIFY(again.host == frame.host);
  }
}

void input_formats() {
  // Every input format, padded rows, and RGB8 output: the same picture as
  // RGBA8 input (mono only matches its own luminance).
  const CameraRig rig = surround_rig(320, 180);
  const OutputGeometry g = ground_view(256, 256, 5.5);
  BlendSettings settings;
  settings.output_format = PixelFormat::kRgb8;
  Stitcher stitcher(rig, g, settings);
  const auto reference =
      stitch_frame(stitcher, render_frame(kWorld, rig, RigPose{}, PixelFormat::kRgba8));
  for (const PixelFormat format : {PixelFormat::kRgb8, PixelFormat::kBgr8, PixelFormat::kBgra8}) {
    for (const std::size_t pad : {std::size_t{0}, std::size_t{13}}) {
      const auto got =
          stitch_frame(stitcher, render_frame(kWorld, rig, RigPose{}, format, {}, 3, pad));
      VERIFY_MSG(got == reference, std::string(format_name(format)) + " pad " +
                                       std::to_string(pad) + " differs from rgba8");
    }
  }
  const auto mono =
      stitch_frame(stitcher, render_frame(kWorld, rig, RigPose{}, PixelFormat::kMono8));
  bool gray = true;
  std::size_t lit = 0;
  for (std::size_t i = 0; i < mono.size(); i += 3) {
    gray = gray && mono[i] == mono[i + 1] && mono[i] == mono[i + 2];
    lit += mono[i] != 0;
  }
  VERIFY(gray);
  VERIFY(lit > mono.size() / 6);
}

void camera_that_sees_nothing() {
  // A fifth camera looking straight up adds nothing to a bird's-eye view.
  const CameraRig rig = surround_rig(320, 180);
  CameraRig with_sky = rig;
  Camera up = ring_rig(1, 320, 180, 60.0, 0.0).cameras[0];
  up.name = "sky";
  up.camera_from_rig = mount({0.0, 0.0, 1.5}, 0.0, 90.0);
  with_sky.cameras.push_back(up);
  const OutputGeometry g = ground_view(256, 256, 5.5);
  Stitcher four(rig, g, BlendSettings{});
  Stitcher five(with_sky, g, BlendSettings{});
  VERIFY(five.covered_pixels().back() == 0);
  const RemapTables tables = five.download_tables();
  VERIFY(tables.cameras.back().width == 0 && tables.cameras.back().uv.empty());
  const auto frame4 = render_frame(kWorld, rig, RigPose{}, PixelFormat::kRgb8);
  const auto frame5 = render_frame(kWorld, with_sky, RigPose{}, PixelFormat::kRgb8);
  VERIFY(stitch_frame(four, frame4) == stitch_frame(five, frame5));
  // With exposure compensation too: it has no overlap, so it keeps gain 1.
  BlendSettings settings;
  settings.exposure_compensation = true;
  Stitcher compensated(with_sky, g, settings);
  (void)stitch_frame(compensated, frame5);
  VERIFY(compensated.gains(nullptr).back() == 1.0F);
  // A rig none of whose cameras sees the output gives the background.
  CameraRig only_sky;
  only_sky.cameras.push_back(up);
  BlendSettings background;
  background.background = {1, 2, 3};
  Stitcher nothing(only_sky, g, background);
  const RigFrame sky_frame = render_frame(kWorld, only_sky, RigPose{}, PixelFormat::kRgb8);
  const auto empty = stitch_frame(nothing, sky_frame);
  bool all_background = true;
  for (std::size_t i = 0; i < empty.size(); i += 4) {
    all_background = all_background && empty[i] == 1 && empty[i + 1] == 2 && empty[i + 2] == 3 &&
                     empty[i + 3] == 0;
  }
  VERIFY(all_background);
}

void cameras_that_see_the_same() {
  // Two copies of one camera: the blend equals that camera alone, and with
  // different exposures, compensation brings them together.
  const CameraRig ring = ring_rig(1, 640, 360, 90.0, 0.0);
  CameraRig twins = ring;
  twins.cameras.push_back(ring.cameras[0]);
  twins.cameras[1].name = "twin";
  OutputGeometry g = wall_panorama(kWorld, Projection::kCylindrical, 1024, 256);
  g.panorama.horizontal_fov_deg = 120.0;
  Stitcher single(ring, g, BlendSettings{});
  Stitcher pair(twins, g, BlendSettings{});
  const auto one = stitch_frame(single, render_frame(kWorld, ring, RigPose{}, PixelFormat::kRgb8));
  const auto two = stitch_frame(pair, render_frame(kWorld, twins, RigPose{}, PixelFormat::kRgb8));
  int worst = 0;
  for (std::size_t i = 0; i < one.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<int>(one[i]) - static_cast<int>(two[i])));
  }
  VERIFY_MSG(worst <= 1, "twins differ from one camera by " + std::to_string(worst));
  BlendSettings settings;
  settings.exposure_compensation = true;
  Stitcher compensated(twins, g, settings);
  (void)stitch_frame(compensated, render_frame(kWorld, twins, RigPose{}, PixelFormat::kRgb8,
                                               std::vector<float>{1.0F, 0.5F}));
  const std::vector<float> gains = compensated.gains(nullptr);
  std::cout << "  twins at exposures 1 and 0.5: gains " << gains[0] << ' ' << gains[1] << '\n';
  VERIFY(std::abs(gains[1] * 0.5 / gains[0] - 1.0) < 0.03);
}

void invalid_input() {
  const CameraRig rig = surround_rig(320, 180);
  Stitcher stitcher(rig, ground_view(64, 64, 5.0), BlendSettings{});
  test::DeviceBuffer out(64 * 64 * 4);
  const auto frame = render_frame(kWorld, rig, RigPose{}, PixelFormat::kRgb8);
  std::vector<DeviceImage> images = frame.images;
  VERIFY(test::throws<std::invalid_argument>(
      [&] { (void)stitcher.stitch(std::span(images).first(3), out.data(), 64 * 4, nullptr); }));
  images[2].width = 321;
  VERIFY(test::throws<std::invalid_argument>(
      [&] { (void)stitcher.stitch(images, out.data(), 64 * 4, nullptr); }));
  VERIFY(test::throws<std::invalid_argument>(
      [&] { (void)stitcher.stitch(frame.images, out.data(), 64 * 3, nullptr); }));
  CameraRig nine;
  for (int i = 0; i < 9; ++i) {
    nine.cameras.push_back(ring_rig(1, 32, 32, 60.0, 0.0).cameras[0]);
    nine.cameras.back().name = std::to_string(i);
  }
  VERIFY(test::throws<std::invalid_argument>(
      [&] { Stitcher s(nine, ground_view(64, 64, 5.0), BlendSettings{}); }));
  BlendSettings bad;
  bad.output_format = PixelFormat::kMono8;
  VERIFY(
      test::throws<std::invalid_argument>([&] { Stitcher s(rig, ground_view(64, 64, 5.0), bad); }));
}

}  // namespace
}  // namespace holoscan::stitching

int main() {
  namespace st = holoscan::stitching;
  if (!st::test::have_gpu()) {
    std::cout << "no CUDA device; skipped\n";
    return 77;
  }
  return st::test::run({
      {"birds_eye_matches_truth", st::birds_eye_matches_truth},
      {"cylindrical_panorama_matches_truth", st::cylindrical_panorama_matches_truth},
      {"spherical_panorama_matches_truth", st::spherical_panorama_matches_truth},
      {"partial_panorama", st::partial_panorama},
      {"remap_matches_cpu_reference", st::remap_matches_cpu_reference},
      {"exposure_compensation_and_seams", st::exposure_compensation_and_seams},
      {"exposure_without_differences", st::exposure_without_differences},
      {"deterministic", st::deterministic},
      {"input_formats", st::input_formats},
      {"camera_that_sees_nothing", st::camera_that_sees_nothing},
      {"cameras_that_see_the_same", st::cameras_that_see_the_same},
      {"invalid_input", st::invalid_input},
  });
}
