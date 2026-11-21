// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The local recordings: a stereo pair of infrared eyes, side by side, with
// their calibration in left.yaml and right.yaml (ROS camera_info files, read
// with holoscan-messages; the pair's extrinsics come from them too), and a
// colour camera's in rgb.yaml. For each clip, the eyes of one frame are
// stitched into a panorama on the GPU and checked: every file's calibration
// is one the stitcher takes; each eye covers the view and they overlap; the
// stitch and its exposure gains equal the CPU reference's on real images; the
// gains are finite and near 1; and where the eyes overlap, their images
// correlate best when the panorama's surface is at the scene's distance, and
// there far better than at infinity (where the extrinsics do not count) or
// with the baseline turned around: the extrinsics from the files are right.
// The correlation, not the PSNR, measures the agreement: the eyes see the
// projector's sparse dots on a dark scene, whose PSNR (about 28 dB) the
// sensor noise sets, moving by under 1 dB between the right and the turned
// baseline. With the projector off the eyes see almost nothing (a spread of
// half a gray level), and only the agreement is not checked. Reads
// $HOLOSCAN_TEST_DATA/stereo-camera/<clip>/; skipped (exit 77) when the
// variable is unset, failed when it names no recording with a left.yaml.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/stitching/camera_rig.hpp>
#include <holoscan/stitching/projection.hpp>
#include <holoscan/stitching/remap.hpp>
#include <holoscan/stitching/stitcher.hpp>

#include "gpu_support.hpp"
#include "test_support.hpp"

namespace holoscan::stitching {
namespace {

namespace fs = std::filesystem;
namespace hm = holoscan::messages;

constexpr int kSkip = 77;
// The frame of each clip that is stitched: the middle of its 60.
constexpr std::size_t kFrame = 30;
// The panorama around the eyes' common view.
constexpr int kOutputWidth = 1024;
constexpr int kOutputHeight = 512;

struct Gray {
  int width{};
  int height{};
  std::vector<std::uint8_t> pixels;
};

// An 8-bit binary PGM (P5), as the recordings hold the infrared eyes.
Gray read_pgm(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error(path.string() + ": cannot open");
  std::string magic;
  file >> magic;
  int values[3] = {};
  for (int& value : values) {
    file >> std::ws;
    while (file.peek() == '#') {
      std::string comment;
      std::getline(file, comment);
      file >> std::ws;
    }
    file >> value;
  }
  file.get();  // The single whitespace before the pixels.
  if (magic != "P5" || values[2] != 255 || values[0] < 1 || values[1] < 1) {
    throw std::runtime_error(path.string() + ": not an 8-bit binary PGM");
  }
  Gray image{values[0], values[1], {}};
  image.pixels.resize(static_cast<std::size_t>(image.width) * image.height);
  file.read(reinterpret_cast<char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.pixels.size()));
  if (!file) throw std::runtime_error(path.string() + ": truncated");
  return image;
}

// The left and right image paths of frame `frame`, from index.csv.
std::pair<fs::path, fs::path> frame_paths(const fs::path& clip, std::size_t frame) {
  std::ifstream index(clip / "index.csv");
  if (!index) throw std::runtime_error((clip / "index.csv").string() + ": cannot open");
  std::string line;
  std::getline(index, line);
  std::vector<std::string> names;
  {
    std::stringstream header(line);
    for (std::string name; std::getline(header, name, ',');) names.push_back(name);
  }
  const auto column = [&](const std::string& name) {
    const auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end()) throw std::runtime_error("index.csv has no column " + name);
    return static_cast<std::size_t>(it - names.begin());
  };
  const std::size_t left = column("left");
  const std::size_t right = column("right");
  for (std::size_t row = 0; std::getline(index, line); ++row) {
    if (row != frame) continue;
    std::vector<std::string> cells;
    std::stringstream cells_of(line);
    for (std::string cell; std::getline(cells_of, cell, ',');) cells.push_back(cell);
    if (cells.size() <= std::max(left, right)) throw std::runtime_error("index.csv: short row");
    return {clip / cells[left], clip / cells[right]};
  }
  throw std::runtime_error((clip / "index.csv").string() + ": fewer frames than " +
                           std::to_string(frame + 1));
}

// A cylindrical panorama of what the eyes see, looking along the rig's +x
// (the left eye's axis), its surface `radius` metres away (0: at infinity).
OutputGeometry panorama(const hm::CameraInfo& left, double radius) {
  OutputGeometry g;
  g.projection = Projection::kCylindrical;
  g.width = kOutputWidth;
  g.height = kOutputHeight;
  const double half_width = std::atan(std::max(left.cx, left.width - 1 - left.cx) / left.fx);
  const double half_height = std::atan(std::max(left.cy, left.height - 1 - left.cy) / left.fy);
  g.panorama.horizontal_fov_deg = 2.0 * half_width * 180.0 / std::numbers::pi;
  g.panorama.elevation_min_deg = -half_height * 180.0 / std::numbers::pi;
  g.panorama.elevation_max_deg = half_height * 180.0 / std::numbers::pi;
  g.panorama.radius = radius;
  return g;
}

// The gray levels a stitch of one camera gives (its RGBA output's red), and
// where that camera sees (alpha 255).
std::vector<std::uint8_t> stitch_alone(const Camera& camera, const OutputGeometry& g,
                                       const test::DeviceBuffer& image, int width, int height) {
  CameraRig rig;
  rig.cameras.push_back(camera);
  Stitcher stitcher(rig, g, BlendSettings{});
  const std::vector<DeviceImage> images{DeviceImage{
      image.data(), width, height, static_cast<std::size_t>(width), PixelFormat::kMono8}};
  test::DeviceBuffer output(static_cast<std::size_t>(g.width) * g.height * 4U);
  test::cuda_check(
      stitcher.stitch(images, output.data(), static_cast<std::size_t>(g.width) * 4U, nullptr),
      "stitch");
  test::cuda_check(cudaDeviceSynchronize(), "stitch");
  return output.download();
}

// How the two eyes agree where both see the panorama: the normalized cross
// correlation of their gray levels, and the PSNR (8-bit peak) of one against
// the other scaled by the ratio of their means there. Also how many pixels
// they share.
struct Agreement {
  double correlation{};
  double psnr_db{};
  double overlap{};  // Of the output, the share both eyes see.
};

Agreement agreement(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  double sum_a = 0.0;
  double sum_b = 0.0;
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); i += 4) {
    if (a[i + 3] != 255 || b[i + 3] != 255) continue;
    sum_a += a[i];
    sum_b += b[i];
    ++n;
  }
  if (n == 0 || sum_b <= 0.0) return {};
  const double scale = sum_a / sum_b;
  const double mean_a = sum_a / static_cast<double>(n);
  const double mean_b = sum_b / static_cast<double>(n);
  double squares = 0.0;
  double ab = 0.0;
  double aa = 0.0;
  double bb = 0.0;
  for (std::size_t i = 0; i < a.size(); i += 4) {
    if (a[i + 3] != 255 || b[i + 3] != 255) continue;
    const double d = a[i] - scale * b[i];
    squares += d * d;
    ab += (a[i] - mean_a) * (b[i] - mean_b);
    aa += (a[i] - mean_a) * (a[i] - mean_a);
    bb += (b[i] - mean_b) * (b[i] - mean_b);
  }
  const double mse = std::max(squares / static_cast<double>(n), 1e-12);
  return Agreement{ab / std::sqrt(std::max(aa * bb, 1e-12)), 10.0 * std::log10(255.0 * 255.0 / mse),
                   static_cast<double>(n) / static_cast<double>(a.size() / 4U)};
}

double mean_and_spread(const Gray& image, double& spread) {
  double sum = 0.0;
  double squares = 0.0;
  for (const std::uint8_t p : image.pixels) {
    sum += p;
    squares += static_cast<double>(p) * p;
  }
  const double n = static_cast<double>(image.pixels.size());
  const double mean = sum / n;
  spread = std::sqrt(std::max(0.0, squares / n - mean * mean));
  return mean;
}

// Checks one clip; returns whether its images have enough contrast for the
// agreement check (with the projector off, the eyes see almost nothing).
bool check_clip(const fs::path& clip) {
  const std::string name = clip.filename().string();
  // Every camera_info file of the clip is one the stitcher takes.
  for (const char* file : {"left.yaml", "right.yaml", "rgb.yaml"}) {
    if (!fs::exists(clip / file)) continue;
    const hm::CameraInfo info = hm::read_camera_info_yaml(clip / file);
    check_camera_info(info, name + "/" + file);
  }
  // The eyes, and their extrinsics, from the pair's files.
  const hm::StereoCameraInfo pair =
      hm::read_stereo_camera_info_yaml(clip / "left.yaml", clip / "right.yaml");
  const RigidTransform left_from_rig = mount({0.0, 0.0, 0.0}, 0.0, 0.0);
  const Camera left{.info = pair.left, .camera_from_rig = left_from_rig};
  const Camera right{.info = pair.right,
                     .camera_from_rig = compose(pair.right_from_left, left_from_rig)};
  // The baseline turned around: the right eye on the other side.
  RigidTransform wrong_extrinsics = pair.right_from_left;
  for (double& t : wrong_extrinsics.translation) t = -t;
  const Camera wrong{.info = pair.right,
                     .camera_from_rig = compose(wrong_extrinsics, left_from_rig)};

  const auto [left_path, right_path] = frame_paths(clip, kFrame);
  const Gray left_image = read_pgm(left_path);
  const Gray right_image = read_pgm(right_path);
  VERIFY_MSG(left_image.width == static_cast<int>(pair.left.width) &&
                 left_image.height == static_cast<int>(pair.left.height) &&
                 right_image.width == static_cast<int>(pair.right.width) &&
                 right_image.height == static_cast<int>(pair.right.height),
             name + ": the images are not the size the calibration gives");
  double spread = 0.0;
  const double mean = mean_and_spread(left_image, spread);
  const bool contrast = spread >= 4.0;

  test::DeviceBuffer left_device(left_image.pixels.size());
  test::DeviceBuffer right_device(right_image.pixels.size());
  test::cuda_check(cudaMemcpy(left_device.data(), left_image.pixels.data(),
                              left_image.pixels.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy");
  test::cuda_check(cudaMemcpy(right_device.data(), right_image.pixels.data(),
                              right_image.pixels.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy");

  // The stitch of both eyes, feathered and exposure compensated, against the
  // CPU reference.
  const OutputGeometry g = panorama(pair.left, 0.0);
  CameraRig rig;
  rig.cameras = {left, right};
  BlendSettings settings;
  settings.exposure_compensation = true;
  Stitcher stitcher(rig, g, settings);
  const std::vector<std::uint64_t>& covered = stitcher.covered_pixels();
  const auto total = static_cast<double>(g.width) * g.height;
  for (std::size_t c = 0; c < covered.size(); ++c) {
    VERIFY_MSG(
        static_cast<double>(covered[c]) / total > 0.5,
        name + ": camera " + std::to_string(c) + " covers " + std::to_string(covered[c] / total));
  }
  const std::vector<DeviceImage> images{
      DeviceImage{left_device.data(), left_image.width, left_image.height,
                  static_cast<std::size_t>(left_image.width), PixelFormat::kMono8},
      DeviceImage{right_device.data(), right_image.width, right_image.height,
                  static_cast<std::size_t>(right_image.width), PixelFormat::kMono8}};
  test::DeviceBuffer output(static_cast<std::size_t>(g.width) * g.height * 4U);
  test::cuda_check(
      stitcher.stitch(images, output.data(), static_cast<std::size_t>(g.width) * 4U, nullptr),
      "stitch");
  test::cuda_check(cudaDeviceSynchronize(), "stitch");
  const std::vector<std::uint8_t> got = output.download();
  const std::vector<float> gains = stitcher.gains(nullptr);
  const std::vector<HostImage> host{
      HostImage{left_image.pixels.data(), left_image.width, left_image.height,
                static_cast<std::size_t>(left_image.width), PixelFormat::kMono8},
      HostImage{right_image.pixels.data(), right_image.width, right_image.height,
                static_cast<std::size_t>(right_image.width), PixelFormat::kMono8}};
  const RemapTables tables = stitcher.download_tables();
  const std::vector<float> want_gains =
      reference_gains(reference_overlap_stats(tables, host), settings, {});
  VERIFY_MSG(gains == want_gains, name + ": the gains differ from the reference's");
  for (const float gain : gains) {
    VERIFY_MSG(std::isfinite(gain) && gain > 0.8F && gain < 1.25F,
               name + ": gain " + std::to_string(gain));
  }
  std::vector<std::uint8_t> want(got.size());
  reference_stitch(tables, host, settings, want_gains, want.data(),
                   static_cast<std::size_t>(g.width) * 4U);
  VERIFY_MSG(got == want, name + ": the stitch differs from the reference's");

  // Where the eyes overlap: their agreement as the panorama's surface moves
  // from 0.75 m to infinity (0), with the files' extrinsics and with the
  // baseline turned around.
  const auto agree_at = [&](const Camera& other, double radius) {
    const OutputGeometry at = panorama(pair.left, radius);
    return agreement(stitch_alone(left, at, left_device, left_image.width, left_image.height),
                     stitch_alone(other, at, right_device, right_image.width, right_image.height));
  };
  Agreement best;
  double best_radius = 0.0;
  double turned_best = -1.0;
  for (const double radius : {0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0}) {
    const Agreement a = agree_at(right, radius);
    if (a.correlation > best.correlation) {
      best = a;
      best_radius = radius;
    }
    turned_best = std::max(turned_best, agree_at(wrong, radius).correlation);
  }
  const Agreement at_infinity = agree_at(right, 0.0);
  std::cout << "  " << name << ": " << left_image.width << 'x' << left_image.height << std::fixed
            << std::setprecision(3) << ", mean " << mean << " spread " << spread << ", gains "
            << gains[0] << ' ' << gains[1] << ", overlap " << best.overlap << ", correlation "
            << best.correlation << " (PSNR " << best.psnr_db << " dB) at " << best_radius << " m, "
            << at_infinity.correlation << " at infinity, " << turned_best
            << " with the baseline turned" << (contrast ? "" : "; too dark to compare") << '\n';
  VERIFY_MSG(best.overlap > 0.5, name + ": the eyes overlap on " + std::to_string(best.overlap));
  if (contrast) {
    VERIFY_MSG(best_radius >= 1.0 && best_radius <= 3.0,
               name + ": the eyes agree best at " + std::to_string(best_radius) + " m");
    VERIFY_MSG(best.correlation >= 0.18,
               name + ": correlation " + std::to_string(best.correlation));
    VERIFY_MSG(best.correlation >= 1.8 * at_infinity.correlation,
               name + ": no better than at infinity");
    VERIFY_MSG(best.correlation >= 1.6 * turned_best,
               name + ": no better than with the baseline turned");
  }
  return contrast;
}

}  // namespace
}  // namespace holoscan::stitching

int main() {
  namespace st = holoscan::stitching;
  namespace fs = std::filesystem;
  const char* data = std::getenv("HOLOSCAN_TEST_DATA");
  if (data == nullptr || *data == '\0') {
    std::cout << "HOLOSCAN_TEST_DATA is not set; skipped\n";
    return st::kSkip;
  }
  if (!st::test::have_gpu()) {
    std::cout << "no CUDA device; skipped\n";
    return st::kSkip;
  }
  std::vector<fs::path> clips;
  const fs::path root = fs::path(data) / "stereo-camera";
  std::error_code error;
  for (const auto& entry : fs::directory_iterator(root, error)) {
    if (fs::exists(entry.path() / "left.yaml") && fs::exists(entry.path() / "right.yaml") &&
        fs::exists(entry.path() / "index.csv")) {
      clips.push_back(entry.path());
    }
  }
  std::sort(clips.begin(), clips.end());
  if (clips.empty()) {
    std::cerr << root << " holds no recording with left.yaml, right.yaml and index.csv\n";
    return 1;
  }
  int compared = 0;
  const int status =
      st::test::run({{"recordings", [&] {
                        for (const fs::path& clip : clips) {
                          try {
                            compared += st::check_clip(clip) ? 1 : 0;
                          } catch (const std::exception& e) {
                            VERIFY_MSG(false, clip.filename().string() + ": " + e.what());
                          }
                        }
                        // With the projector on, a clip shows enough.
                        VERIFY_MSG(compared > 0, "no clip had contrast enough");
                      }}});
  return status;
}
