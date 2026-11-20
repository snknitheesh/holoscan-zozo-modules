// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The kernels on the local recordings: for each clip in
// $HOLOSCAN_TEST_DATA/stereo-camera/<clip>/ with a depth.yaml, its first
// depth images (16-bit PGM, 0.000452 m per count) are deprojected with that
// camera, and NormalEstimator (the window and the k-nearest search) and
// PlaneSegmenter run on the clouds. Real scenes have no ground truth, so the
// checks are the results' own rules: unit normals facing the camera,
// curvature within its range, the same bits on a second run, every plane's
// inliers within its distance and angle and counted as labeled, and, for a
// clip with enough depth, most points with a normal and a plane found.
// Skipped (exit 77) when HOLOSCAN_TEST_DATA is not set, as in CI, whose tests
// make their own data.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/normals/normals.hpp>
#include <holoscan/normals/planes.hpp>

#include "test_support.hpp"

namespace holoscan::normals {
namespace {

namespace fs = std::filesystem;
namespace hm = holoscan::messages;

constexpr float kDepthScale = 0.000452F;
constexpr std::size_t kFrames = 3;
// A clip with fewer valid points than this has too little depth to expect a
// normal for most points, or a plane.
constexpr std::size_t kEnoughDepth = 50000;

struct DepthImage {
  int width{};
  int height{};
  std::vector<std::uint16_t> counts;
};

// A binary 16-bit PGM (big-endian samples).
DepthImage read_pgm16(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::string magic;
  int max_value = 0;
  DepthImage image;
  in >> magic >> image.width >> image.height >> max_value;
  in.get();
  if (!in || magic != "P5" || max_value <= 255 || image.width <= 0 || image.height <= 0) {
    throw std::runtime_error(path.string() + ": not a 16-bit binary PGM");
  }
  const std::size_t n = static_cast<std::size_t>(image.width) * image.height;
  std::vector<unsigned char> bytes(2 * n);
  in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!in) {
    throw std::runtime_error(path.string() + ": truncated");
  }
  image.counts.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    image.counts[i] = static_cast<std::uint16_t>((bytes[2 * i] << 8) | bytes[2 * i + 1]);
  }
  return image;
}

template <typename T>
class Device {
 public:
  explicit Device(std::size_t count) : count_(count) {
    if (cudaMalloc(reinterpret_cast<void**>(&data_), std::max<std::size_t>(count, 1) * sizeof(T)) !=
        cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
  }
  ~Device() { cudaFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  [[nodiscard]] T* get() const { return data_; }
  [[nodiscard]] std::vector<T> host() const {
    std::vector<T> out(count_);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(out.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed");
    }
    return out;
  }

 private:
  T* data_{};
  std::size_t count_{};
};

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void check_frame(const std::string& name, const DepthImage& image, const hm::CameraInfo& camera,
                 std::size_t& valid_total, std::size_t& planes_total) {
  const std::size_t n = image.counts.size();
  // The organized cloud, NaN where there is no depth.
  std::vector<float> xyz(3 * n, std::numeric_limits<float>::quiet_NaN());
  std::size_t valid = 0;
  for (int v = 0; v < image.height; ++v) {
    for (int u = 0; u < image.width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * image.width + u;
      if (image.counts[i] == 0U) {
        continue;
      }
      const double z = image.counts[i] * static_cast<double>(kDepthScale);
      xyz[3 * i] = static_cast<float>((u - camera.cx) * z / camera.fx);
      xyz[3 * i + 1] = static_cast<float>((v - camera.cy) * z / camera.fy);
      xyz[3 * i + 2] = static_cast<float>(z);
      ++valid;
    }
  }
  valid_total += valid;
  Device<float> points(3 * n);
  if (cudaMemcpy(points.get(), xyz.data(), xyz.size() * sizeof(float), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  NormalEstimator estimator;
  for (const NeighborSearch search : {NeighborSearch::kWindow, NeighborSearch::kKnn}) {
    const std::string at = name + (search == NeighborSearch::kWindow ? " window" : " knn");
    NormalParams params;
    params.search = search;
    const bool window = search == NeighborSearch::kWindow;
    const int width = window ? image.width : static_cast<int>(n);
    const int height = window ? image.height : 1;
    Device<float> normals(3 * n);
    Device<float> curvature(n);
    VERIFY(estimator.estimate(points.get(), static_cast<std::uint32_t>(n), width, height, params,
                              normals.get(), curvature.get(), nullptr) == cudaSuccess);
    const std::vector<float> got = normals.host();
    const std::vector<float> got_curvature = curvature.host();
    // Again, for the same bits.
    VERIFY(estimator.estimate(points.get(), static_cast<std::uint32_t>(n), width, height, params,
                              normals.get(), curvature.get(), nullptr) == cudaSuccess);
    VERIFY_MSG(same_bits(got, normals.host()) && same_bits(got_curvature, curvature.host()),
               at + ": the same bits on a second run");
    std::size_t with_normal = 0;
    std::size_t not_unit = 0;
    std::size_t facing_away = 0;
    std::size_t out_of_range = 0;
    std::size_t without_point = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const float* p = xyz.data() + 3 * i;
      const float* q = got.data() + 3 * i;
      if (!std::isfinite(q[0])) {
        continue;
      }
      ++with_normal;
      without_point += std::isfinite(p[0]) ? 0U : 1U;
      const double length =
          std::sqrt(static_cast<double>(q[0]) * q[0] + static_cast<double>(q[1]) * q[1] +
                    static_cast<double>(q[2]) * q[2]);
      not_unit += std::fabs(length - 1.0) > 1e-4 ? 1U : 0U;
      facing_away += q[0] * p[0] + q[1] * p[1] + q[2] * p[2] > 1e-6F ? 1U : 0U;
      const float c = got_curvature[i];
      out_of_range += c >= -1e-6F && c <= 1.0F / 3.0F + 1e-5F ? 0U : 1U;
    }
    VERIFY_MSG(without_point == 0, at + ": a normal only where there is a point");
    VERIFY_MSG(not_unit == 0, at + ": " + std::to_string(not_unit) + " normals not unit");
    VERIFY_MSG(facing_away == 0, at + ": " + std::to_string(facing_away) + " facing away");
    VERIFY_MSG(out_of_range == 0, at + ": " + std::to_string(out_of_range) + " curvatures");
    const double share = valid == 0 ? 0.0 : static_cast<double>(with_normal) / valid;
    if (valid >= kEnoughDepth) {
      VERIFY_MSG(share >= 0.5, at + ": only " + std::to_string(share) + " of points with a normal");
    }
    std::cout << at << ": valid=" << valid << " with_normal=" << with_normal << '\n';
    if (!window) {
      continue;
    }

    // The planes of the window normals.
    const PlaneParams plane_params;
    const std::uint32_t planes = plane_params.max_planes;
    Device<std::uint8_t> labels(n);
    Device<float> coefficients(4 * planes);
    Device<float> centroids(3 * planes);
    Device<std::uint32_t> inliers(planes);
    VERIFY(PlaneSegmenter{}.segment(
               points.get(), normals.get(), static_cast<std::uint32_t>(n), plane_params,
               PlaneOutputs{labels.get(), coefficients.get(), centroids.get(), inliers.get()},
               nullptr) == cudaSuccess);
    const std::vector<std::uint8_t> label = labels.host();
    const std::vector<float> plane = coefficients.host();
    const std::vector<std::uint32_t> count = inliers.host();
    std::uint32_t found = 0;
    while (found < planes && std::isfinite(plane[4 * found])) {
      ++found;
    }
    planes_total += found;
    if (valid >= kEnoughDepth) {
      VERIFY_MSG(found != 0, at + ": a plane is found");
    }
    // The plane reported is the least-squares refit of exactly its inliers,
    // not the hypothesis that took them, so points near the edge of its
    // distance and angle can end up a little past them (on the recordings, at
    // most 2.1 times the distance, 1.1 degrees past the angle, and 1 % of the
    // points past the distance).
    const double min_cos =
        std::cos((plane_params.max_normal_angle_deg + 3.0) * std::numbers::pi / 180.0);
    std::vector<std::uint32_t> labeled(planes + 1U, 0U);
    std::size_t off_plane = 0;
    std::size_t past_distance = 0;
    std::size_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint8_t k = label[i];
      if (k == kLabelNoPlane || k == kLabelInvalid) {
        continue;
      }
      if (k > found) {
        ++off_plane;
        continue;
      }
      ++labeled[k];
      ++total;
      const float* a = plane.data() + 4 * (k - 1U);
      const float* p = xyz.data() + 3 * i;
      const float* q = got.data() + 3 * i;
      const double distance = std::fabs(a[0] * p[0] + a[1] * p[1] + a[2] * p[2] + a[3]);
      const double agreement = std::fabs(a[0] * q[0] + a[1] * q[1] + a[2] * q[2]);
      off_plane +=
          distance > 2.5 * plane_params.distance_threshold || agreement < min_cos ? 1U : 0U;
      past_distance += distance > plane_params.distance_threshold ? 1U : 0U;
    }
    VERIFY_MSG(off_plane == 0, at + ": " + std::to_string(off_plane) + " labels off their plane");
    VERIFY_MSG(static_cast<double>(past_distance) <= 0.03 * static_cast<double>(total),
               at + ": " + std::to_string(past_distance) + " of " + std::to_string(total) +
                   " labels past the plane's distance");
    for (std::uint32_t k = 0; k < planes; ++k) {
      VERIFY_MSG(count[k] == (k < found ? labeled[k + 1U] : 0U),
                 at + ": plane " + std::to_string(k) + "'s inliers are its labels");
    }
    std::cout << at << ": planes=" << found << " labeled=" << total
              << " past_distance=" << past_distance << '\n';
  }
}

void check_clip(const fs::path& clip) {
  const hm::CameraInfo camera = hm::read_camera_info_yaml(clip / "depth.yaml");
  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(clip / "depth")) {
    if (entry.path().extension() == ".pgm") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  files.resize(std::min(files.size(), kFrames));
  VERIFY_MSG(!files.empty(), clip.string() + " has depth images");
  std::size_t valid = 0;
  std::size_t planes = 0;
  for (const fs::path& file : files) {
    const DepthImage image = read_pgm16(file);
    VERIFY_MSG(
        std::cmp_equal(image.width, camera.width) && std::cmp_equal(image.height, camera.height),
        file.string() + " is of the camera's size");
    check_frame(clip.filename().string() + "/" + file.filename().string(), image, camera, valid,
                planes);
  }
  std::cout << clip.filename().string() << ": frames=" << files.size() << " valid=" << valid
            << " planes=" << planes << '\n';
}

}  // namespace
}  // namespace holoscan::normals

int main() {
  namespace hn = holoscan::normals;
  const char* root = std::getenv("HOLOSCAN_TEST_DATA");
  if (root == nullptr || *root == '\0') {
    std::cout << "skipped: HOLOSCAN_TEST_DATA is not set\n";
    return 77;
  }
  std::vector<std::filesystem::path> clips;
  const std::filesystem::path recordings = std::filesystem::path(root) / "stereo-camera";
  if (std::filesystem::is_directory(recordings)) {
    for (const auto& entry : std::filesystem::directory_iterator(recordings)) {
      if (std::filesystem::is_regular_file(entry.path() / "depth.yaml")) {
        clips.push_back(entry.path());
      }
    }
  }
  std::sort(clips.begin(), clips.end());
  if (clips.empty()) {
    std::cout << "skipped: no clip with a depth.yaml under " << recordings.string() << '\n';
    return 77;
  }
  std::vector<hn::test::TestCase> cases;
  for (const auto& clip : clips) {
    cases.push_back({clip.filename().string(), [clip] { hn::check_clip(clip); }});
  }
  return hn::test::run(cases);
}
