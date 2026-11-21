// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// EuclideanClusterOp on the local recordings: for each clip in
// $HOLOSCAN_TEST_DATA/stereo-camera/<clip>/ with a depth.yaml, its first depth
// images (16-bit PGM) are deprojected on the host through that camera_info
// file and clustered by the operator, in both connectivities. Real depth has
// noise, holes and flying pixels that the synthetic scene only imitates; the
// labels must still equal the CPU reference's exactly, and the summaries
// agree with them. Skipped (exit 77) when HOLOSCAN_TEST_DATA is not set, as
// in CI, whose tests make their own data.
//
//   clustering_real_data_test --depth-scale <meters per depth unit>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/clustering/clustering.hpp>
#include <holoscan/clustering/euclidean_cluster_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "reference.hpp"
#include "tensor_support.hpp"
#include "test_support.hpp"

namespace holoscan::clustering {
namespace {

namespace fs = std::filesystem;
namespace hm = holoscan::messages;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
// The depth images of each clip that are clustered.
constexpr std::size_t kFrames = 2;
constexpr float kTolerance = 0.025F;
constexpr std::uint32_t kMinClusterSize = 100;

struct DepthImage {
  int width{};
  int height{};
  std::vector<std::uint16_t> counts;
};

// A binary 16-bit PGM (P5, maxval above 255, big-endian samples).
DepthImage read_pgm16(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  int maxval = 0;
  DepthImage image;
  file >> magic >> image.width >> image.height >> maxval;
  file.get();  // The one whitespace after the header.
  if (!file || magic != "P5" || maxval <= 255 || image.width <= 0 || image.height <= 0) {
    throw std::runtime_error(path.string() + ": not a binary 16-bit PGM");
  }
  std::vector<unsigned char> bytes(static_cast<std::size_t>(image.width) * image.height * 2U);
  file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!file) {
    throw std::runtime_error(path.string() + ": truncated");
  }
  image.counts.resize(bytes.size() / 2U);
  for (std::size_t i = 0; i < image.counts.size(); ++i) {
    image.counts[i] = static_cast<std::uint16_t>((bytes[2 * i] << 8U) | bytes[2 * i + 1]);
  }
  return image;
}

// The organized cloud of a depth image through `camera` (an undistorted
// pinhole, as a depth camera's file describes it): NaN where there is no depth.
std::vector<float> deproject(const DepthImage& depth, const hm::CameraInfo& camera, float scale) {
  std::vector<float> xyz(depth.counts.size() * 3U);
  const auto fx = static_cast<float>(camera.fx);
  const auto fy = static_cast<float>(camera.fy);
  const auto cx = static_cast<float>(camera.cx);
  const auto cy = static_cast<float>(camera.cy);
  for (int v = 0; v < depth.height; ++v) {
    for (int u = 0; u < depth.width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * depth.width + u;
      const float z = static_cast<float>(depth.counts[i]) * scale;
      float* p = xyz.data() + 3 * i;
      if (depth.counts[i] == 0U) {
        p[0] = p[1] = p[2] = std::numeric_limits<float>::quiet_NaN();
        continue;
      }
      p[0] = (static_cast<float>(u) - cx) * z / fx;
      p[1] = (static_cast<float>(v) - cy) * z / fy;
      p[2] = z;
    }
  }
  return xyz;
}

hm::schema::PointCloudT cloud_message(const std::vector<float>& xyz, int width, int height,
                                      const std::string& frame_id) {
  const auto absent = [](DLDataType dtype) {
    return std::make_shared<holoscan::Tensor>(test::make({0}, dtype, nullptr, 0, true));
  };
  hm::schema::PointCloudT message;
  message.header = std::make_shared<holoscan::schema::HeaderT>();
  message.header->frame_id = frame_id;
  message.organized = true;
  message.valid_count = -1;
  message.points = std::make_shared<holoscan::Tensor>(
      test::make({height, width, 3}, kFloat32, xyz.data(), xyz.size() * sizeof(float), true));
  message.colors = absent(kUInt8);
  message.normals = absent(kFloat32);
  message.intensity = absent(kFloat32);
  return message;
}

struct Totals {
  std::uint64_t clouds{};
  std::uint64_t clusters{};
  std::uint64_t clustered_points{};
};

// Clusters one cloud with the operator and checks it against the CPU.
void check_cloud(const std::string& name, const std::vector<float>& xyz, int width, int height,
                 const std::string& frame_id, Connectivity connectivity, Totals& totals) {
  const ClusterParams params{.tolerance = kTolerance,
                             .min_cluster_size = kMinClusterSize,
                             .connectivity = connectivity,
                             .image_window = 2};
  auto h = holoscan::testing::create_operator_test<EuclideanClusterOp>(EuclideanClusterConfig{
      .clustering = params,
      .max_points = static_cast<std::uint32_t>(width) * static_cast<std::uint32_t>(height)});
  h.push_samples("point_cloud",
                 {holoscan::testing::sample_at(cloud_message(xyz, width, height, frame_id),
                                               std::chrono::milliseconds(100))});
  h.input_source_device("point_cloud", holoscan::DeviceId{0});
  for (const char* port : {"labels", "clusters"}) {
    h.output_device(port, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::Tensor>(port, 1);
  }
  const auto result = h.run();
  VERIFY_MSG(result.ok(), name + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto labels = result.snapshots<holoscan::Tensor>("labels");
  const auto clusters = result.snapshots<holoscan::Tensor>("clusters");
  VERIFY_MSG(labels.size() == 1 && clusters.size() == 1, name + ": one output");
  if (labels.size() != 1 || clusters.size() != 1) return;
  std::vector<std::int32_t> got(labels[0].bytes().size() / sizeof(std::int32_t));
  std::memcpy(got.data(), labels[0].bytes().data(), got.size() * sizeof(std::int32_t));
  std::vector<ClusterSummary> summaries(clusters[0].bytes().size() / sizeof(ClusterSummary));
  std::memcpy(summaries.data(), clusters[0].bytes().data(),
              summaries.size() * sizeof(ClusterSummary));

  const std::vector<std::int32_t> want =
      connectivity == Connectivity::kImage
          ? test::reference_labels(test::reference_image_components(xyz, width, kTolerance, 2),
                                   params)
          : test::reference_labels(test::reference_components(xyz, kTolerance), params);
  VERIFY_MSG(got == want, name + ": labels equal the CPU reference's");
  // Each summary counts its cluster's points, largest first.
  std::vector<std::uint32_t> counted(summaries.size(), 0U);
  bool labels_in_range = true;
  for (const std::int32_t label : got) {
    if (label >= static_cast<std::int32_t>(summaries.size())) {
      labels_in_range = false;
    } else if (label >= 0) {
      ++counted[static_cast<std::size_t>(label)];
    }
  }
  VERIFY_MSG(labels_in_range, name + ": every label has a summary");
  for (std::size_t k = 0; k < summaries.size(); ++k) {
    VERIFY_MSG(summaries[k].id == static_cast<float>(k) &&
                   summaries[k].point_count == static_cast<float>(counted[k]) &&
                   counted[k] >= kMinClusterSize &&
                   (k == 0 || summaries[k - 1].point_count >= summaries[k].point_count),
               name + ": summary " + std::to_string(k));
    totals.clustered_points += counted[k];
  }
  totals.clusters += summaries.size();
  ++totals.clouds;
  std::cout << name << ": " << summaries.size() << " clusters\n";
}

int run(float scale) {
  const char* root = std::getenv("HOLOSCAN_TEST_DATA");
  if (root == nullptr || *root == '\0') {
    std::cout << "SKIPPED: HOLOSCAN_TEST_DATA is not set\n";
    return 77;
  }
  std::vector<fs::path> clips;
  const fs::path cameras = fs::path(root) / "stereo-camera";
  if (fs::is_directory(cameras)) {
    for (const auto& entry : fs::directory_iterator(cameras)) {
      if (fs::is_regular_file(entry.path() / "depth.yaml")) {
        clips.push_back(entry.path());
      }
    }
  }
  std::sort(clips.begin(), clips.end());
  if (clips.empty()) {
    std::cerr << "no stereo-camera/<clip>/depth.yaml in " << root << '\n';
    return 1;
  }
  Totals totals;
  for (const fs::path& clip : clips) {
    const hm::CameraInfo camera = hm::read_camera_info_yaml(clip / "depth.yaml");
    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(clip / "depth")) {
      if (entry.path().extension() == ".pgm") {
        images.push_back(entry.path());
      }
    }
    std::sort(images.begin(), images.end());
    images.resize(std::min(images.size(), kFrames));
    for (const fs::path& image : images) {
      const DepthImage depth = read_pgm16(image);
      if (std::cmp_not_equal(depth.width, camera.width) ||
          std::cmp_not_equal(depth.height, camera.height)) {
        VERIFY_MSG(false, image.string() + ": not of depth.yaml's size");
        continue;
      }
      const std::vector<float> xyz = deproject(depth, camera, scale);
      const std::string name = clip.filename().string() + "/" + image.filename().string();
      for (const Connectivity connectivity : {Connectivity::kEuclidean, Connectivity::kImage}) {
        check_cloud(name + (connectivity == Connectivity::kImage ? " image" : " euclidean"), xyz,
                    depth.width, depth.height, camera.header.frame_id, connectivity, totals);
      }
    }
  }
  // Loosely: the recordings show objects, so some cloud must have some.
  VERIFY_MSG(totals.clusters > 0, "some cluster in the recordings");
  std::cout << "clouds=" << totals.clouds << " clusters=" << totals.clusters
            << " clustered_points=" << totals.clustered_points << '\n';
  return test::failures() == 0 ? 0 : 1;
}

}  // namespace
}  // namespace holoscan::clustering

int main(int argc, char** argv) try {
  float scale = 0.0F;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string_view option{argv[i]};
    const std::string_view value{argv[i + 1]};
    if (option == "--depth-scale") {
      const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), scale);
      if (error != std::errc() || end != value.data() + value.size()) {
        scale = 0.0F;
      }
    }
  }
  if (!(scale > 0.0F) || !std::isfinite(scale)) {
    std::cerr << "usage: " << argv[0] << " --depth-scale <meters per depth unit>\n";
    return 2;
  }
  return holoscan::clustering::run(scale);
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
