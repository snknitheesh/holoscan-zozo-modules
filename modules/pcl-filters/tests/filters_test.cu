// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The filters on the GPU, without a graph, on synthetic clouds made at test
// time: the scene of synthetic_scene.hpp (ground, boxes, a sphere, depth noise,
// flying pixels and missing depth), random clouds, and clouds made to reach
// every edge. The crop box, the masks, the voxel grid and outlier removal are
// checked exactly against CPU references; the ground plane against the scene's
// truth; every filter for determinism across runs and fresh instances; and all
// of them on empty, all-invalid and huge clouds.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/pcl_filters/filters.hpp>
#include <holoscan/pcl_filters/synthetic_scene.hpp>

#include "reference.hpp"
#include "test_support.hpp"

namespace holoscan::pcl_filters {
namespace {

using test::Random;

// Device memory holding a copy of host data, freed on scope exit.
template <typename T>
class Device {
 public:
  explicit Device(std::size_t count) : count_(count) {
    if (cudaMalloc(&data_, std::max<std::size_t>(count, 1U) * sizeof(T)) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
  }
  explicit Device(const std::vector<T>& host) : Device(host.size()) {
    if (!host.empty() && cudaMemcpy(data_, host.data(), host.size() * sizeof(T),
                                    cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed");
    }
  }
  ~Device() { cudaFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  [[nodiscard]] T* get() const { return data_; }
  [[nodiscard]] std::vector<T> host(std::size_t count) const {
    std::vector<T> result(count);
    if (count != 0 && cudaMemcpy(result.data(), data_, count * sizeof(T), cudaMemcpyDeviceToHost) !=
                          cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed");
    }
    return result;
  }
  [[nodiscard]] std::vector<T> host() const { return host(count_); }

 private:
  T* data_{};
  std::size_t count_{};
};

std::uint32_t read_count(const std::uint32_t* device_count) {
  std::uint32_t count = 0;
  if (cudaMemcpy(&count, device_count, sizeof(count), cudaMemcpyDeviceToHost) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return count;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct SceneCloud {
  SyntheticScene scene;
  std::vector<float> xyz;
  std::vector<std::uint8_t> rgb;
  std::vector<SceneLabel> labels;
};

SceneCloud scene_cloud(int width, int height, std::uint32_t frame, float pitch_deg = 25.0F) {
  SceneCloud cloud;
  cloud.scene = synthetic_scene(width, height, frame, pitch_deg);
  synthetic_scene_cloud(cloud.scene, cloud.xyz, cloud.rgb, cloud.labels);
  return cloud;
}

// A cloud of `count` random points in a 4 m cube, every `invalid_every`-th
// one with a NaN or infinite coordinate.
std::vector<float> random_cloud(std::uint32_t count, std::uint64_t seed,
                                std::uint32_t invalid_every = 17) {
  Random random(seed);
  std::vector<float> xyz(3 * static_cast<std::size_t>(count));
  for (std::uint32_t i = 0; i < count; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      xyz[3 * i + axis] = random.uniform(-2.0F, 2.0F);
    }
    if (invalid_every != 0 && i % invalid_every == 3) {
      xyz[3 * i + i % 3] = i % 2 == 0 ? std::numeric_limits<float>::quiet_NaN()
                                      : std::numeric_limits<float>::infinity();
    }
  }
  return xyz;
}

std::vector<std::uint8_t> random_colors(std::size_t count, std::uint64_t seed) {
  Random random(seed);
  std::vector<std::uint8_t> rgb(3 * count);
  for (auto& value : rgb) {
    value = static_cast<std::uint8_t>(random.next() & 0xFFU);
  }
  return rgb;
}

// The scene's synthetic GPU render matches its host version bit for bit.
void check_scene_render() {
  const SceneCloud host = scene_cloud(160, 120, 3);
  Device<float> depth(160 * 120);
  Device<std::uint8_t> rgb(160 * 120 * 3);
  Device<float> xyz(160 * 120 * 3);
  VERIFY(render_synthetic_scene(host.scene, depth.get(), rgb.get(), xyz.get(), nullptr) ==
         cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<float> got = xyz.host();
  std::size_t wrong = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const bool both_nan = std::isnan(got[i]) && std::isnan(host.xyz[i]);
    wrong += !both_nan && std::memcmp(&got[i], &host.xyz[i], sizeof(float)) != 0 ? 1U : 0U;
  }
  VERIFY_MSG(wrong == 0, std::to_string(wrong) + " rendered coordinates differ from the host");
  VERIFY(rgb.host() == host.rgb);
  // Every label occurs, and the ground plane holds for ground points.
  std::size_t counts[4] = {};
  float plane[4];
  synthetic_ground_plane(host.scene, plane);
  double worst = 0.0;
  for (std::size_t i = 0; i < host.labels.size(); ++i) {
    ++counts[static_cast<int>(host.labels[i])];
    if (host.labels[i] == SceneLabel::kGround) {
      const float* p = host.xyz.data() + 3 * i;
      worst = std::max(worst, std::abs(static_cast<double>(plane[0]) * p[0] + plane[1] * p[1] +
                                       plane[2] * p[2] + plane[3]));
    }
  }
  VERIFY(counts[0] > 0 && counts[1] > 0 && counts[2] > 0 && counts[3] > 0);
  // Noise of 4 mm along the ray, at most 2 * sqrt(3) sigma.
  VERIFY_MSG(worst < 0.02, "ground points lie on the plane, within " + std::to_string(worst));
}

void check_crop() {
  const SceneCloud scene = scene_cloud(160, 120, 0);
  std::vector<float> xyz = scene.xyz;
  const std::vector<float> random = random_cloud(5000, 7);
  xyz.insert(xyz.end(), random.begin(), random.end());
  const auto n = static_cast<std::uint32_t>(xyz.size() / 3);
  Device<float> device(xyz);
  Device<std::uint8_t> keep(n);

  std::vector<CropBoxParams> cases;
  CropBoxParams box;
  box.min[0] = -1.0F;
  box.max[0] = 1.0F;
  box.min[1] = -0.5F;
  box.max[1] = 1.5F;
  box.min[2] = 0.5F;
  box.max[2] = 3.0F;
  cases.push_back(box);
  box.keep_outside = true;
  cases.push_back(box);
  CropBoxParams pass;  // A pass-through on z.
  pass.min[2] = 1.0F;
  pass.max[2] = 2.5F;
  cases.push_back(pass);
  CropBoxParams range;  // A distance range.
  range.min_distance = 1.5F;
  range.max_distance = 3.0F;
  cases.push_back(range);
  CropBoxParams turned = box;  // In a body frame: x forward, y left, z up, 1.2 m up.
  turned.keep_outside = false;
  turned.use_transform = true;
  turned.transform = Transform{{0.0F, 0.0F, 1.0F, 0.1F,   //
                                -1.0F, 0.0F, 0.0F, 0.0F,  //
                                0.0F, -1.0F, 0.0F, 1.2F}};
  turned.min[0] = 1.0F;
  turned.max[0] = 4.0F;
  turned.min[1] = -1.0F;
  turned.max[1] = 1.0F;
  turned.min[2] = 0.05F;
  turned.max[2] = 2.0F;
  cases.push_back(turned);
  // A transform that is not rigid, as an affine one may be.
  CropBoxParams sheared = turned;
  sheared.transform.matrix[1] = 0.3F;
  sheared.min_distance = 1.0F;
  sheared.max_distance = 4.0F;
  cases.push_back(sheared);

  for (std::size_t c = 0; c < cases.size(); ++c) {
    VERIFY(launch_crop_box(device.get(), n, cases[c], keep.get(), nullptr) == cudaSuccess);
    VERIFY(cudaDeviceSynchronize() == cudaSuccess);
    const std::vector<std::uint8_t> got = keep.host();
    const std::vector<std::uint8_t> want = test::crop_reference(xyz, cases[c]);
    const auto kept = std::count(want.begin(), want.end(), std::uint8_t{1});
    VERIFY_MSG(got == want, "crop case " + std::to_string(c) + " differs from the reference");
    VERIFY_MSG(kept > 100 && kept < static_cast<std::ptrdiff_t>(n) - 100,
               "crop case " + std::to_string(c) +
                   " keeps some points but not all: " + std::to_string(kept));
    for (std::uint32_t i = 0; i < n; ++i) {
      if (!test::valid(xyz.data() + 3 * i) && got[i] != 0U) {
        VERIFY_MSG(false, "an invalid point was kept");
        break;
      }
    }
  }
}

// apply_mask and Compactor against a host mask.
void check_masks() {
  const std::uint32_t n = 10007;
  const std::vector<float> xyz = random_cloud(n, 3);
  const std::vector<std::uint8_t> rgb = random_colors(n, 4);
  std::vector<std::uint8_t> mask(n);
  Random random(5);
  for (auto& value : mask) {
    value = random.next() % 3 == 0 ? 0U : 1U;
  }
  Device<float> device_xyz(xyz);
  Device<std::uint8_t> device_rgb(rgb);
  Device<std::uint8_t> device_mask(mask);
  Device<float> out_xyz(3 * n);
  Device<std::uint8_t> out_rgb(3 * n);

  VERIFY(apply_mask(device_xyz.get(), device_rgb.get(), device_mask.get(), n, out_xyz.get(),
                    out_rgb.get(), nullptr) == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<float> organized = out_xyz.host();
  const std::vector<std::uint8_t> organized_rgb = out_rgb.host();
  std::size_t wrong = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      if (mask[i] != 0U) {
        wrong += std::memcmp(&organized[3 * i + k], &xyz[3 * i + k], 4) != 0 ||
                 organized_rgb[3 * i + k] != rgb[3 * i + k];
      } else {
        wrong += !std::isnan(organized[3 * i + k]) || organized_rgb[3 * i + k] != 0U;
      }
    }
  }
  VERIFY_MSG(wrong == 0, std::to_string(wrong) + " organized values differ");

  Compactor compactor;
  VERIFY(compactor.count(device_mask.get(), n, nullptr) == cudaSuccess);
  const std::uint32_t kept = read_count(compactor.device_count());
  VERIFY(kept == static_cast<std::uint32_t>(std::count(mask.begin(), mask.end(), 1U)));
  VERIFY(compactor.write(device_xyz.get(), device_rgb.get(), out_xyz.get(), out_rgb.get(),
                         nullptr) == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  std::vector<float> want_xyz;
  std::vector<std::uint8_t> want_rgb;
  for (std::uint32_t i = 0; i < n; ++i) {
    if (mask[i] != 0U) {
      want_xyz.insert(want_xyz.end(), xyz.begin() + 3 * i, xyz.begin() + 3 * i + 3);
      want_rgb.insert(want_rgb.end(), rgb.begin() + 3 * i, rgb.begin() + 3 * i + 3);
    }
  }
  const std::vector<float> got = out_xyz.host(3 * kept);
  const bool same = got.size() == want_xyz.size() &&
                    std::memcmp(got.data(), want_xyz.data(), got.size() * 4) == 0;
  VERIFY_MSG(same, "packed points differ, or are out of order");
  VERIFY(out_rgb.host(3 * kept) == want_rgb);
}

// Runs the voxel grid; returns points and colors.
test::VoxelResult run_voxels(VoxelGrid& grid, const Device<float>& xyz,
                             const Device<std::uint8_t>* rgb, std::uint32_t n, float size,
                             VoxelMode mode) {
  test::VoxelResult result;
  if (grid.group(xyz.get(), n, size, nullptr) != cudaSuccess) {
    VERIFY_MSG(false, "VoxelGrid::group failed");
    return result;
  }
  const std::uint32_t voxels = read_count(grid.device_count());
  Device<float> out_xyz(3 * static_cast<std::size_t>(voxels));
  Device<std::uint8_t> out_rgb(3 * static_cast<std::size_t>(voxels));
  VERIFY(grid.write(xyz.get(), rgb != nullptr ? rgb->get() : nullptr, mode, out_xyz.get(),
                    rgb != nullptr ? out_rgb.get() : nullptr, nullptr) == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  result.xyz = out_xyz.host(3 * static_cast<std::size_t>(voxels));
  if (rgb != nullptr) {
    result.rgb = out_rgb.host(3 * static_cast<std::size_t>(voxels));
  }
  return result;
}

void check_voxels() {
  const SceneCloud scene = scene_cloud(320, 240, 1);
  std::vector<float> xyz = scene.xyz;
  std::vector<std::uint8_t> rgb = scene.rgb;
  const std::vector<float> random = random_cloud(20000, 11);
  xyz.insert(xyz.end(), random.begin(), random.end());
  const std::vector<std::uint8_t> random_rgb = random_colors(20000, 12);
  rgb.insert(rgb.end(), random_rgb.begin(), random_rgb.end());
  // Points beyond the grid's range are dropped; one right at its edge is not.
  const float edge = 0.01F * static_cast<float>(kVoxelIndexRange);
  for (const float x : {2.0F * edge, -2.0F * edge, edge * 0.99F}) {
    xyz.insert(xyz.end(), {x, 0.5F, 0.5F});
    rgb.insert(rgb.end(), {1U, 2U, 3U});
  }
  const auto n = static_cast<std::uint32_t>(xyz.size() / 3);
  Device<float> device_xyz(xyz);
  Device<std::uint8_t> device_rgb(rgb);
  VoxelGrid grid;
  for (const float size : {0.01F, 0.05F, 0.2F, 100.0F}) {
    for (const VoxelMode mode : {VoxelMode::kCentroid, VoxelMode::kFirstPoint}) {
      const std::string name =
          "voxel " + std::to_string(size) + (mode == VoxelMode::kCentroid ? " centroid" : " first");
      const test::VoxelResult want = test::voxel_reference(xyz, rgb, size, mode);
      const test::VoxelResult got = run_voxels(grid, device_xyz, &device_rgb, n, size, mode);
      VERIFY_MSG(got.xyz.size() == want.xyz.size(),
                 name + ": " + std::to_string(got.xyz.size() / 3) + " voxels, expected " +
                     std::to_string(want.xyz.size() / 3));
      VERIFY_MSG(same_bits(got.xyz, want.xyz), name + ": points differ from the reference");
      VERIFY_MSG(got.rgb == want.rgb, name + ": colors differ from the reference");
      if (size == 100.0F) {
        // The eight voxels around the origin hold every point but the far ones.
        VERIFY_MSG(want.xyz.size() / 3 <= 8 + 3, name + ": a few voxels");
      }
    }
  }
  // Without colors.
  const test::VoxelResult want = test::voxel_reference(xyz, {}, 0.05F, VoxelMode::kCentroid);
  const test::VoxelResult got =
      run_voxels(grid, device_xyz, nullptr, n, 0.05F, VoxelMode::kCentroid);
  VERIFY(same_bits(got.xyz, want.xyz));
  VERIFY(grid.group(device_xyz.get(), n, 0.0F, nullptr) == cudaErrorInvalidValue);
  VERIFY(grid.group(device_xyz.get(), n, std::nanf(""), nullptr) == cudaErrorInvalidValue);
}

struct GroundResult {
  float plane[4]{};
  std::vector<std::uint8_t> keep;
  std::uint32_t ground{};
};

GroundResult run_ground(GroundSegmenter& segmenter, const Device<float>& xyz, std::uint32_t n,
                        const GroundParams& params) {
  GroundResult result;
  Device<float> plane(4);
  Device<std::uint8_t> keep(n);
  Device<std::uint32_t> ground(1);
  VERIFY(segmenter.segment(xyz.get(), n, params, plane.get(), keep.get(), ground.get(), nullptr) ==
         cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<float> host_plane = plane.host();
  std::copy(host_plane.begin(), host_plane.end(), result.plane);
  result.keep = keep.host();
  result.ground = ground.host()[0];
  return result;
}

GroundParams scene_ground_params(const SyntheticScene& scene) {
  GroundParams params;
  float truth[4];
  synthetic_ground_plane(scene, truth);
  // Up is only a hint: tilted 10 degrees from the truth.
  params.up[0] = std::sin(10.0F * std::numbers::pi_v<float> / 180.0F);
  params.up[1] = truth[1];
  params.up[2] = truth[2];
  params.distance_threshold = 0.02F;
  params.max_iterations = 200;
  return params;
}

double plane_angle_deg(const float* a, const float* b) {
  const double dot = static_cast<double>(a[0]) * b[0] + static_cast<double>(a[1]) * b[1] +
                     static_cast<double>(a[2]) * b[2];
  return std::acos(std::clamp(dot, -1.0, 1.0)) * 180.0 / std::numbers::pi;
}

void check_ground_plane() {
  GroundSegmenter segmenter;
  for (const float pitch : {15.0F, 25.0F, 40.0F}) {
    for (const std::uint32_t frame : {0U, 31U}) {
      const SceneCloud scene = scene_cloud(320, 240, frame, pitch);
      const auto n = static_cast<std::uint32_t>(scene.labels.size());
      Device<float> xyz(scene.xyz);
      float truth[4];
      synthetic_ground_plane(scene.scene, truth);
      const std::string name = "pitch " + std::to_string(pitch) + " frame " + std::to_string(frame);
      for (const bool refine : {false, true}) {
        GroundParams params = scene_ground_params(scene.scene);
        params.refine = refine;
        const GroundResult result = run_ground(segmenter, xyz, n, params);
        const double angle = plane_angle_deg(result.plane, truth);
        const double offset = std::abs(static_cast<double>(result.plane[3]) - truth[3]);
        // A plane from three noisy points is coarse; the refit is not.
        const double max_angle = refine ? 0.2 : 3.0;
        const double max_offset = refine ? 0.003 : 0.05;
        VERIFY_MSG(angle < max_angle && offset < max_offset,
                   name + (refine ? " refined" : " raw") + ": plane is " + std::to_string(angle) +
                       " deg and " + std::to_string(offset) + " m from the truth");
        std::size_t ground = 0, ground_left = 0, objects = 0, objects_left = 0, removed = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
          const bool kept = result.keep[i] != 0U;
          removed += test::valid(scene.xyz.data() + 3 * i) && !kept ? 1U : 0U;
          if (scene.labels[i] == SceneLabel::kGround) {
            ++ground;
            ground_left += kept ? 1U : 0U;
          } else if (scene.labels[i] == SceneLabel::kObject) {
            ++objects;
            objects_left += kept ? 1U : 0U;
          } else if (scene.labels[i] == SceneLabel::kNone) {
            VERIFY_MSG(!kept, "a point without depth was kept");
          }
        }
        VERIFY_MSG(removed == result.ground, name + ": the ground count matches the mask");
        if (refine) {
          std::cout << "  " << name << ": plane " << angle << " deg, " << 1000.0 * offset
                    << " mm from the truth\n";
          const double ground_kept = static_cast<double>(ground_left) / ground;
          const double objects_kept = static_cast<double>(objects_left) / objects;
          VERIFY_MSG(ground_kept < 0.005,
                     name + ": " + std::to_string(100.0 * ground_kept) + "% of the ground is left");
          VERIFY_MSG(objects_kept > 0.93,
                     name + ": " + std::to_string(100.0 * objects_kept) + "% of objects are kept");
        }
      }
    }
  }
}

void check_ground_modes() {
  GroundSegmenter segmenter;
  const SceneCloud scene = scene_cloud(160, 120, 0);
  const auto n = static_cast<std::uint32_t>(scene.labels.size());
  Device<float> xyz(scene.xyz);
  float truth[4];
  synthetic_ground_plane(scene.scene, truth);

  // Height mode: the plane up . p = height, everything at or below it ground.
  GroundParams height;
  height.mode = GroundMode::kHeight;
  height.up[0] = 0.0F;
  height.up[1] = 2.0F * truth[1];  // Normalized by the filter.
  height.up[2] = 2.0F * truth[2];
  height.height = 0.05F - truth[3];  // 5 cm above the ground.
  const GroundResult result = run_ground(segmenter, xyz, n, height);
  VERIFY(std::abs(result.plane[1] - truth[1]) < 1e-6F &&
         std::abs(result.plane[2] - truth[2]) < 1e-6F);
  VERIFY(std::abs(result.plane[3] + height.height) < 1e-6F);
  std::size_t wrong = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    const float* p = scene.xyz.data() + 3 * i;
    const bool keep = test::valid(p) && result.plane[0] * p[0] + result.plane[1] * p[1] +
                                                result.plane[2] * p[2] + result.plane[3] >
                                            0.0F;
    wrong += keep != (result.keep[i] != 0U) ? 1U : 0U;
  }
  VERIFY_MSG(wrong == 0, std::to_string(wrong) + " points differ from the height threshold");

  // A wall only (the plane x = 1) is never ground with up along z.
  std::vector<float> wall;
  Random random(9);
  for (int i = 0; i < 5000; ++i) {
    wall.insert(wall.end(), {1.0F, random.uniform(-1.0F, 1.0F), random.uniform(0.0F, 2.0F)});
  }
  Device<float> wall_xyz(wall);
  GroundParams params;
  const GroundResult none = run_ground(segmenter, wall_xyz, 5000, params);
  VERIFY_MSG(std::isnan(none.plane[0]) && std::isnan(none.plane[3]), "a wall gives no plane");
  VERIFY_MSG(std::count(none.keep.begin(), none.keep.end(), 1U) == 5000 && none.ground == 0,
             "without a plane every point stays");

  // Unusable settings.
  Device<float> plane(4);
  Device<std::uint8_t> keep(n);
  GroundParams bad;
  bad.up[0] = bad.up[1] = bad.up[2] = 0.0F;
  VERIFY(segmenter.segment(xyz.get(), n, bad, plane.get(), keep.get(), nullptr, nullptr) ==
         cudaErrorInvalidValue);
  bad = GroundParams{};
  bad.max_iterations = 0;
  VERIFY(segmenter.segment(xyz.get(), n, bad, plane.get(), keep.get(), nullptr, nullptr) ==
         cudaErrorInvalidValue);
  bad.max_iterations = kMaxIterationsLimit + 1U;
  VERIFY(segmenter.segment(xyz.get(), n, bad, plane.get(), keep.get(), nullptr, nullptr) ==
         cudaErrorInvalidValue);
}

void check_statistical() {
  // A small scene, so that the reference may compare every pair of points.
  const SceneCloud scene = scene_cloud(128, 96, 2);
  const auto n = static_cast<std::uint32_t>(scene.labels.size());
  Device<float> xyz(scene.xyz);
  Device<std::uint8_t> keep(n);
  OutlierFilter filter;
  for (const std::uint32_t k : {1U, 8U, 20U}) {
    for (const float radius : {0.15F, 0.5F}) {
      OutlierParams params;
      params.k = k;
      params.search_radius = radius;
      params.std_multiplier = 1.0F;
      const std::string name = "k " + std::to_string(k) + " radius " + std::to_string(radius);
      VERIFY(filter.filter(xyz.get(), n, params, keep.get(), nullptr) == cudaSuccess);
      VERIFY(cudaDeviceSynchronize() == cudaSuccess);
      std::vector<float> mean(n);
      VERIFY(cudaMemcpy(mean.data(), filter.device_mean_distances(), n * sizeof(float),
                        cudaMemcpyDeviceToHost) == cudaSuccess);
      double threshold = 0.0;
      VERIFY(cudaMemcpy(&threshold, filter.device_threshold(), sizeof(double),
                        cudaMemcpyDeviceToHost) == cudaSuccess);
      const std::vector<float> want = test::knn_mean_reference(scene.xyz, k, radius);
      std::size_t wrong = 0;
      for (std::uint32_t i = 0; i < n; ++i) {
        const bool both_nan = std::isnan(mean[i]) && std::isnan(want[i]);
        wrong += !both_nan && std::memcmp(&mean[i], &want[i], 4) != 0 ? 1U : 0U;
      }
      VERIFY_MSG(wrong == 0,
                 name + ": " + std::to_string(wrong) + " mean distances differ from brute force");
      const double want_threshold = test::threshold_reference(want, 1.0);
      VERIFY_MSG(std::abs(threshold - want_threshold) <= 1e-9 * std::abs(want_threshold),
                 name + ": threshold " + std::to_string(threshold) + ", expected " +
                     std::to_string(want_threshold));
      const std::vector<std::uint8_t> got = keep.host();
      wrong = 0;
      for (std::uint32_t i = 0; i < n; ++i) {
        const bool expect = std::isfinite(want[i]) && static_cast<double>(want[i]) <= threshold;
        wrong += expect != (got[i] != 0U) ? 1U : 0U;
      }
      VERIFY_MSG(wrong == 0, name + ": " + std::to_string(wrong) + " decisions differ");
    }
  }
}

void check_radius() {
  const SceneCloud scene = scene_cloud(128, 96, 4);
  const auto n = static_cast<std::uint32_t>(scene.labels.size());
  Device<float> xyz(scene.xyz);
  Device<std::uint8_t> keep(n);
  OutlierFilter filter;
  for (const float radius : {0.03F, 0.1F}) {
    for (const std::uint32_t neighbors : {0U, 1U, 4U, 30U}) {
      OutlierParams params;
      params.mode = OutlierMode::kRadius;
      params.radius = radius;
      params.min_neighbors = neighbors;
      VERIFY(filter.filter(xyz.get(), n, params, keep.get(), nullptr) == cudaSuccess);
      VERIFY(cudaDeviceSynchronize() == cudaSuccess);
      VERIFY_MSG(keep.host() == test::radius_reference(scene.xyz, radius, neighbors),
                 "radius " + std::to_string(radius) + " neighbors " + std::to_string(neighbors) +
                     " differs from brute force");
    }
  }
  OutlierParams bad;
  bad.k = 0;
  VERIFY(filter.filter(xyz.get(), n, bad, keep.get(), nullptr) == cudaErrorInvalidValue);
  bad.k = kMaxNeighbors + 1U;
  VERIFY(filter.filter(xyz.get(), n, bad, keep.get(), nullptr) == cudaErrorInvalidValue);
  bad = OutlierParams{};
  bad.mode = OutlierMode::kRadius;
  bad.radius = 0.0F;
  VERIFY(filter.filter(xyz.get(), n, bad, keep.get(), nullptr) == cudaErrorInvalidValue);
}

// On the full-size scene: the flying pixels go, the surfaces stay.
void check_outlier_rates() {
  const SceneCloud scene = scene_cloud(640, 480, 5);
  const auto n = static_cast<std::uint32_t>(scene.labels.size());
  Device<float> xyz(scene.xyz);
  Device<std::uint8_t> keep(n);
  OutlierFilter filter;
  OutlierParams statistical;
  statistical.k = 8;
  statistical.search_radius = 0.2F;
  statistical.std_multiplier = 2.0F;
  OutlierParams radius;
  radius.mode = OutlierMode::kRadius;
  radius.radius = 0.03F;
  radius.min_neighbors = 3;
  for (const OutlierParams& params : {statistical, radius}) {
    const std::string name = params.mode == OutlierMode::kStatistical ? "statistical" : "radius";
    VERIFY(filter.filter(xyz.get(), n, params, keep.get(), nullptr) == cudaSuccess);
    VERIFY(cudaDeviceSynchronize() == cudaSuccess);
    const std::vector<std::uint8_t> got = keep.host();
    std::size_t outliers = 0, outliers_left = 0, surface = 0, surface_left = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
      if (scene.labels[i] == SceneLabel::kOutlier) {
        ++outliers;
        outliers_left += got[i];
      } else if (scene.labels[i] != SceneLabel::kNone) {
        ++surface;
        surface_left += got[i];
      }
    }
    const double removed = 1.0 - static_cast<double>(outliers_left) / outliers;
    const double kept = static_cast<double>(surface_left) / surface;
    std::cout << "  " << name << ": " << 100.0 * removed << "% of " << outliers
              << " outliers removed, " << 100.0 * kept << "% of " << surface
              << " surface points kept\n";
    VERIFY_MSG(removed > 0.9, name + ": too few outliers removed");
    VERIFY_MSG(kept > 0.9, name + ": too few surface points kept");
  }
}

// Every filter gives the same bits on every run, and from a fresh instance.
void check_determinism() {
  const SceneCloud scene = scene_cloud(640, 480, 6);
  const auto n = static_cast<std::uint32_t>(scene.labels.size());
  Device<float> xyz(scene.xyz);
  Device<std::uint8_t> rgb(scene.rgb);
  std::vector<float> first_plane;
  std::vector<std::uint8_t> first_ground, first_outliers;
  test::VoxelResult first_voxels;
  for (int run = 0; run < 4; ++run) {
    GroundSegmenter segmenter;
    const GroundResult ground = run_ground(segmenter, xyz, n, scene_ground_params(scene.scene));
    const std::vector<float> plane(ground.plane, ground.plane + 4);
    OutlierFilter filter;
    Device<std::uint8_t> keep(n);
    VERIFY(filter.filter(xyz.get(), n, OutlierParams{.search_radius = 0.2F}, keep.get(), nullptr) ==
           cudaSuccess);
    const std::vector<std::uint8_t> outliers = keep.host();
    VoxelGrid grid;
    const test::VoxelResult voxels = run_voxels(grid, xyz, &rgb, n, 0.03F, VoxelMode::kCentroid);
    if (run == 0) {
      first_plane = plane;
      first_ground = ground.keep;
      first_outliers = outliers;
      first_voxels = voxels;
      continue;
    }
    VERIFY_MSG(same_bits(plane, first_plane), "the plane changes between runs");
    VERIFY_MSG(ground.keep == first_ground, "the ground changes between runs");
    VERIFY_MSG(outliers == first_outliers, "the outliers change between runs");
    VERIFY_MSG(same_bits(voxels.xyz, first_voxels.xyz) && voxels.rgb == first_voxels.rgb,
               "the voxels change between runs");
  }
  // Another seed draws other hypotheses, and still finds the ground.
  GroundSegmenter segmenter;
  GroundParams params = scene_ground_params(scene.scene);
  params.seed = 12345;
  const GroundResult other = run_ground(segmenter, xyz, n, params);
  float truth[4];
  synthetic_ground_plane(scene.scene, truth);
  VERIFY(plane_angle_deg(other.plane, truth) < 0.2);
}

// Empty and all-invalid clouds: nothing kept, nothing found, no failure.
void check_degenerate() {
  for (const std::uint32_t n : {0U, 1U, 2U, 1000U}) {
    std::vector<float> xyz(3 * static_cast<std::size_t>(n),
                           std::numeric_limits<float>::quiet_NaN());
    Device<float> device(xyz);
    Device<std::uint8_t> keep(n);
    const std::string name = std::to_string(n) + " invalid points";

    const auto none_kept = [&keep, n] {
      const std::vector<std::uint8_t> mask = keep.host();
      return std::count(mask.begin(), mask.end(), 0U) == static_cast<std::ptrdiff_t>(n);
    };
    VERIFY(launch_crop_box(device.get(), n, CropBoxParams{}, keep.get(), nullptr) == cudaSuccess);
    VERIFY(cudaDeviceSynchronize() == cudaSuccess);
    VERIFY_MSG(none_kept(), name + ": crop");

    VoxelGrid grid;
    VERIFY(grid.group(device.get(), n, 0.1F, nullptr) == cudaSuccess);
    VERIFY_MSG(read_count(grid.device_count()) == 0U, name + ": no voxels");
    VERIFY(grid.write(device.get(), nullptr, VoxelMode::kCentroid, nullptr, nullptr, nullptr) ==
           cudaSuccess);

    GroundSegmenter segmenter;
    const GroundResult ground = run_ground(segmenter, device, n, GroundParams{});
    VERIFY_MSG(std::isnan(ground.plane[0]) && ground.ground == 0U, name + ": no plane");
    VERIFY_MSG(
        std::count(ground.keep.begin(), ground.keep.end(), 0U) == static_cast<std::ptrdiff_t>(n),
        name + ": ground");

    OutlierFilter filter;
    for (const OutlierMode mode : {OutlierMode::kStatistical, OutlierMode::kRadius}) {
      VERIFY(filter.filter(device.get(), n, OutlierParams{.mode = mode}, keep.get(), nullptr) ==
             cudaSuccess);
      VERIFY(cudaDeviceSynchronize() == cudaSuccess);
      VERIFY_MSG(none_kept(), name + ": outliers");
    }

    Compactor compactor;
    VERIFY(compactor.count(keep.get(), n, nullptr) == cudaSuccess);
    VERIFY_MSG(read_count(compactor.device_count()) == 0U, name + ": nothing to pack");
  }
  // A single valid point: one voxel, no plane, no neighbors.
  std::vector<float> one{0.5F, 0.5F, 0.5F};
  Device<float> device(one);
  VoxelGrid grid;
  const test::VoxelResult voxels = run_voxels(grid, device, nullptr, 1, 0.1F, VoxelMode::kCentroid);
  VERIFY(voxels.xyz == one);
  GroundSegmenter segmenter;
  VERIFY(std::isnan(run_ground(segmenter, device, 1, GroundParams{}).plane[0]));
}

// A 4096 x 2048 cloud, 8.4 million points: a scene tiled with random points.
void check_huge() {
  const SceneCloud scene = scene_cloud(4096, 2048, 7);
  const auto n = static_cast<std::uint32_t>(scene.labels.size());
  Device<float> xyz(scene.xyz);
  Device<std::uint8_t> rgb(scene.rgb);
  Device<std::uint8_t> keep(n);

  CropBoxParams box;
  box.max_distance = 4.0F;
  VERIFY(launch_crop_box(xyz.get(), n, box, keep.get(), nullptr) == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  VERIFY_MSG(keep.host() == test::crop_reference(scene.xyz, box), "huge: crop");

  VoxelGrid grid;
  const test::VoxelResult got = run_voxels(grid, xyz, &rgb, n, 0.02F, VoxelMode::kCentroid);
  const test::VoxelResult want =
      test::voxel_reference(scene.xyz, scene.rgb, 0.02F, VoxelMode::kCentroid);
  VERIFY_MSG(same_bits(got.xyz, want.xyz) && got.rgb == want.rgb,
             "huge: voxels differ from the reference (" + std::to_string(got.xyz.size() / 3) +
                 " vs " + std::to_string(want.xyz.size() / 3) + ")");

  GroundSegmenter segmenter;
  const GroundResult ground = run_ground(segmenter, xyz, n, scene_ground_params(scene.scene));
  float truth[4];
  synthetic_ground_plane(scene.scene, truth);
  VERIFY_MSG(
      plane_angle_deg(ground.plane, truth) < 0.1 && std::abs(ground.plane[3] - truth[3]) < 0.002,
      "huge: the plane");

  OutlierFilter filter;
  OutlierParams params;
  params.mode = OutlierMode::kRadius;
  params.radius = 0.01F;
  params.min_neighbors = 3;
  VERIFY(filter.filter(xyz.get(), n, params, keep.get(), nullptr) == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<std::uint8_t> kept = keep.host();
  std::size_t outliers = 0, outliers_left = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    if (scene.labels[i] == SceneLabel::kOutlier) {
      ++outliers;
      outliers_left += kept[i];
    }
  }
  VERIFY_MSG(outliers_left < outliers / 10, "huge: most outliers removed");
  params = OutlierParams{.k = 4, .search_radius = 0.01F};
  VERIFY(filter.filter(xyz.get(), n, params, keep.get(), nullptr) == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);

  Compactor compactor;
  VERIFY(compactor.count(keep.get(), n, nullptr) == cudaSuccess);
  const std::vector<std::uint8_t> mask = keep.host();
  VERIFY(read_count(compactor.device_count()) ==
         static_cast<std::uint32_t>(std::count(mask.begin(), mask.end(), 1U)));
}

}  // namespace
}  // namespace holoscan::pcl_filters

int main() {
  using holoscan::pcl_filters::test::TestCase;
  namespace pf = holoscan::pcl_filters;
  return holoscan::pcl_filters::test::run({
      TestCase{"synthetic scene render", pf::check_scene_render},
      TestCase{"crop box and pass-through against the reference", pf::check_crop},
      TestCase{"organized and packed masks", pf::check_masks},
      TestCase{"voxel grid against the reference", pf::check_voxels},
      TestCase{"ground plane against the truth", pf::check_ground_plane},
      TestCase{"ground height mode, no plane, bad settings", pf::check_ground_modes},
      TestCase{"statistical outliers against brute force", pf::check_statistical},
      TestCase{"radius outliers against brute force", pf::check_radius},
      TestCase{"outlier rates on the scene", pf::check_outlier_rates},
      TestCase{"determinism across runs", pf::check_determinism},
      TestCase{"empty and all-invalid clouds", pf::check_degenerate},
      TestCase{"huge cloud", pf::check_huge},
  });
}
