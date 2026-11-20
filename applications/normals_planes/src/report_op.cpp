// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "report_op.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <holoscan/normals/planes.hpp>
#include <holoscan/normals/synthetic_scene.hpp>

namespace normals_planes_app {
namespace {

using std::chrono_literals::operator""ns;
namespace hm = holoscan::messages;
namespace hn = holoscan::normals;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "ReportOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         std::string("ReportOp: ") + what + ": " + cudaGetErrorString(status)};
}

double angle_deg(const float* a, const float* b) {
  const double dot = static_cast<double>(a[0]) * b[0] + static_cast<double>(a[1]) * b[1] +
                     static_cast<double>(a[2]) * b[2];
  return std::acos(std::clamp(dot, -1.0, 1.0)) * 180.0 / std::numbers::pi;
}

// The q-quantile of `values`; callers check that there are some.
double percentile(std::vector<double>& values, double q) {
  const auto at = std::min(values.size() - 1, static_cast<std::size_t>(q * values.size()));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(at), values.end());
  return values[at];
}

// Pinned host memory of `bytes`, freed with it.
class Pinned {
 public:
  explicit Pinned(std::size_t bytes) : bytes_(bytes) {
    if (const cudaError_t status = cudaMallocHost(&data_, std::max<std::size_t>(bytes, 1U));
        status != cudaSuccess) {
      throw std::runtime_error(std::string("ReportOp: cudaMallocHost: ") +
                               cudaGetErrorString(status));
    }
  }
  ~Pinned() { cudaFreeHost(data_); }
  Pinned(const Pinned&) = delete;
  Pinned& operator=(const Pinned&) = delete;

  template <typename T>
  [[nodiscard]] T* as() const noexcept {
    return static_cast<T*>(data_);
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* data_{};
  std::size_t bytes_{};
};

}  // namespace

struct ReportOp::HostCopies {
  HostCopies(std::size_t pixels, std::size_t planes)
      : normals(12U * pixels),
        curvature(4U * pixels),
        labels(pixels),
        coefficients(16U * planes),
        centroids(12U * planes),
        inliers(4U * planes) {}
  HostCopies(const HostCopies&) = delete;
  HostCopies& operator=(const HostCopies&) = delete;

  Pinned normals;
  Pinned curvature;
  Pinned labels;
  Pinned coefficients;
  Pinned centroids;
  Pinned inliers;
};

ReportOp::ReportOp(ReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("ReportOp needs a report");
  }
}

ReportOp::~ReportOp() = default;

void ReportOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: the frame being checked, and the next one's outputs,
  // which arrive one by one while it is.
  spec.input(point_cloud, "point_cloud").queue_depth(2U);
  spec.input(curvature, "curvature").queue_depth(2U);
  spec.input(labels, "labels").queue_depth(2U);
  spec.input(coefficients, "coefficients").queue_depth(2U);
  spec.input(centroids, "centroids").queue_depth(2U);
  spec.input(inliers, "inliers").queue_depth(2U);
}

holoscan::Contract ReportOp::contract() const {
  holoscan::Contract result;
  result.trigger(
      holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                              point_cloud, curvature, labels, coefficients, centroids, inliers}},
                          .window = 0ns});
  result.input(point_cloud, holoscan::Exact{});
  for (const auto* port : {&curvature, &labels, &coefficients, &centroids, &inliers}) {
    result.input(*port, holoscan::Exact{});
  }
  return result;
}

void ReportOp::start() {
  copies_ = std::make_unique<HostCopies>(
      static_cast<std::size_t>(config_.width) * static_cast<std::size_t>(config_.height),
      config_.max_planes);
  last_frame_id_ = 0;
  finished_ = false;
}

void ReportOp::stop() { copies_.reset(); }

holoscan::expected<void, holoscan::Error> ReportOp::compute(holoscan::ExecutionContext& context) {
  auto cloud_sample = point_cloud.receive();
  auto curvature_sample = curvature.receive();
  auto label_sample = labels.receive();
  auto coefficient_sample = coefficients.receive();
  auto centroid_sample = centroids.receive();
  auto inlier_sample = inliers.receive();
  if (!cloud_sample) {
    return holoscan::make_unexpected(cloud_sample.error());
  }
  for (const auto* failed :
       {&curvature_sample, &label_sample, &coefficient_sample, &centroid_sample, &inlier_sample}) {
    if (!*failed) {
      return holoscan::make_unexpected(failed->error());
    }
  }
  const int width = config_.width;
  const int height = config_.height;
  const hm::schema::PointCloudT& cloud = cloud_sample->data;
  if (auto checked = hm::check_point_cloud(cloud); !checked) {
    return holoscan::make_unexpected(invalid(checked.error().message));
  }
  if (!hm::is_present(cloud.normals)) {
    return holoscan::make_unexpected(invalid("the cloud carries no normals"));
  }
  const std::span<const std::int64_t> shape = cloud.normals->shape_span();
  if (shape.size() != 3 || shape[0] != height || shape[1] != width || shape[2] != 3) {
    return holoscan::make_unexpected(invalid("expected organized [H, W, 3] normals"));
  }
  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  const std::size_t planes = config_.max_planes;

  // Every copy is queued, then the host waits once.
  HostCopies& host = *copies_;
  const std::pair<const holoscan::Tensor*, const Pinned*> copies[] = {
      {cloud.normals.get(), &host.normals},      {&curvature_sample->data, &host.curvature},
      {&label_sample->data, &host.labels},       {&coefficient_sample->data, &host.coefficients},
      {&centroid_sample->data, &host.centroids}, {&inlier_sample->data, &host.inliers}};
  // The copies go on the activation's stream, which the runtime owns: a stream
  // given to copy_to_host_async() becomes the inputs' retirement boundary, and
  // the runtime still records on it when it closes its pools after the
  // operators are gone, so a stream of the operator's own would not live long
  // enough. Without one, each copy waits for itself.
  const cudaStream_t stream = context.cuda_stream().get();
  for (const auto& [tensor, to] : copies) {
    auto copied = stream != nullptr
                      ? tensor->copy_to_host_async(to->as<void>(), to->bytes(), stream)
                      : tensor->copy_to_host(to->as<void>(), to->bytes());
    if (!copied) {
      return copied;
    }
  }
  if (stream != nullptr) {
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
    }
  }
  const float* normals = host.normals.as<float>();
  const float* curvature_values = host.curvature.as<float>();
  const std::uint8_t* label_values = host.labels.as<std::uint8_t>();
  const float* got = host.coefficients.as<float>();
  const float* centroid_values = host.centroids.as<float>();
  const std::uint32_t* inlier_values = host.inliers.as<std::uint32_t>();

  Report& report = *config_.report;
  const std::uint64_t frame_id = cloud_sample->metadata.frame_id;
  if (last_frame_id_ == 0) {
    // Frames before the first cloud came before the first camera info.
    report.before_camera_info = static_cast<std::uint32_t>(frame_id - 1U);
  } else {
    report.frame_gaps += frame_id != last_frame_id_ + 1U ? 1U : 0U;
  }
  last_frame_id_ = frame_id;
  const hn::SyntheticScene scene =
      hn::synthetic_scene(width, height, static_cast<std::uint32_t>(frame_id - 1U));
  std::vector<hn::ScenePixel> truth(pixels);
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      truth[static_cast<std::size_t>(v) * width + u] = hn::synthetic_scene_pixel(scene, u, v);
    }
  }

  // The normals: how many of the room's points got one, their angle error on
  // the inside of each surface, and the curvature there.
  const int r = config_.window_radius;
  std::size_t shown = 0;
  std::size_t with_normal = 0;
  std::vector<double> angles;
  std::vector<double> flat;
  std::vector<double> sphere;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * width + u;
      const std::uint8_t surface = truth[i].surface;
      if (surface == hn::kSurfaceNone) {
        continue;
      }
      ++shown;
      if (!std::isfinite(normals[3 * i])) {
        continue;
      }
      ++with_normal;
      bool alone = true;
      for (int dv = -r; dv <= r && alone; ++dv) {
        for (int du = -r; du <= r && alone; ++du) {
          const int row = v + dv;
          const int column = u + du;
          if (row >= 0 && row < height && column >= 0 && column < width) {
            const std::uint8_t other =
                truth[static_cast<std::size_t>(row) * width + column].surface;
            alone = other == hn::kSurfaceNone || other == surface;
          }
        }
      }
      if (!alone) {
        continue;
      }
      const double angle = angle_deg(normals + 3 * i, truth[i].normal);
      report.flipped += angle > 90.0 ? 1U : 0U;
      angles.push_back(angle);
      float plane[4];
      (hn::synthetic_surface_plane(scene, surface, plane) ? flat : sphere)
          .push_back(curvature_values[i]);
    }
  }
  const double share = shown == 0 ? 0.0 : static_cast<double>(with_normal) / shown;
  report.worst_normal_share = std::min(report.worst_normal_share, share);
  // Without normals to measure, a frame fails rather than passing on nothing.
  report.worst_median_deg =
      std::max(report.worst_median_deg, angles.empty() ? 180.0 : percentile(angles, 0.5));
  report.worst_p95_deg =
      std::max(report.worst_p95_deg, angles.empty() ? 180.0 : percentile(angles, 0.95));
  report.flat_sphere +=
      !sphere.empty() && !flat.empty() && percentile(sphere, 0.5) > percentile(flat, 0.5) ? 0U : 1U;

  // The large planes among those found.
  std::uint32_t found = 0;
  while (found < planes && std::isfinite(got[4 * found])) {
    ++found;
  }
  report.planes += found;
  // Each plane's inliers are the points with its label.
  std::vector<std::size_t> labeled(planes + 1U, 0U);
  for (std::size_t i = 0; i < pixels; ++i) {
    if (label_values[i] != hn::kLabelNoPlane && label_values[i] <= planes) {
      ++labeled[label_values[i]];
    }
  }
  for (std::uint32_t k = 0; k < planes; ++k) {
    const std::size_t want = k < found ? labeled[k + 1U] : 0U;
    report.wrong_inliers += inlier_values[k] == want ? 0U : 1U;
  }
  for (const std::uint8_t surface :
       {hn::kSurfaceFloor, hn::kSurfaceBackWall, hn::kSurfaceSideWall, hn::kSurfaceTableTop}) {
    float want[4];
    hn::synthetic_surface_plane(scene, surface, want);
    int best = -1;
    double best_angle = 180.0;
    double best_offset = 1e30;
    for (std::uint32_t k = 0; k < found; ++k) {
      const double angle = angle_deg(got + 4 * k, want);
      const double offset = std::fabs(got[4 * k + 3] - want[3]);
      if (angle < 5.0 && offset < best_offset) {
        best = static_cast<int>(k);
        best_angle = angle;
        best_offset = offset;
      }
    }
    // The centroid of the plane's inliers lies on the true plane.
    double centroid_offset = 1e9;
    if (best >= 0) {
      const float* c = centroid_values + 3 * best;
      centroid_offset =
          std::fabs(static_cast<double>(want[0]) * c[0] + static_cast<double>(want[1]) * c[1] +
                    static_cast<double>(want[2]) * c[2] + want[3]);
    }
    std::size_t on_surface = 0;
    std::size_t both = 0;
    const auto label = static_cast<std::uint8_t>(best + 1);
    for (std::size_t i = 0; i < pixels && best >= 0; ++i) {
      on_surface += truth[i].surface == surface ? 1U : 0U;
      both += truth[i].surface == surface && label_values[i] == label ? 1U : 0U;
    }
    const std::size_t with_label = best >= 0 ? labeled[label] : 0U;
    const double recall = on_surface == 0 ? 0.0 : static_cast<double>(both) / on_surface;
    const double precision = with_label == 0 ? 0.0 : static_cast<double>(both) / with_label;
    if (best < 0 || best_angle > kMaxPlaneAngleDeg || best_offset > kMaxPlaneOffsetM ||
        centroid_offset > kMaxPlaneOffsetM || recall < kMinRecall || precision < kMinPrecision) {
      ++report.wrong_planes;
    }
    report.worst_plane_angle_deg = std::max(report.worst_plane_angle_deg, best_angle);
    report.worst_plane_offset_m =
        std::max(report.worst_plane_offset_m, best < 0 ? 1e9 : best_offset);
    report.worst_centroid_offset_m = std::max(report.worst_centroid_offset_m, centroid_offset);
    report.worst_recall = std::min(report.worst_recall, recall);
    report.worst_precision = std::min(report.worst_precision, precision);
  }

  if (report.frames++ == 0) {
    HOLOSCAN_LOG_INFO("ReportOp: the first frame is checked");
  }
  if (config_.finished) {
    config_.finished->store(static_cast<std::uint32_t>(frame_id), std::memory_order_release);
  }
  if (!finished_ && frame_id >= config_.expected_frames) {
    finished_ = true;
    context.request_completion();
  }
  return {};
}

}  // namespace normals_planes_app
