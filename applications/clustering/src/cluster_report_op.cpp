// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "cluster_report_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/clustering/synthetic_scene.hpp>

namespace clustering_app {
namespace {

using std::chrono_literals::operator""ns;
namespace hc = holoscan::clustering;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "ClusterReportOp: " + message};
}

double pairs(double n) { return n * (n - 1.0) / 2.0; }

// The adjusted Rand index of a contingency table of `rows` x `columns` counts.
double adjusted_rand_index(const std::vector<double>& table, std::size_t columns) {
  const std::size_t rows = table.size() / columns;
  std::vector<double> row_sums(rows, 0.0);
  std::vector<double> column_sums(columns, 0.0);
  double index = 0.0;
  double total = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < columns; ++c) {
      const double n = table[r * columns + c];
      index += pairs(n);
      row_sums[r] += n;
      column_sums[c] += n;
      total += n;
    }
  }
  double a = 0.0;
  double b = 0.0;
  for (const double n : row_sums) {
    a += pairs(n);
  }
  for (const double n : column_sums) {
    b += pairs(n);
  }
  const double expected = total < 2.0 ? 0.0 : a * b / pairs(total);
  const double maximum = 0.5 * (a + b);
  return maximum == expected ? 1.0 : (index - expected) / (maximum - expected);
}

}  // namespace

ClusterReportOp::ClusterReportOp(ClusterReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("ClusterReportOp needs a report");
  }
}

void ClusterReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(points, "points").queue_depth(2U);
  spec.input(labels, "labels").queue_depth(2U);
  spec.input(clusters, "clusters").queue_depth(2U);
}

holoscan::Contract ClusterReportOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{points, labels, clusters}},
      .window = 0ns});
  result.input(points, holoscan::Exact{});
  result.input(labels, holoscan::Exact{});
  result.input(clusters, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> ClusterReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto cloud = points.receive();
  auto label_sample = labels.receive();
  auto cluster_sample = clusters.receive();
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  if (!label_sample) {
    return holoscan::make_unexpected(std::move(label_sample).error());
  }
  if (!cluster_sample) {
    return holoscan::make_unexpected(std::move(cluster_sample).error());
  }
  const int width = config_.width;
  const int height = config_.height;
  const std::span<const std::int64_t> shape = cloud->data.shape_span();
  const std::span<const std::int64_t> label_shape = label_sample->data.shape_span();
  const std::span<const std::int64_t> summary_shape = cluster_sample->data.shape_span();
  if (shape.size() != 3 || shape[0] != height || shape[1] != width || shape[2] != 3 ||
      label_shape.size() != 2 || label_shape[0] != height || label_shape[1] != width ||
      summary_shape.size() != 2 || summary_shape[1] != hc::kSummaryFields) {
    return holoscan::make_unexpected(
        invalid("expected the organized [H, W, 3] cloud, [H, W] labels and [K, 26] clusters"));
  }
  const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const auto k = static_cast<std::size_t>(summary_shape[0]);
  xyz_.resize(3 * pixels);
  labels_.resize(pixels);
  summaries_.resize(k);
  if (auto copied =
          cloud->data.copy_to_host(xyz_.data(), xyz_.size() * sizeof(float), context.cuda_stream());
      !copied) {
    return copied;
  }
  if (auto copied = label_sample->data.copy_to_host(labels_.data(), pixels * sizeof(std::int32_t),
                                                    context.cuda_stream());
      !copied) {
    return copied;
  }
  if (auto copied = cluster_sample->data.copy_to_host(
          summaries_.data(), k * sizeof(hc::ClusterSummary), context.cuda_stream());
      !copied) {
    return copied;
  }

  ClusterReport& report = *config_.report;
  const std::uint64_t frame_id = cloud->metadata.frame_id;
  report.frame_gaps += frame_id != last_frame_id_ + 1U ? 1U : 0U;
  last_frame_id_ = frame_id;
  const hc::SyntheticScene scene =
      hc::synthetic_scene(width, height, static_cast<std::uint32_t>(frame_id - 1U));

  // Each valid point's true object, objects below the minimum size as noise.
  constexpr std::size_t kObjects = hc::kSceneGroups;
  std::vector<std::int32_t> truth(pixels, -1);
  std::array<std::uint32_t, kObjects> object_points{};
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * width + u;
      const float* p = xyz_.data() + 3 * i;
      const std::int32_t group = hc::synthetic_scene_pixel(scene, u, v).group;
      if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) && group >= 0) {
        truth[i] = group;
        ++object_points[static_cast<std::size_t>(group)];
      }
    }
  }
  std::uint32_t objects = 0;
  for (std::size_t i = 0; i < pixels; ++i) {
    if (truth[i] >= 0 &&
        object_points[static_cast<std::size_t>(truth[i])] < config_.min_cluster_size) {
      truth[i] = -1;
    }
  }
  for (const std::uint32_t n : object_points) {
    objects += n >= config_.min_cluster_size ? 1U : 0U;
  }
  report.clusters += k;
  if (k != objects) {
    ++report.wrong_counts;
  }

  // The contingency table of labels (rows, -1 first) against the truth.
  const std::size_t columns = kObjects + 1U;
  std::vector<double> table((k + 1U) * columns, 0.0);
  std::vector<std::uint32_t> label_points(k, 0U);
  std::vector<std::array<double, 3>> truth_centroid(kObjects, {0.0, 0.0, 0.0});
  std::vector<std::array<float, 6>> truth_box(
      kObjects, {INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY});
  bool consistent = true;
  for (std::size_t i = 0; i < pixels; ++i) {
    const float* p = xyz_.data() + 3 * i;
    const bool valid = std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
    const std::int32_t label = labels_[i];
    if (label >= static_cast<std::int32_t>(k) || (!valid && label != -1)) {
      consistent = false;
      continue;
    }
    if (label >= 0) {
      ++label_points[static_cast<std::size_t>(label)];
    }
    if (!valid) {
      continue;
    }
    table[static_cast<std::size_t>(label + 1) * columns + static_cast<std::size_t>(truth[i] + 1)] +=
        1.0;
    if (truth[i] >= 0) {
      auto& c = truth_centroid[static_cast<std::size_t>(truth[i])];
      auto& box = truth_box[static_cast<std::size_t>(truth[i])];
      for (int axis = 0; axis < 3; ++axis) {
        c[axis] += p[axis];
        box[axis] = std::min(box[axis], p[axis]);
        box[3 + axis] = std::max(box[3 + axis], p[axis]);
      }
    }
  }
  for (std::size_t c = 0; c < k; ++c) {
    consistent = consistent && summaries_[c].id == static_cast<float>(c) &&
                 summaries_[c].point_count == static_cast<float>(label_points[c]) &&
                 (c == 0 || summaries_[c - 1].point_count >= summaries_[c].point_count);
  }
  report.inconsistent += consistent ? 0U : 1U;
  const double rand_index = adjusted_rand_index(table, columns);
  report.min_rand_index = std::min(report.min_rand_index, rand_index);

  // Each object's cluster: the one holding most of its points.
  for (std::size_t g = 0; g < kObjects; ++g) {
    if (object_points[g] < config_.min_cluster_size) {
      continue;
    }
    std::size_t best = 0;
    for (std::size_t c = 1; c < k; ++c) {
      if (table[(c + 1) * columns + g + 1] > table[(best + 1) * columns + g + 1]) {
        best = c;
      }
    }
    if (k == 0) {
      ++report.wrong_objects;
      continue;
    }
    const double shared = table[(best + 1) * columns + g + 1];
    double truth_total = 0.0;
    for (std::size_t c = 0; c <= k; ++c) {
      truth_total += table[c * columns + g + 1];
    }
    const double iou = shared / (truth_total + summaries_[best].point_count - shared);
    double centroid_error = 0.0;
    double box_error = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double mean = truth_centroid[g][axis] / truth_total;
      centroid_error = std::max(centroid_error, std::abs(summaries_[best].centroid[axis] - mean));
      box_error = std::max(
          {box_error,
           std::abs(static_cast<double>(summaries_[best].min[axis]) - truth_box[g][axis]),
           std::abs(static_cast<double>(summaries_[best].max[axis]) - truth_box[g][3 + axis])});
    }
    report.min_iou = std::min(report.min_iou, iou);
    report.max_centroid_error_m = std::max(report.max_centroid_error_m, centroid_error);
    report.max_box_error_m = std::max(report.max_box_error_m, box_error);
    if (!(iou >= kMinIoU && centroid_error <= kMaxCentroidErrorM && box_error <= kMaxBoxErrorM)) {
      ++report.wrong_objects;
    }
  }

  ++report.frames;
  if (config_.finished) {
    config_.finished->fetch_add(1U, std::memory_order_release);
  }
  if (!finished_ && report.frames >= config_.expected_frames) {
    finished_ = true;
    context.request_completion();
  }
  return {};
}

}  // namespace clustering_app
