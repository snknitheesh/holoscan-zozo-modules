// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "trajectory_view_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/icp/icp_op.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

namespace icp_odometry_app {
namespace {

using std::chrono_literals::operator""ns;
namespace icp = holoscan::icp;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
namespace viz = holoscan::visualization;
constexpr auto kHeaderBytes = static_cast<std::int64_t>(viz::kTextRunHeaderBytes);
constexpr std::int64_t kLabelRowBytes = kHeaderBytes + kViewLabelBytes;

using Color = std::array<std::uint8_t, 3>;
constexpr Color kBackground{24, 24, 28};
constexpr Color kGrid{52, 52, 60};
constexpr Color kTruth{60, 200, 90};
constexpr Color kEstimate{255, 150, 40};
constexpr Color kNear{60, 220, 80};
constexpr Color kFair{240, 210, 60};
constexpr Color kFar{230, 60, 50};

holoscan::TensorOutputSpec host_tensor(std::uint32_t rank, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = kUInt8,
              .rank = rank,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

int panel_width(const icp::PinholeCamera& camera) { return std::min(camera.width, 640); }

int panel_height(const icp::PinholeCamera& camera) {
  return static_cast<int>(
      std::lround(static_cast<double>(camera.height) * panel_width(camera) / camera.width));
}

// Reads a host or device tensor of `count` floats (or doubles) into `out`.
template <typename T>
holoscan::expected<void, holoscan::Error> read(const holoscan::Tensor& tensor, std::size_t count,
                                               std::vector<T>& out, cudaStream_t stream) {
  if (tensor.size() != static_cast<std::int64_t>(count) || tensor.dtype().bits != sizeof(T) * 8U ||
      tensor.dtype().code != kDLFloat) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument, "TrajectoryViewOp: an input has the wrong size"});
  }
  out.resize(count);
  // The host reads it: wait for its producer first.
  if (auto ready = tensor.synchronize_for_host(); !ready) {
    return ready;
  }
  return tensor.copy_to_host(out.data(), count * sizeof(T), stream);
}

void put(std::uint8_t* rgba, int width, int height, int x, int y, const Color& c) {
  if (x < 0 || y < 0 || x >= width || y >= height) return;
  std::uint8_t* p = rgba + 4 * (static_cast<std::size_t>(y) * width + x);
  p[0] = c[0];
  p[1] = c[1];
  p[2] = c[2];
  p[3] = 255;
}

// A line from (x0, y0) to (x1, y1), Bresenham's.
void line(std::uint8_t* rgba, int width, int height, int x0, int y0, int x1, int y1,
          const Color& c) {
  const int dx = std::abs(x1 - x0);
  const int dy = -std::abs(y1 - y0);
  const int sx = x0 < x1 ? 1 : -1;
  const int sy = y0 < y1 ? 1 : -1;
  int error = dx + dy;
  for (int steps = 0; steps < 8192; ++steps) {
    put(rgba, width, height, x0, y0, c);
    if (x0 == x1 && y0 == y1) break;
    const int e2 = 2 * error;
    if (e2 >= dy) {
      error += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      error += dx;
      y0 += sy;
    }
  }
}

void label(std::uint8_t* row, float y, std::uint32_t color, const std::string& text) {
  std::memset(row, 0, static_cast<std::size_t>(kLabelRowBytes));
  const viz::TextRunHeader header{.x = 0.01F, .y = y, .size_px = 18.0F, .color = color};
  std::memcpy(row, &header, sizeof(header));
  std::memcpy(row + viz::kTextRunHeaderBytes, text.data(),
              std::min<std::size_t>(text.size(), kViewLabelBytes));
}

}  // namespace

int view_width(const icp::PinholeCamera& camera) { return 2 * panel_width(camera); }
int view_height(const icp::PinholeCamera& camera) { return panel_height(camera); }

TrajectoryViewOp::TrajectoryViewOp(TrajectoryViewConfig config)
    : config_(std::move(config)),
      panel_width_(panel_width(config_.camera)),
      panel_height_(panel_height(config_.camera)) {
  // Fit the plot to the whole path, with a margin.
  min_x_ = min_z_ = std::numeric_limits<double>::max();
  max_x_ = max_z_ = std::numeric_limits<double>::lowest();
  for (std::uint32_t frame = 0; frame <= config_.path_frames; ++frame) {
    const icp::Matrix4 pose = icp::synthetic_trajectory(frame, config_.path_frames);
    min_x_ = std::min(min_x_, pose[3]);
    max_x_ = std::max(max_x_, pose[3]);
    min_z_ = std::min(min_z_, pose[11]);
    max_z_ = std::max(max_z_, pose[11]);
  }
  // Odometry starts at the path's first pose: the estimate is relative to it.
  first_truth_ = icp::synthetic_trajectory(0, config_.path_frames);
  const double margin = 0.25 * std::max(max_x_ - min_x_, max_z_ - min_z_) + 0.1;
  min_x_ -= margin;
  max_x_ += margin;
  min_z_ -= margin;
  max_z_ += margin;
}

void TrajectoryViewOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(pose, "pose").queue_depth(2U);
  spec.input(true_pose, "true_pose").queue_depth(2U);
  spec.input(stats, "stats").queue_depth(2U);
  spec.input(point_cloud, "point_cloud").queue_depth(2U);
  spec.input(aligned, "aligned").queue_depth(2U);
  const auto pixels = static_cast<std::uint64_t>(view_width(config_.camera)) *
                      static_cast<std::uint64_t>(view_height(config_.camera));
  spec.output(view, "view")
      .produces_tensor(host_tensor(3U, pixels * 4U))
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
  spec.output(labels, "labels")
      .produces_tensor(
          host_tensor(2U, static_cast<std::uint64_t>(kViewLabelRows * kLabelRowBytes)));
}

holoscan::Contract TrajectoryViewOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::ports(pose, true_pose, stats, point_cloud, aligned), .window = 0ns});
  result.input(pose, holoscan::Exact{});
  result.input(true_pose, holoscan::Exact{});
  result.input(stats, holoscan::Exact{});
  result.input(point_cloud, holoscan::Exact{});
  result.input(aligned, holoscan::Exact{});
  return result;
}

void TrajectoryViewOp::draw_trajectories(std::uint8_t* rgba) const {
  const int width = view_width(config_.camera);
  const int w = panel_width_;
  const int h = panel_height_;
  // One scale for both axes, so the path keeps its shape.
  const double scale = std::min((w - 1) / (max_x_ - min_x_), (h - 1) / (max_z_ - min_z_));
  const auto to_pixel = [&](const icp::Matrix4& m) {
    const int x = static_cast<int>(std::lround((m[3] - min_x_) * scale));
    const int y = h - 1 - static_cast<int>(std::lround((m[11] - min_z_) * scale));
    return std::array<int, 2>{x, y};
  };
  // A grid every half meter.
  for (double gx = std::ceil(min_x_ * 2.0) / 2.0; gx <= max_x_; gx += 0.5) {
    const int x = static_cast<int>(std::lround((gx - min_x_) * scale));
    line(rgba, width, h, x, 0, x, h - 1, kGrid);
  }
  for (double gz = std::ceil(min_z_ * 2.0) / 2.0; gz <= max_z_; gz += 0.5) {
    const int y = h - 1 - static_cast<int>(std::lround((gz - min_z_) * scale));
    line(rgba, width, h, 0, y, w - 1, y, kGrid);
  }
  for (const auto* path : {&truth_, &estimated_}) {
    const Color& color = path == &truth_ ? kTruth : kEstimate;
    for (std::size_t i = 1; i < path->size(); ++i) {
      const auto a = to_pixel((*path)[i - 1U]);
      const auto b = to_pixel((*path)[i]);
      line(rgba, width, h, a[0], a[1], b[0], b[1], color);
    }
    if (!path->empty()) {
      // The camera now, and where it looks (its z axis, from above).
      const icp::Matrix4& m = path->back();
      const auto p = to_pixel(m);
      for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) put(rgba, width, h, p[0] + dx, p[1] + dy, color);
      }
      const double heading = 0.25 * scale;
      line(rgba, width, h, p[0], p[1], p[0] + static_cast<int>(std::lround(m[2] * heading)),
           p[1] - static_cast<int>(std::lround(m[10] * heading)), color);
    }
  }
}

void TrajectoryViewOp::draw_alignment(std::uint8_t* rgba) const {
  const int width = view_width(config_.camera);
  const icp::PinholeCamera& k = config_.camera;
  const double to_panel = static_cast<double>(panel_width_) / k.width;
  const std::vector<float>& base = previous_.empty() ? current_ : previous_;
  // The previous frame's depth in gray, nearer brighter.
  for (int y = 0; y < panel_height_; ++y) {
    for (int x = 0; x < panel_width_; ++x) {
      const int u = std::min(k.width - 1, static_cast<int>(x / to_panel));
      const int v = std::min(k.height - 1, static_cast<int>(y / to_panel));
      const float z = base[3U * (static_cast<std::size_t>(v) * k.width + u) + 2U];
      if (!std::isfinite(z)) continue;
      const auto g = static_cast<std::uint8_t>(std::clamp(200.0F - 22.0F * z, 30.0F, 200.0F));
      put(rgba, width, panel_height_, panel_width_ + x, y, Color{g, g, g});
    }
  }
  if (previous_.empty()) return;
  // The current cloud in the previous camera, nearest point per panel pixel.
  std::vector<float> nearest(static_cast<std::size_t>(panel_width_) * panel_height_,
                             std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i + 2U < aligned_.size(); i += 3U) {
    const float px = aligned_[i];
    const float py = aligned_[i + 1U];
    const float pz = aligned_[i + 2U];
    if (!(pz > 0.0F) || !std::isfinite(px) || !std::isfinite(py)) continue;
    const auto u = static_cast<int>(std::lround(k.fx * px / pz + k.cx));
    const auto v = static_cast<int>(std::lround(k.fy * py / pz + k.cy));
    if (u < 0 || v < 0 || u >= k.width || v >= k.height) continue;
    const int x = static_cast<int>(u * to_panel);
    const int y = static_cast<int>(v * to_panel);
    if (x >= panel_width_ || y >= panel_height_) continue;
    float& z = nearest[static_cast<std::size_t>(y) * panel_width_ + x];
    if (pz >= z) continue;
    z = pz;
    const float* q = previous_.data() + 3U * (static_cast<std::size_t>(v) * k.width + u);
    if (!std::isfinite(q[2])) continue;
    const float d = std::hypot(px - q[0], py - q[1], pz - q[2]);
    put(rgba, width, panel_height_, panel_width_ + x, y,
        d <= 0.02F ? kNear : (d <= 0.05F ? kFair : kFar));
  }
}

holoscan::expected<void, holoscan::Error> TrajectoryViewOp::compute(
    holoscan::ExecutionContext& context) {
  auto pose_sample = pose.receive();
  if (!pose_sample) return holoscan::make_unexpected(std::move(pose_sample).error());
  auto truth_sample = true_pose.receive();
  if (!truth_sample) return holoscan::make_unexpected(std::move(truth_sample).error());
  auto stats_sample = stats.receive();
  if (!stats_sample) return holoscan::make_unexpected(std::move(stats_sample).error());
  auto cloud_sample = point_cloud.receive();
  if (!cloud_sample) return holoscan::make_unexpected(std::move(cloud_sample).error());
  auto aligned_sample = aligned.receive();
  if (!aligned_sample) return holoscan::make_unexpected(std::move(aligned_sample).error());

  cudaStream_t stream = context.cuda_stream().get();
  const std::size_t floats = static_cast<std::size_t>(config_.camera.width) *
                             static_cast<std::size_t>(config_.camera.height) * 3U;
  if (!logged_copy_) {
    logged_copy_ = true;
    HOLOSCAN_LOG_INFO("TrajectoryViewOp: copies each frame's clouds to the host to draw them");
  }
  std::vector<double> estimate, truth, values;
  std::swap(previous_, current_);
  if (auto r = read(pose_sample->data, 16U, estimate, stream); !r) {
    return r;
  }
  if (auto r = read(truth_sample->data, 16U, truth, stream); !r) {
    return r;
  }
  if (auto r = read(stats_sample->data, icp::kStatsSize, values, stream); !r) {
    return r;
  }
  if (auto r = read(cloud_sample->data, floats, current_, stream); !r) {
    return r;
  }
  if (auto r = read(aligned_sample->data, floats, aligned_, stream); !r) {
    return r;
  }
  icp::Matrix4 estimated{};
  icp::Matrix4 world_from_camera{};
  std::copy(estimate.begin(), estimate.end(), estimated.begin());
  std::copy(truth.begin(), truth.end(), world_from_camera.begin());
  // The plot shows both in the world: the estimate starts where the truth does.
  estimated_.push_back(icp::compose(first_truth_, estimated));
  truth_.push_back(world_from_camera);

  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*pose_sample);
  const int width = view_width(config_.camera);
  const int height = view_height(config_.camera);
  auto image_loan = view.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{height, width, 4}, .dtype = kUInt8});
  if (!image_loan) return holoscan::make_unexpected(std::move(image_loan).error());
  if (auto written =
          image_loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                -> holoscan::expected<void, holoscan::Error> {
            auto* rgba = writer.data_as<std::uint8_t>();
            for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
              rgba[4 * i] = kBackground[0];
              rgba[4 * i + 1] = kBackground[1];
              rgba[4 * i + 2] = kBackground[2];
              rgba[4 * i + 3] = 255;
            }
            draw_trajectories(rgba);
            draw_alignment(rgba);
            return {};
          });
      !written) {
    return written;
  }
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = holoscan::schema::ImageEncoding_RGBA8;
  if (auto sent = view.emit_tensor(std::move(*image_loan), image, emit); !sent) {
    return sent;
  }

  const icp::TransformError error = icp::transform_error(
      estimated, icp::compose(icp::rigid_inverse(first_truth_), world_from_camera));
  char lines[kViewLabelRows][kViewLabelBytes + 1];
  std::snprintf(lines[0], sizeof(lines[0]), "frame %u/%u  green: truth  orange: estimate",
                static_cast<unsigned>(truth_.size()), static_cast<unsigned>(config_.count));
  std::snprintf(lines[1], sizeof(lines[1]), "pose error %.1f mm  %.2f deg",
                error.translation * 1000.0, error.rotation_rad * 180.0 / 3.14159265358979323846);
  std::snprintf(lines[2], sizeof(lines[2]), "fitness %.2f  rmse %.1f mm  iterations %d",
                values[icp::kStatFitness], values[icp::kStatRmse] * 1000.0,
                static_cast<int>(values[icp::kStatIterations]));
  auto label_loan = labels.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 2>{kViewLabelRows, kLabelRowBytes}, .dtype = kUInt8});
  if (!label_loan) return holoscan::make_unexpected(std::move(label_loan).error());
  if (auto written =
          label_loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                -> holoscan::expected<void, holoscan::Error> {
            auto* rows = writer.data_as<std::uint8_t>();
            for (std::int64_t row = 0; row < kViewLabelRows; ++row) {
              label(rows + row * kLabelRowBytes, 0.02F + 0.06F * static_cast<float>(row),
                    0xFFFFFFFFU, lines[row]);
            }
            return {};
          });
      !written) {
    return written;
  }
  return labels.emit(std::move(*label_loan), emit);
}

}  // namespace icp_odometry_app
