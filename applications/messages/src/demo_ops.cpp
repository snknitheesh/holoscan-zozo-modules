// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "demo_ops.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>

namespace messages_demo {
namespace {

using std::chrono_literals::operator""ns;

holoscan::Error failure(const std::string& what) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "messages demo: " + what};
}

// One cloud's values: frame `frame` of the synthetic scene. Every 9th point
// is invalid (NaN), every 13th normal unknown.
struct CloudValues {
  std::vector<float> points;
  std::vector<std::uint8_t> colors;
  std::vector<float> normals;
  std::int64_t valid{};
};

CloudValues cloud_values(std::uint32_t frame) {
  CloudValues v;
  const std::int64_t n = kCloudHeight * kCloudWidth;
  v.points.reserve(3 * n);
  v.colors.reserve(3 * n);
  v.normals.reserve(3 * n);
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t row = i / kCloudWidth;
    const std::int64_t column = i % kCloudWidth;
    const bool invalid = (i + frame) % 9 == 0;
    v.valid += invalid ? 0 : 1;
    // A plane 2 m ahead, tilted, moving 1 mm a frame.
    const float z = 2.0F + 0.01F * static_cast<float>(row) + 0.001F * static_cast<float>(frame);
    const float xyz[3] = {0.01F * static_cast<float>(column - kCloudWidth / 2),
                          0.01F * static_cast<float>(row - kCloudHeight / 2), z};
    const float normal[3] = {0.0F, -0.7071F, -0.7071F};
    for (int k = 0; k < 3; ++k) {
      v.points.push_back(invalid ? NAN : xyz[k]);
      v.colors.push_back(static_cast<std::uint8_t>((row * 5 + column * 3 + k * 80 + frame) & 255));
      v.normals.push_back(i % 13 == 0 ? NAN : normal[k]);
    }
  }
  return v;
}

double max_difference(const hm::Matrix3& a, const hm::Matrix3& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < 9; ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
  return worst;
}

double max_difference(const hm::Vector3& a, const hm::Vector3& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < 3; ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
  return worst;
}

// Every field that the files hold, exactly, and the extrinsics to 1e-9.
bool check_calibration(const hm::StereoCameraInfo& got, const hm::StereoCameraInfo& truth,
                       Report& report) {
  const double error =
      std::max(max_difference(got.right_from_left.rotation, truth.right_from_left.rotation),
               max_difference(got.right_from_left.translation, truth.right_from_left.translation));
  double worst = report.max_extrinsics_error.load();
  while (error > worst && !report.max_extrinsics_error.compare_exchange_weak(worst, error)) {
  }
  hm::CameraInfo left = got.left;
  hm::CameraInfo right = got.right;
  left.header = truth.left.header;
  right.header = truth.right.header;
  const bool ok = left == truth.left && right == truth.right && error < 1e-9 &&
                  got.header.frame_id == truth.header.frame_id && got.baseline == 0.0 &&
                  got.doffs == 0.0;
  if (!ok) {
    HOLOSCAN_LOG_ERROR(
        "messages demo: the camera info differs from the calibration written "
        "(extrinsics error {})",
        error);
  }
  return ok;
}

bool busy(const holoscan::Error& error) {
  return error.code == holoscan::ErrorCode::kBackpressured;
}

// Ends the run when the last checker is done.
void finish(Report& report, holoscan::ExecutionContext& context) {
  if (report.running.fetch_sub(1) == 1) context.request_completion();
}

bool same_bytes(const holoscan::Tensor& tensor, const void* expected, std::size_t bytes) {
  if (static_cast<std::size_t>(tensor.nbytes()) != bytes) return false;
  std::vector<std::uint8_t> host(bytes);
  if (!tensor.copy_to_host(host.data(), bytes)) return false;
  return std::memcmp(host.data(), expected, bytes) == 0;
}

}  // namespace

// --- FrameSourceOp -----------------------------------------------------------

void FrameSourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(left, "left");
  spec.output(right, "right");
}

holoscan::Contract FrameSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = period_});
  return result;
}

void FrameSourceOp::start() {
  sent_ = 0;
  pending_.reset();
}

holoscan::expected<void, holoscan::Error> FrameSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (sent_ >= count_) return {};
  if (!pending_) {
    pending_ =
        holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = sent_ + 1U};
    left_sent_ = false;
  }
  Frame frame;
  frame.capture_timestamp_ns = pending_->capture_time->timestamp_ns;
  frame.device_sequence = sent_ + 1U;
  if (!left_sent_) {
    frame.frame_id = "left";
    auto sent = left.emit(frame, *pending_);
    // A checker without room: the pair again on the next tick.
    if (!sent) return busy(sent.error()) ? holoscan::expected<void, holoscan::Error>{} : sent;
    left_sent_ = true;
  }
  frame.frame_id = "right";
  auto sent = right.emit(frame, *pending_);
  if (!sent) return busy(sent.error()) ? holoscan::expected<void, holoscan::Error>{} : sent;
  pending_.reset();
  ++sent_;
  return {};
}

// --- CalibrationCheckOp -------------------------------------------------------

void CalibrationCheckOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(left, "left").queue_depth(2U);
  spec.input(right, "right").queue_depth(2U);
  spec.input(camera_info, "camera_info");
}

holoscan::Contract CalibrationCheckOp::contract() const {
  return hm::camera_info_contract(camera_info, left, right);
}

void CalibrationCheckOp::start() {
  info_.reset();
  calibration_ok_ = false;
}

holoscan::expected<void, holoscan::Error> CalibrationCheckOp::compute(
    holoscan::ExecutionContext& context) {
  Report& report = *config_.report;
  auto info = info_.update(camera_info);
  if (!info) return holoscan::make_unexpected(std::move(info).error());
  if (*info != nullptr) {
    if (info_.changed()) calibration_ok_ = check_calibration(**info, config_.truth, report);
    report.frames_with_camera_info.fetch_add(1);
    if (!calibration_ok_) report.wrong_camera_info.fetch_add(1);
  }
  if (report.frames.fetch_add(1) + 1 == config_.frames) finish(report, context);
  return {};
}

// --- CloudSourceOp -------------------------------------------------------------

void CloudSourceOp::setup(holoscan::OperatorSpec& spec) {
  hm::declare_point_cloud_output(
      spec, cloud, "cloud",
      {.max_points = static_cast<std::uint64_t>(kCloudHeight * kCloudWidth),
       .colors = true,
       .normals = true,
       .memory_kind = config_.memory});
}

holoscan::Contract CloudSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void CloudSourceOp::start() { sent_ = 0; }

holoscan::expected<void, holoscan::Error> CloudSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (sent_ >= config_.count) return {};
  const CloudValues values = cloud_values(sent_);
  const hm::PointCloudShape shape = hm::organized_cloud(kCloudHeight, kCloudWidth);
  auto loan = hm::allocate_point_cloud(cloud, shape, {.colors = true, .normals = true},
                                       context.cuda_stream());
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  const bool host = config_.memory == holoscan::MemoryKind::kHost;
  const auto put = [&](auto field, const void* data,
                       std::size_t bytes) -> holoscan::expected<void, holoscan::Error> {
    const auto fill = [&](const holoscan::TensorOutputWriteGuard& writer)
        -> holoscan::expected<void, holoscan::Error> {
      if (host) {
        std::memcpy(writer.data(), data, bytes);
        return {};
      }
      // From pageable memory, the copy has read `data` when it returns.
      if (const cudaError_t copied = cudaMemcpyAsync(
              writer.data(), data, bytes, cudaMemcpyHostToDevice, context.cuda_stream().get());
          copied != cudaSuccess) {
        return holoscan::make_unexpected(failure(cudaGetErrorString(copied)));
      }
      return {};
    };
    return host ? loan->write_host_and_commit(field, fill)
                : loan->write_and_commit(field, context.cuda_stream(), fill);
  };
  if (auto r = put(hm::PointCloudFields::points, values.points.data(), values.points.size() * 4);
      !r) {
    return r;
  }
  if (auto r = put(hm::PointCloudFields::colors, values.colors.data(), values.colors.size()); !r) {
    return r;
  }
  if (auto r = put(hm::PointCloudFields::normals, values.normals.data(), values.normals.size() * 4);
      !r) {
    return r;
  }
  const holoscan::TimePoint now = context.activation_time();
  if (auto sent = hm::emit_point_cloud(cloud, std::move(*loan),
                                       hm::Header{"left", now.timestamp_ns, sent_ + 1U}, shape,
                                       values.valid, holoscan::EmitOptions{.capture_time = now});
      !sent) {
    // A checker without room: this cloud again on the next tick.
    return busy(sent.error()) ? holoscan::expected<void, holoscan::Error>{} : sent;
  }
  ++sent_;
  return {};
}

// --- CloudCheckOp ---------------------------------------------------------------

void CloudCheckOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(cloud, "cloud").queue_depth(2U);
}

holoscan::Contract CloudCheckOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{cloud});
  return result;
}

holoscan::expected<void, holoscan::Error> CloudCheckOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = cloud.receive();
  if (!sample) return holoscan::make_unexpected(std::move(sample).error());
  const hm::schema::PointCloudT& c = sample->data;
  Report& report = *config_.report;
  bool ok = static_cast<bool>(hm::check_point_cloud(c)) && c.header &&
            c.header->device_sequence >= 1 && c.organized && hm::is_present(c.colors) &&
            hm::is_present(c.normals) && !hm::is_present(c.intensity);
  if (ok) {
    const CloudValues expected =
        cloud_values(static_cast<std::uint32_t>(c.header->device_sequence - 1));
    ok = c.valid_count == expected.valid &&
         same_bytes(*c.points, expected.points.data(), expected.points.size() * 4) &&
         same_bytes(*c.colors, expected.colors.data(), expected.colors.size()) &&
         same_bytes(*c.normals, expected.normals.data(), expected.normals.size() * 4);
  }
  if (!ok) report.wrong_clouds.fetch_add(1);
  if (report.clouds.fetch_add(1) + 1 == config_.count) finish(report, context);
  return {};
}

}  // namespace messages_demo
