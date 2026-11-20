// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "scene_source_op.hpp"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/pcl_filters/synthetic_scene.hpp>

namespace pcl_filters_app {
namespace {

namespace hm = holoscan::messages;
namespace pf = holoscan::pcl_filters;

holoscan::expected<void, holoscan::Error> rendered(cudaError_t status) {
  if (status == cudaSuccess) {
    return {};
  }
  return holoscan::make_unexpected(
      holoscan::Error{holoscan::ErrorCode::kFailure,
                      std::string("SceneSourceOp: render: ") + cudaGetErrorString(status)});
}

}  // namespace

void validate_config(const SceneSourceConfig& config) {
  if (config.width < 64 || config.height < 48 || config.width > 4096 || config.height > 4096) {
    throw std::invalid_argument("scene images must be 64x48 to 4096x4096");
  }
  if (config.count == 0 || config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the scene source needs a positive count and period");
  }
}

SceneSourceOp::SceneSourceOp(SceneSourceConfig config) : config_(std::move(config)) {
  validate_config(config_);
}

void SceneSourceOp::setup(holoscan::OperatorSpec& spec) {
  hm::declare_point_cloud_output(
      spec, point_cloud, "point_cloud",
      {.max_points = static_cast<std::uint64_t>(config_.width) * config_.height, .colors = true});
}

holoscan::Contract SceneSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SceneSourceOp::start() { emitted_ = 0; }

holoscan::expected<void, holoscan::Error> SceneSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the reporter completes the run.
  }
  if (config_.finished && config_.finished->load(std::memory_order_acquire) < emitted_) {
    return {};  // The previous frame is still in flight.
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "SceneSourceOp: the runtime gave no CUDA stream"});
  }
  const pf::SyntheticScene scene = pf::synthetic_scene(config_.width, config_.height, emitted_);
  const hm::PointCloudShape shape = hm::organized_cloud(config_.height, config_.width);
  auto loan = hm::allocate_point_cloud(point_cloud, shape, {.colors = true}, stream);
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  if (auto written =
          loan->write_and_commit(hm::PointCloudFields::points, stream,
                                 [&](const holoscan::TensorOutputWriteGuard& points) {
                                   return rendered(pf::render_synthetic_scene(
                                       scene, nullptr, nullptr, points.data_as<float>(), stream));
                                 });
      !written) {
    return written;
  }
  if (auto written = loan->write_and_commit(hm::PointCloudFields::colors, stream,
                                            [&](const holoscan::TensorOutputWriteGuard& colors) {
                                              return rendered(pf::render_synthetic_scene(
                                                  scene, nullptr, colors.data_as<std::uint8_t>(),
                                                  nullptr, stream));
                                            });
      !written) {
    return written;
  }
  const holoscan::TimePoint capture = context.activation_time();
  const hm::Header header{.frame_id = kCameraFrame, .capture_timestamp_ns = capture.timestamp_ns};
  // Frame numbers count from 1; 0 means none.
  auto sent = hm::emit_point_cloud(point_cloud, std::move(*loan), header, shape, -1,
                                   {.capture_time = capture, .frame_id = emitted_ + 1U});
  if (!sent && sent.error().code == holoscan::ErrorCode::kBackpressured) {
    return {};  // A consumer is still busy: render the frame again on the next tick.
  }
  if (!sent) {
    return sent;
  }
  ++emitted_;
  return {};
}

}  // namespace pcl_filters_app
