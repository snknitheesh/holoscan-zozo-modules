// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// SceneSourceOp: frames of holoscan-pcl-filters' synthetic scene as a depth
// camera's point clouds: organized PointCloud messages with colors, rendered
// on the GPU into device memory, a fixed number on a clock, then idle.
//
// The filters cannot send a refused output again, so a full queue must never
// meet them: with `finished`, the source keeps one frame in flight, sending
// the next only once the last consumer has counted the previous one there.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/point_cloud.hpp>

namespace pcl_filters_app {

// The clouds' frame: the synthetic camera's optical frame.
inline constexpr const char* kCameraFrame = "camera_optical";

struct SceneSourceConfig {
  int width{640};
  int height{480};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Frames the last consumer has finished; none sends on every tick.
  std::shared_ptr<const std::atomic<std::uint32_t>> finished;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const SceneSourceConfig& config);

class SceneSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit SceneSourceOp(SceneSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // [H, W, 3] points of synthetic_scene(width, height, f) in metres in
  // kCameraFrame, NaN without depth, and their colors; frame f has frame
  // number f + 1.
  holoscan::Output<holoscan::messages::schema::PointCloudT> point_cloud;

 private:
  SceneSourceConfig config_;
  std::uint32_t emitted_{};
};

}  // namespace pcl_filters_app
