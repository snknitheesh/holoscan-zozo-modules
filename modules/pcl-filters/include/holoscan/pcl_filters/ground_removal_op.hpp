// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// GroundRemovalOp: finds the ground plane by RANSAC on the GPU (or takes a
// height threshold), removes the points on it, and emits the plane. See
// cloud_filter_op.hpp for the cloud ports and GroundSegmenter in filters.hpp
// for the method.

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/pcl_filters/cloud_filter_op.hpp>
#include <holoscan/pcl_filters/filters.hpp>

namespace holoscan::pcl_filters {

struct GroundRemovalConfig {
  GroundParams ground;
  // Declare the `plane` output. An output must be connected when it exists.
  bool emit_plane{true};
  // The output's layout, size and fields.
  CloudSettings cloud;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const GroundRemovalConfig& config);

class GroundRemovalOp final : public CloudFilterOp {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit GroundRemovalOp(GroundRemovalConfig config);
  ~GroundRemovalOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // With `emit_plane`: float32 [4] (a, b, c, d), a x + b y + c z + d = 0 with
  // (a, b, c) a unit normal toward `up`, in metres in the cloud's frame
  // (its header's frame_id), in device memory; NaN when no plane was found.
  // Sent after the cloud, with its sample's capture time and frame number.
  holoscan::Output<holoscan::Tensor> plane;

 private:
  GroundRemovalConfig config_;
  std::unique_ptr<GroundSegmenter> segmenter_;
  // The plane without `emit_plane`, which the filter needs all the same.
  float* plane_scratch_{};
};

}  // namespace holoscan::pcl_filters
