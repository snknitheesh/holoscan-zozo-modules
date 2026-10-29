// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CloudFilterOp: what the module's operators share. Each takes a point cloud,
// optionally with its colors, and emits the filtered cloud in the same
// layout as holoscan-point-cloud's:
//
//   points           float32 [N, 3], or organized [H, W, 3]; packed, in host,
//                    pinned or device memory on the operator's GPU. A point
//                    with a non-finite coordinate is invalid.
//   colors           uint8 RGB of the points' shape (with `color`).
//   filtered_points  float32, in device memory: with OutputLayout::kKeep the
//                    input's shape, removed points NaN; with kDense, [M, 3]
//                    of the kept points in input order. The shape's first
//                    extent is then the count.
//   filtered_colors  uint8 RGB of the filtered points' shape (with `color`),
//                    black where a point was removed.
//
// With colors, the operator runs once per cloud when the colors with exactly
// the same capture time have arrived, as DepthToPointCloudOp emits them. The
// outputs carry the cloud's capture time and frame id; the points go out
// before the colors.
//
// A dense output's size is known only once the filter has run, so the
// operator waits for its GPU work once per frame before allocating it; an
// organized output never waits.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/pcl_filters/filters.hpp>

namespace holoscan::pcl_filters {

enum class OutputLayout : std::uint8_t {
  kKeep,   // The input's shape; removed points NaN. Organized clouds stay organized.
  kDense,  // [M, 3] of the kept points, in input order.
};

// The settings every operator of the module has.
struct CloudSettings {
  // Declare the `colors` input and the `filtered_colors` output.
  bool color{false};
  OutputLayout layout{OutputLayout::kKeep};
  // Largest cloud accepted, 1..kMaxPointsLimit points; a larger one fails the
  // frame. It sizes the outputs. The default fits 1920x1080.
  std::uint32_t max_points{1920U * 1080U};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_settings(const CloudSettings& settings);

class CloudFilterOp : public holoscan::Operator<> {
 public:
  ~CloudFilterOp() override;
  CloudFilterOp(const CloudFilterOp&) = delete;
  CloudFilterOp& operator=(const CloudFilterOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;

  holoscan::Input<holoscan::Tensor> points;
  holoscan::Input<holoscan::Tensor> colors;
  holoscan::Output<holoscan::Tensor> filtered_points;
  holoscan::Output<holoscan::Tensor> filtered_colors;

 protected:
  // A received cloud, on the GPU.
  struct Cloud {
    holoscan::Sample<holoscan::Tensor> points;
    std::optional<holoscan::Sample<holoscan::Tensor>> colors;
    const float* xyz{};
    const std::uint8_t* rgb{};
    std::uint32_t count{};
    cudaStream_t stream{};
  };

  // @throws std::invalid_argument if `settings` are not valid.
  CloudFilterOp(std::string name, CloudSettings settings);

  // Receives and checks the next cloud and its colors, copying host input to
  // the GPU.
  [[nodiscard]] holoscan::expected<Cloud, holoscan::Error> receive(
      holoscan::ExecutionContext& context);
  // A device buffer of one byte per point, for a keep mask.
  [[nodiscard]] holoscan::expected<std::uint8_t*, holoscan::Error> mask(const Cloud& cloud);
  // Emits the points `keep` marks, in the configured layout.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_kept(const Cloud& cloud,
                                                                    const std::uint8_t* keep);
  // Emits a dense cloud of the `device_count` points that `write` writes:
  // waits for the count, allocates the outputs and calls write(xyz, rgb) on
  // the cloud's stream, rgb null without colors.
  using DenseWriter = std::function<cudaError_t(float* xyz, std::uint8_t* rgb)>;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_dense(
      const Cloud& cloud, const std::uint32_t* device_count, const DenseWriter& write);

  [[nodiscard]] holoscan::Error invalid(const std::string& message) const;
  [[nodiscard]] holoscan::Error cuda_failure(const char* what, cudaError_t status) const;
  // The error of a failed emit on `port`, naming the operator and the port.
  [[nodiscard]] holoscan::unexpected<holoscan::Error> emit_failure(const char* port,
                                                                   holoscan::Error error) const;
  [[nodiscard]] const CloudSettings& settings() const noexcept { return settings_; }

 private:
  struct Impl;

  std::string name_;
  CloudSettings settings_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::pcl_filters
