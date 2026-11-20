// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CheckOp: the demo's ground truth. Takes each processed image with its
// statistics and histograms, measures the image's PSNR against the synthetic
// camera's truth through the same tone curve, rebuilds the image's histograms
// from its pixels and requires IspOp's to match them bin for bin, keeps the
// white balance gains and exposure error the ISP reports, and counts the
// frame; the last frame ends the run. With the cuda backend all of it runs on
// the GPU and the operator waits once per frame, for a small result; with the
// cpu backend it runs on the host.

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/isp/isp_params.hpp>
#include <holoscan/isp/synthetic_raw.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace isp_demo {

struct Report {
  std::uint32_t frames{};
  std::uint32_t frame_gaps{};      // Frames that arrived out of order or not at all.
  std::uint32_t bad_histograms{};  // Frames whose histograms differ from their image's.
  double worst_psnr{99.0};
  double psnr_sum{};
  std::array<float, 3> wb_gains{};  // The last frame's.
  float ae_error_ev{};              // The last frame's.
  float mean_luma{};
};

struct CheckConfig {
  holoscan::isp::SyntheticRawConfig camera;
  holoscan::isp::IspParams params;
  std::uint32_t expected_frames{};
  std::shared_ptr<Report> report;
};

class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(CheckConfig config);
  ~CheckOp() override;
  CheckOp(const CheckOp&) = delete;
  CheckOp& operator=(const CheckOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // RGB8 [H, W, 3].
  holoscan::Input<holoscan::schema::ImageT> image;
  // float32 [stats::kStatsSize].
  holoscan::Input<holoscan::Tensor> stats;
  // uint32 [kHistogramChannels, kHistogramBins].
  holoscan::Input<holoscan::Tensor> histogram;

 private:
  void release() noexcept;

  CheckConfig config_;
  std::vector<float> lut_;
  // The host path's copies.
  std::vector<std::uint8_t> pixels_;
  std::vector<std::uint32_t> counts_;
  // The GPU path's buffers.
  float* device_lut_{};
  std::uint32_t* device_bins_{};
  void* device_result_{};
  void* host_result_{};  // Pinned.
  std::string start_error_;
};

}  // namespace isp_demo
