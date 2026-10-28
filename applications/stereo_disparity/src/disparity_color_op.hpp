// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// DisparityColorOp: a disparity image in, an RGBA8 color map of it out, for
// HoloViz: the jet map from blue (far) to dark red (near), black where invalid.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace stereo_disparity_app {

// Float disparity, rows `pitch` bytes apart, as packed RGBA8 rows: `max_disparity`
// and nearer dark red, invalid (0) black. Queued on `stream`.
[[nodiscard]] cudaError_t colorize_disparity(const float* disparity, std::size_t pitch, int width,
                                             int height, float max_disparity, std::uint8_t* rgba,
                                             cudaStream_t stream);

struct DisparityColorConfig {
  // The disparity shown dark red; larger ones are too.
  float max_disparity{128.0F};
  // Largest image the output is sized for.
  int max_width{1920};
  int max_height{1080};
};

class DisparityColorOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for an unusable config.
  explicit DisparityColorOp(DisparityColorConfig config);
  ~DisparityColorOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO32F [H, W] in CUDA device memory, as StereoDisparityOp emits it.
  holoscan::Input<holoscan::schema::ImageT> disparity;
  // RGBA8 [H, W, 4] on the GPU, with the disparity's capture time.
  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  DisparityColorConfig config_;
  cudaStream_t own_stream_{};
};

}  // namespace stereo_disparity_app
