// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// IspOp: a raw Bayer image in, a display-ready RGB image out, through
// IspPipeline (isp_pipeline.hpp) on the configured backend, with the frame's
// histograms and statistics when asked.
//
// The image is emitted before the statistics. Should a statistics emit fail,
// as into a full queue of a blocking connection, the image has already gone
// out and compute() returns the error: connect consumers that only need the
// newest frame with QueuePolicy::kDropOldest.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/isp/isp_params.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::isp {

// How the input's samples are stored, beyond what its encoding says.
enum class Packing : std::uint8_t {
  kNone,       // One uint8 or uint16 per sample, as the tensor's element type says.
  kMipiRaw10,  // MIPI CSI-2 RAW10 in a uint8 [H, row bytes] tensor.
  kMipiRaw12,  // MIPI CSI-2 RAW12 in a uint8 [H, row bytes] tensor.
};

struct IspOpConfig {
  // kCuda (the default), kCpu, or kAuto, which picks at construction (see resolve_auto()).
  Backend backend{Backend::kCuda};
  IspParams params;
  OutputFormat output_format{OutputFormat::kRgb8};
  // The mosaic of MONO8 and MONO16 inputs, which name none. A BAYER_* input
  // names its own; when this is set too, the two must agree.
  std::optional<BayerPattern> pattern;
  // Significant bits of each sample; 0 takes the image's significant_bits, or
  // without them the whole container (8, 16; 10 and 12 when packed).
  int bit_depth{0};
  Packing packing{Packing::kNone};
  // Declare the `histogram` and `stats` outputs.
  bool stats{false};
  // Largest image the outputs are sized for, 2..16384.
  int max_width{3840};
  int max_height{2160};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const IspOpConfig& config);

class IspOp final : public holoscan::Operator<> {
 public:
  // Resolves Backend::kAuto.
  // @throws std::invalid_argument if `config` is not valid.
  explicit IspOp(IspOpConfig config);
  ~IspOp() override;
  IspOp(const IspOp&) = delete;
  IspOp& operator=(const IspOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The backend this operator runs: kCuda or kCpu.
  [[nodiscard]] Backend backend() const noexcept { return backend_; }

  // BAYER_RGGB, BAYER_BGGR, BAYER_GRBG, BAYER_GBRG, MONO8 or MONO16; uint8 or
  // uint16 [H, W] or [H, W, 1], rows packed or padded; with packing, uint8
  // [H, row bytes] and the descriptor's width. In host or CUDA device memory.
  holoscan::Input<holoscan::schema::ImageT> raw;
  // RGB8, RGBA8, RGB16 or RGB32F [H, W, channels], packed: in device memory
  // for the cuda backend, host memory for cpu. The descriptor keeps the raw
  // image's header, exposure time and gain; the sample its capture time and frame id.
  holoscan::Output<holoscan::schema::ImageT> image;
  // With `stats`: uint32 [kHistogramChannels, kHistogramBins].
  holoscan::Output<holoscan::Tensor> histogram;
  // With `stats`: float32 [stats::kStatsSize].
  holoscan::Output<holoscan::Tensor> stats;

 private:
  struct State;

  IspOpConfig config_;
  Backend backend_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::isp
