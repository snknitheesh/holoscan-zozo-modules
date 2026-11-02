// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ExposureFusionOp: one 8-bit image of a high-dynamic-range scene, on the GPU,
// either from 2 or 3 differently exposed brackets (Mertens exposure fusion),
// matched by capture time, or from one high-bit-depth frame (Reinhard's global
// tone map).

#pragma once

#include <chrono>
#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/denoise/exposure_fusion.hpp>
#include <holoscan/denoise/fusion_params.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::denoise {

enum class FusionMode : std::uint8_t {
  kFusion,   // Brackets in `bracket0`, `bracket1` (and `bracket2`).
  kToneMap,  // One high-bit-depth frame in `hdr`.
};

struct ExposureFusionOpConfig {
  FusionMode mode{FusionMode::kFusion};
  // kFusion: 2 or 3.
  int brackets{3};
  // kFusion: the brackets of one fused image are those whose capture times lie
  // within this window of each other (inclusive); 0 needs equal capture times.
  // 0..1 s.
  std::chrono::nanoseconds match_window{0};
  FusionParams fusion;
  ToneMapParams tone_map;
  // Largest image the output is sized for, 1..16384.
  int max_width{3840};
  int max_height{2160};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const ExposureFusionOpConfig& config);

class ExposureFusionOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit ExposureFusionOp(ExposureFusionOpConfig config);
  ~ExposureFusionOp() override;
  ExposureFusionOp(const ExposureFusionOp&) = delete;
  ExposureFusionOp& operator=(const ExposureFusionOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // kFusion: MONO8, RGB8 or RGBA8 brackets of one size and format, uint8 [H, W]
  // or [H, W, channels], in host or CUDA device memory, in any order of exposure.
  // `bracket2` exists with 3 brackets.
  holoscan::Input<holoscan::schema::ImageT> bracket0;
  holoscan::Input<holoscan::schema::ImageT> bracket1;
  holoscan::Input<holoscan::schema::ImageT> bracket2;
  // kToneMap: MONO16, RGB16, RGBA16 (code values of significant_bits, or 16),
  // MONO32F, RGB32F or RGBA32F (linear, any scale), in host or CUDA device memory.
  holoscan::Input<holoscan::schema::ImageT> hdr;
  // The brackets' format (alpha 255), or MONO8, RGB8, RGBA8 for 1, 3, 4 HDR
  // channels; uint8 [H, W, channels], packed, in CUDA device memory. The
  // descriptor keeps bracket0's (or the HDR frame's) header; the sample its
  // capture time and frame id.
  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  struct State;

  ExposureFusionOpConfig config_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::denoise
