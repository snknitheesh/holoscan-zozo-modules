// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// VideoEncoderOp: images in, one H.264 or H.265 access unit out per image, encoded by the
// hardware (NVENC through the Video Codec SDK, or the Jetson's V4L2 encoder), optionally also
// written to a raw Annex-B file (.h264 / .h265) that ffplay and most players open.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <filesystem>
#include <memory>

#include <holoscan/codec/codec_types.hpp>
#include <holoscan/codec/encoded_video_frame_schema_traits.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::codec {

struct VideoEncoderConfig {
  EncoderSettings settings;
  // Largest image the output is sized for: each access unit may take up to this many pixels'
  // worth of NV12 (width x height x 1.5 bytes).
  int max_width{1920};
  int max_height{1080};
  // When set, every access unit is also appended to this file, which is created (or truncated) at
  // start: a raw Annex-B elementary stream.
  std::filesystem::path output_path;
  // On a change of image size, start a new stream at the new size (a keyframe with new parameter
  // sets); when false, fail the frame instead.
  bool allow_resolution_change{true};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const VideoEncoderConfig& config);

class VideoEncoderOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit VideoEncoderOp(VideoEncoderConfig config);
  ~VideoEncoderOp() override;
  VideoEncoderOp(const VideoEncoderOp&) = delete;
  VideoEncoderOp& operator=(const VideoEncoderOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  // Chooses the backend (for kAuto, the first that works here), logging which and why once, and
  // opens the output file. Fails when no backend can encode the codec.
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The backend in use once started, kAuto before. Safe to call from any thread.
  [[nodiscard]] Backend resolved_backend() const noexcept;

  // RGB8, BGR8, RGBA8, BGRA8 or NV12 (with its plane layouts), rows packed within each row, in
  // CUDA device memory, or in host memory, which costs a copy to the device (logged once). At
  // least 160 pixels on each side.
  holoscan::Input<holoscan::schema::ImageT> image;
  holoscan::Output<schema::EncodedVideoFrameT> bitstream;

 private:
  struct State;
  VideoEncoderConfig config_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::codec
