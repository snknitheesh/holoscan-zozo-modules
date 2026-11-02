// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// VideoDecoderOp: H.264 or H.265 access units in (as VideoEncoderOp emits them), decoded images
// out in CUDA device memory, decoded by the hardware (NVDEC through the Video Codec SDK, or the
// Jetson's V4L2 decoder).

#pragma once

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

struct VideoDecoderConfig {
  Backend backend{Backend::kAuto};
  // RGB8, BGR8, RGBA8, BGRA8, or NV12 (as one byte tensor with its two plane layouts).
  PixelFormat output_format{PixelFormat::kRgb8};
  // Largest image the output is sized for.
  int max_width{1920};
  int max_height{1080};
  // Each image out from the compute() of its own access unit, for streams with one reference
  // frame and no reordering, as VideoEncoderOp makes them in its low-latency mode. Turn it off
  // for other streams, whose images then come some access units later. See VideoDecoder.
  bool low_latency{true};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const VideoDecoderConfig& config);

class VideoDecoderOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit VideoDecoderOp(VideoDecoderConfig config);
  ~VideoDecoderOp() override;
  VideoDecoderOp(const VideoDecoderOp&) = delete;
  VideoDecoderOp& operator=(const VideoDecoderOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The backend in use once the first access unit arrived, kAuto before. Safe to call from any
  // thread.
  [[nodiscard]] Backend resolved_backend() const noexcept;

  // One access unit per sample. The decoder is built for the first one's codec, and built again
  // when the codec or the frame size changes. Access units before the first keyframe, and empty
  // ones, give no image.
  holoscan::Input<schema::EncodedVideoFrameT> bitstream;
  // One image per decoded frame, of the access unit's width and height, with its header and its
  // pts as header.capture_timestamp_ns.
  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  struct State;
  VideoDecoderConfig config_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::codec
