// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's checks: CodecReportOp pairs each decoded image with its access unit and measures it
// against the frame the source sent (PSNR), the bitrate and the order; RenderCheckOp checks the
// frames HoloViz renders offscreen.

#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/codec/codec_types.hpp>
#include <holoscan/codec/encoded_video_frame_schema_traits.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace codec_app {

// What the run measured, read by main() once it is over, and the latest frame meanwhile.
struct CodecReport {
  // Frames reported: the source's count of frames done.
  std::atomic<std::uint32_t> frames{0};
  // The index of the latest frame reported.
  std::atomic<std::uint32_t> latest{0};
  std::mutex mutex;  // Guards the fields below.
  double psnr_sum{};
  double psnr_min{1e9};
  std::uint64_t bytes{};
  std::uint64_t keyframes{};
  // Problems seen, each one line; the run fails when there is any.
  std::vector<std::string> failures;
  // Frames HoloViz rendered offscreen and RenderCheckOp checked, and those whose two halves
  // matched frames the source sent.
  std::uint32_t rendered{};
  std::uint32_t rendered_matched{};

  void fail(std::string message) {
    const std::scoped_lock lock(mutex);
    if (failures.size() < 20) failures.push_back(std::move(message));
  }
};

struct CodecReportConfig {
  std::shared_ptr<CodecReport> report;
  int width{};
  int height{};
  holoscan::codec::Codec codec{holoscan::codec::Codec::kH264};
  std::string backend;
  // Frames the run expects; 0 runs until stopped. When reached, the run completes.
  std::uint32_t expected_frames{};
  // Each frame's PSNR must reach this, in dB; 0 checks nothing.
  double min_psnr{};
  // Every frame must come, in order; otherwise frames may be skipped (real time).
  bool lossless{true};
  // Declare the `stats` output: a HoloViz text layer.
  bool text{false};
};

class CodecReportOp final : public holoscan::Operator<> {
 public:
  explicit CodecReportOp(CodecReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::codec::schema::EncodedVideoFrameT> bitstream;
  // RGB8 in device memory, as VideoDecoderOp gives it.
  holoscan::Input<holoscan::schema::ImageT> decoded;
  // Text runs for a HoloViz text layer in normalized coordinates: labels and the bitrate.
  holoscan::Output<holoscan::Tensor> stats;

 private:
  CodecReportConfig config_;
  std::uint32_t seen_{};
  std::uint64_t last_sequence_{};
  // Bytes of the access units of the last second, by capture time.
  std::deque<std::pair<std::int64_t, std::uint64_t>> recent_;
  std::vector<std::uint8_t> host_;
};

struct RenderCheckConfig {
  std::shared_ptr<CodecReport> report;
  // The size of one side: the rendered frame is twice as wide.
  int width{};
  int height{};
  // The decoded half must reach this PSNR, in dB, against a frame the source sent.
  double min_psnr{};
  // When set, the last rendered frame is written here as a binary PPM.
  std::filesystem::path image_file;
};

// Takes the frames HoloViz renders into device memory: the original on the left, the decoded image
// on the right. Each half must match one of the frames the source sent around the latest one
// reported (HoloViz shows the newest original, which runs a few frames ahead of the decoded one).
class RenderCheckOp final : public holoscan::Operator<> {
 public:
  explicit RenderCheckOp(RenderCheckConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // RGBA8 [height, 2 x width, 4] in device memory.
  holoscan::Input<holoscan::schema::ImageT> rendered;

 private:
  RenderCheckConfig config_;
  std::vector<std::uint8_t> host_;
};

}  // namespace codec_app
