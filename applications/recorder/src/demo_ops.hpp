// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's own operators: a source of synthetic streams, each a known byte
// pattern stamped with a known capture time and frame id; a sink that checks
// every replayed sample against them, byte for byte; and a sink that counts.

#pragma once

#include <atomic>
#include <chrono>
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
#include <holoscan/recorder/format.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace recorder_app {

namespace schema = holoscan::schema;

// One synthetic stream: an image of `width` x `height` in `encoding`, or a tensor.
struct Stream {
  std::string name;
  holoscan::recorder::StreamKind kind{holoscan::recorder::StreamKind::kImage};
  schema::ImageEncoding encoding{schema::ImageEncoding_CUSTOM};
  DLDataType dtype{kDLUInt, 8U, 1U};
  std::vector<std::int64_t> shape;
  std::vector<holoscan::recorder::PlaneLayout> planes;
  bool device{};
  std::int32_t width{};
  std::int32_t height{};

  [[nodiscard]] std::size_t bytes() const;
};

// The ci mode's streams, at `width` x `height`: RGB8 and MONO16 from the GPU,
// MONO8 and NV12 from the host, and a float32 [6] tensor.
[[nodiscard]] std::vector<Stream> ci_streams(int width, int height);
// One image stream for the benchmark: "rgb8" or "mono8", in host or device memory.
[[nodiscard]] Stream bench_stream(const std::string& encoding, int width, int height, bool device);

// Byte `i` of frame `frame` of stream number `stream`.
[[nodiscard]] std::uint8_t pattern(std::size_t stream, std::uint64_t frame, std::size_t i);
// What frame `frame` is stamped with.
[[nodiscard]] std::int64_t capture_ns(std::uint64_t frame);
[[nodiscard]] std::uint64_t frame_id_of(std::uint64_t frame);

struct SourceStats {
  // Frames sent on every stream, and frames dropped because a consumer refused one.
  std::atomic<std::uint64_t> sent{0};
  std::atomic<std::uint64_t> dropped{0};
  std::atomic<std::int64_t> first_ns{0};
  std::atomic<std::int64_t> last_ns{0};
};

struct SourceConfig {
  std::vector<Stream> streams;
  std::uint64_t frames{};
  std::chrono::nanoseconds period{std::chrono::milliseconds{5}};
  // Live: a frame a consumer refuses is dropped, and the source makes exactly
  // `frames` compute calls. Otherwise it sends the rest of the frame on the next
  // tick, with up to `frames` * 20 calls in all.
  bool live{};
  // Make each stream's pattern once, and stamp only the frame number into its
  // first bytes: for the benchmark, whose frames nothing checks, so that making
  // them does not limit its rate.
  bool reuse_pattern{};
  std::shared_ptr<SourceStats> stats;
};

class SyntheticSourceOp final : public holoscan::Operator<> {
 public:
  explicit SyntheticSourceOp(SourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  [[nodiscard]] const holoscan::Output<schema::ImageT>& image(const std::string& name) const;
  [[nodiscard]] const holoscan::Output<holoscan::Tensor>& tensor(const std::string& name) const;

 private:
  [[nodiscard]] std::size_t index(const std::string& name) const;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> send(std::size_t stream,
                                                               holoscan::ExecutionContext& context);

  SourceConfig config_;
  std::vector<std::unique_ptr<holoscan::Output<schema::ImageT>>> images_;
  std::vector<std::unique_ptr<holoscan::Output<holoscan::Tensor>>> tensors_;
  std::vector<std::size_t> pending_;
  std::uint64_t frame_{};
  std::vector<std::vector<std::uint8_t>> scratch_;
};

struct VerifyStats {
  std::atomic<std::uint64_t> samples{0};
  // Samples whose bytes, shape, element type, capture time, frame id or image
  // descriptor differ from what the source sent; and samples out of order.
  std::atomic<std::uint64_t> mismatches{0};
};

struct VerifyConfig {
  std::vector<Stream> streams;
  std::uint64_t frames{};
  std::shared_ptr<VerifyStats> stats;
};

// Checks every sample of every stream against the source's pattern, and ends
// the run when it has seen every frame of every stream.
class VerifySinkOp final : public holoscan::Operator<> {
 public:
  explicit VerifySinkOp(VerifyConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  [[nodiscard]] const holoscan::Input<schema::ImageT>& image(const std::string& name) const;
  [[nodiscard]] const holoscan::Input<holoscan::Tensor>& tensor(const std::string& name) const;

 private:
  [[nodiscard]] std::size_t index(const std::string& name) const;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> check(
      std::size_t stream, const holoscan::Tensor& tensor, const schema::ImageT* image,
      const holoscan::SampleMetadata& metadata, holoscan::ExecutionContext& context);

  VerifyConfig config_;
  std::vector<std::unique_ptr<holoscan::Input<schema::ImageT>>> images_;
  std::vector<std::unique_ptr<holoscan::Input<holoscan::Tensor>>> tensors_;
  std::vector<std::uint64_t> next_;
  std::vector<std::uint8_t> host_;
};

struct CountStats {
  std::atomic<std::uint64_t> frames{0};
  std::atomic<std::int64_t> first_capture_ns{0};
  std::atomic<std::int64_t> last_capture_ns{0};
};

// Counts the images it gets, and ends the run after `count` of them (0: never).
class CountSinkOp final : public holoscan::Operator<> {
 public:
  CountSinkOp(std::uint64_t count, std::shared_ptr<CountStats> stats);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<schema::ImageT> image;

 private:
  std::uint64_t count_;
  std::shared_ptr<CountStats> stats_;
};

}  // namespace recorder_app
