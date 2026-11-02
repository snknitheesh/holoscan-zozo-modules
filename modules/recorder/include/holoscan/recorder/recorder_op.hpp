// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// RecorderOp: records named streams of a graph to disk, each sample with its
// capture time, for ReplayerOp to play back. A stream is images
// (schema::ImageT) or bare tensors, in host or device memory, of any pixel
// format, element type and shape.
//
// It never holds the graph back. compute() only copies each sample into pinned
// host memory (device data asynchronously, on the operator's CUDA stream) and
// queues it; a writer thread of its own writes the queue to disk. When the disk
// falls behind and the queue is full, a sample is dropped and counted, and a
// warning says so at most once a second; compute() never waits for the disk.
// Connect a live source with QueuePolicy::kDropOldest edges, so that the
// recorder, too, never holds back its producers; a lossless (kBlocking) edge
// records every sample the queue has room for.
//
// The recording is opened in start() and finished in stop(), which writes what
// is still queued. A write that fails (a full disk) fails the run at the next
// compute().

#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
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

namespace holoscan::recorder {

// One input of the recorder, and the stream it records.
struct RecorderStream {
  std::string name;
  StreamKind kind{StreamKind::kImage};
};

// What the recorder has done, for the application to read from any thread. The
// runtime may construct an operator more than once, so read these, not the
// operator you constructed.
struct RecorderStats {
  // Samples the recorder was given.
  std::atomic<std::uint64_t> received{0};
  // Samples written to disk, and their data bytes.
  std::atomic<std::uint64_t> recorded{0};
  std::atomic<std::uint64_t> bytes{0};
  // Samples dropped because the writer's queue was full: the disk fell behind.
  std::atomic<std::uint64_t> dropped{0};
  // The most samples that were waiting for the disk at once.
  std::atomic<std::uint32_t> max_queued{0};
};

struct RecorderConfig {
  // The recording's folder: new, empty, or (with `overwrite`) a recording.
  std::string directory;
  // The inputs, one per stream; at least one.
  std::vector<RecorderStream> streams;
  bool overwrite{false};
  // Calibration blobs, each a JSON value, by name, stored in recording.json.
  std::map<std::string, std::string> calibration;
  // Free-form text, stored in recording.json.
  std::map<std::string, std::string> metadata;
  // The writer's queue: at most this many samples, in at most this much pinned
  // host memory, wait for the disk. A sample larger than `queue_bytes` is
  // always dropped.
  std::uint32_t queue_samples{32};
  std::uint64_t queue_bytes{std::uint64_t{512} << 20U};
  // Each input's queue depth, for an edge that does not set one.
  std::uint32_t input_queue_depth{4};
  // Writes no faster than this, in bytes per second; 0 for as fast as the disk
  // goes. For a disk shared with other work, or to play a slow one in a test.
  std::uint64_t max_write_bytes_per_second{0};
  // Counts what was recorded and dropped, when given.
  std::shared_ptr<RecorderStats> stats;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const RecorderConfig& config);

class RecorderOp final : public holoscan::Operator<> {
 public:
  // Checks `config`; opens nothing: the runtime may construct an operator more
  // than once. @throws std::invalid_argument if `config` is not valid.
  explicit RecorderOp(RecorderConfig config);
  ~RecorderOp() override;
  RecorderOp(const RecorderOp&) = delete;
  RecorderOp& operator=(const RecorderOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  // Creates the recording and starts the writer.
  // @throws holoscan::RuntimeError if the recording cannot be created.
  void start() override;
  // Writes what is queued, finishes the recording and stops the writer.
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The input of image stream `name`. @throws std::out_of_range if there is none.
  [[nodiscard]] const holoscan::Input<holoscan::schema::ImageT>& image(std::string_view name) const;
  // The input of tensor stream `name`. @throws std::out_of_range if there is none.
  [[nodiscard]] const holoscan::Input<holoscan::Tensor>& tensor(std::string_view name) const;

 private:
  class Writer;

  // Copies one sample into the writer's queue, or drops it.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> enqueue(
      std::uint32_t stream, const holoscan::Tensor* tensor, const holoscan::schema::ImageT* image,
      const holoscan::SampleMetadata& metadata, holoscan::ExecutionContext& context);
  void count_drop(std::uint32_t stream);

  RecorderConfig config_;
  // One port per stream, in the order of config_.streams; an image stream's
  // tensor port and a tensor stream's image port are never declared.
  std::vector<std::unique_ptr<holoscan::Input<holoscan::schema::ImageT>>> images_;
  std::vector<std::unique_ptr<holoscan::Input<holoscan::Tensor>>> tensors_;
  std::unique_ptr<Writer> writer_;
  // Per stream: samples seen and dropped.
  std::vector<std::uint64_t> sequence_;
  std::vector<std::uint64_t> dropped_;
  // Drops since the last warning, and when it was.
  std::uint64_t unreported_drops_{};
  std::int64_t last_warning_ns_{};
  bool warned_{};
};

}  // namespace holoscan::recorder
