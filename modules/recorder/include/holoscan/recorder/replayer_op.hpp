// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ReplayerOp: plays a recording back into a graph. Each recorded stream is an
// output of the same name, which sends its samples with their recorded capture
// times, frame ids and image descriptors, byte for byte, in host or device
// memory. A stereo camera recording plays the same way, with outputs left,
// right and, when recorded, rgb and depth.
//
// Samples go out in the order of their capture times, at most one per output
// each period, either as fast as consumers take them or when their recorded
// time comes. A sample a consumer refuses on a lossless (kBlocking) edge is sent
// again on the next tick, so nothing is lost; on kDropOldest edges nothing is
// refused. When the recording ends the operator idles, or starts over.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/memory_kind.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/recorder/format.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::recorder {

enum class ReplayPacing : std::uint8_t {
  // Every sample in turn, as fast as the consumers take them: for tests.
  kAsFastAsPossible,
  // Each sample when its time comes: its capture time after the first one's,
  // from the first compute().
  kRealtime,
};

// Where the outputs' tensors are.
enum class ReplayMemory : std::uint8_t {
  kHost,    // Host memory.
  kDevice,  // CUDA device memory, copied on the operator's stream. Bind the
            // outputs' device with bind_device().
};

// What the replayer has sent, for the application to read from any thread.
struct ReplayerStats {
  std::atomic<std::uint64_t> emitted{0};
  // Passes completed over the recording.
  std::atomic<std::uint64_t> passes{0};
  // Whether the recording has ended and the replayer idles.
  std::atomic<bool> finished{false};
  // With kRealtime: how late a sample went out at worst, after its time.
  std::atomic<std::int64_t> max_lateness_ns{0};
};

struct ReplayerConfig {
  // A recording's folder, or a stereo camera recording's.
  std::string directory;
  ReplayPacing pacing{ReplayPacing::kAsFastAsPossible};
  // Start over at the end. Each pass after the first adds the recording's length
  // (plus one typical interval) to the capture times, and its largest frame id
  // to the frame ids, so both keep increasing.
  bool loop{false};
  // The samples to play, by their sequence number on each stream (a stereo
  // camera recording's frame number): first_frame <= sequence < end_frame.
  std::uint64_t first_frame{0};
  std::uint64_t end_frame{std::numeric_limits<std::uint64_t>::max()};
  // The streams to play, by name; empty for all of them.
  std::vector<std::string> streams;
  ReplayMemory memory{ReplayMemory::kHost};
  // How often the operator runs: it sends at most one sample per output each
  // period, so this bounds the rate, and with kRealtime the timing's grain.
  std::chrono::nanoseconds period{std::chrono::milliseconds{1}};
  std::shared_ptr<ReplayerStats> stats;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const ReplayerConfig& config);

class ReplayerOp final : public holoscan::Operator<> {
 public:
  // Opens and checks the recording, which gives the outputs.
  // @throws std::invalid_argument if `config` is not valid or names a stream the
  // recording does not have; RecordingError if the recording is not readable.
  explicit ReplayerOp(ReplayerConfig config);
  ~ReplayerOp() override;
  ReplayerOp(const ReplayerOp&) = delete;
  ReplayerOp& operator=(const ReplayerOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The output of image stream `name`. @throws std::out_of_range if there is none.
  [[nodiscard]] const holoscan::Output<holoscan::schema::ImageT>& image(
      std::string_view name) const;
  // The output of tensor stream `name`. @throws std::out_of_range if there is none.
  [[nodiscard]] const holoscan::Output<holoscan::Tensor>& tensor(std::string_view name) const;

  // The streams played, in the recording's order: each one's output has its name.
  [[nodiscard]] std::vector<std::string> stream_names() const;
  [[nodiscard]] const RecordingReader& recording() const noexcept { return *reader_; }
  // Samples one pass plays.
  [[nodiscard]] std::size_t samples() const noexcept { return order_.size(); }

  // With ReplayMemory::kDevice, places every output on `device` in `options`, for
  // the operator added as `operator_path`. Does nothing for host memory.
  void bind_device(holoscan::CompileOptions& options, const std::string& operator_path,
                   holoscan::DeviceId device = holoscan::DeviceId{0}) const;

 private:
  struct Port;
  struct Staging;

  [[nodiscard]] holoscan::expected<void, holoscan::Error> send(const Record& record,
                                                               holoscan::ExecutionContext& context);

  ReplayerConfig config_;
  std::unique_ptr<RecordingReader> reader_;
  // One output per stream played, in the recording's order.
  std::vector<std::unique_ptr<Port>> outputs_;
  // Each recorded stream's output, if it is played.
  std::vector<std::optional<std::size_t>> output_of_stream_;
  // The records played, as indices into the recording's, in capture-time order.
  std::vector<std::size_t> order_;
  std::int64_t pass_span_ns_{};
  std::uint64_t pass_frames_{};
  // Playback state, reset at start().
  std::size_t position_{};
  std::uint64_t pass_{};
  std::optional<std::int64_t> start_ns_;
  bool finished_{};
  std::unique_ptr<Staging> staging_;
};

}  // namespace holoscan::recorder
