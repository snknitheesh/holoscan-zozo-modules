// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// RecorderOp on its own: it never holds back a producer, and counts what it
// drops when the disk cannot keep up; it records tensors with padded rows
// packed; and it refuses configurations and folders it cannot use.

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/recorder/format.hpp>
#include <holoscan/recorder/recorder_op.hpp>

#include "graph_support.hpp"
#include "test_support.hpp"

namespace holoscan::recorder::test {
namespace {

const holoscan::ConnectionOptions kLossless{.queue_policy = holoscan::QueuePolicy::kBlocking};

// A disk slower than the source: 512 KiB frames every 5 ms (100 MB/s) to a
// writer held to 10 MB/s with room for four frames. The source is never
// refused, so the graph never waits; most frames are dropped, and counted; and
// every frame that was recorded is whole, with gaps in the sequence numbers
// where frames were dropped.
void slow_disk_drops() {
  constexpr std::uint64_t kFrames = 40;
  const std::vector<Stream> streams{Stream{.name = "camera",
                                           .encoding = schema::ImageEncoding_MONO8,
                                           .dtype = kU8,
                                           .shape = {512, 1024}}};
  const std::filesystem::path folder = fresh_folder("slow_disk");
  holoscan::Graph graph{"slow_disk"};
  auto sent = std::make_shared<std::atomic<std::uint64_t>>(0);
  auto refused = std::make_shared<std::atomic<std::uint64_t>>(0);
  auto source =
      graph.op<PatternSourceOp>("source", SourceConfig{.streams = streams,
                                                       .frames = kFrames,
                                                       .period = std::chrono::milliseconds{5},
                                                       .calls = kFrames,
                                                       .live = true,
                                                       .sent = sent,
                                                       .refused = refused});
  auto stats = std::make_shared<RecorderStats>();
  auto recorder =
      graph.op<RecorderOp>("recorder", RecorderConfig{.directory = folder.string(),
                                                      .streams = {{"camera", StreamKind::kImage}},
                                                      .queue_samples = 4,
                                                      .max_write_bytes_per_second = 10'000'000,
                                                      .stats = stats});
  // A lossless edge, so that a recorder that held the source back would show
  // as a refused frame rather than as a frame the edge dropped.
  graph.add_flow(source->image("camera"), recorder->image("camera"), kLossless);
  const RunOutcome outcome = run_graph(graph);
  VERIFY_MSG(outcome.compiled && outcome.completed, outcome.diagnostics);
  VERIFY(*sent == kFrames);
  VERIFY_MSG(*refused == 0U, "the recorder refused " + std::to_string(refused->load()) +
                                 " frame(s): it held the source back");
  VERIFY(stats->received == kFrames);
  VERIFY_MSG(stats->dropped > kFrames / 2,
             "dropped " + std::to_string(stats->dropped.load()) + " of " + std::to_string(kFrames));
  VERIFY(stats->recorded + stats->dropped == kFrames);
  VERIFY(stats->max_queued <= 4U);

  const RecordingReader reader(folder);
  VERIFY(reader.complete());
  VERIFY(reader.records().size() == stats->recorded);
  VERIFY(reader.streams()[0].dropped == stats->dropped);
  bool gap = false;
  std::uint64_t previous = 0;
  for (const Record& record : reader.records()) {
    if (&record != &reader.records().front() && record.sequence != previous + 1) gap = true;
    previous = record.sequence;
    // The sequence number is the frame's: the source's frame `sequence`.
    VERIFY(record.timestamp_ns == capture_ns(record.sequence));
    std::vector<std::byte> data(streams[0].bytes());
    reader.read(record, data);
    VERIFY_MSG(std::memcmp(data.data(), pattern_bytes(0, record.sequence, data.size()).data(),
                           data.size()) == 0,
               "frame " + std::to_string(record.sequence) + ": data");
  }
  VERIFY_MSG(gap, "the dropped frames leave gaps in the sequence numbers");
}

// Sends one 10x16 uint8 tensor whose rows are 32 bytes apart, in host or
// device memory.
class PaddedSourceOp final : public holoscan::Operator<> {
 public:
  static constexpr std::int64_t kRows = 10;
  static constexpr std::int64_t kColumns = 16;
  static constexpr std::int64_t kPitch = 32;

  explicit PaddedSourceOp(bool device) : device_(device) {}

  void setup(holoscan::OperatorSpec& spec) override {
    static constexpr std::array<std::int64_t, 2> kShape{kRows, kColumns};
    static constexpr std::array<std::uint64_t, 2> kStrides{kPitch, 1};
    spec.output(tensor, "tensor")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind =
                        device_ ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost,
                    .dtype = kU8,
                    .rank = 2U,
                    .strides = holoscan::TensorStridePolicy::kBoundedStrided,
                },
            .bounds = holoscan::tensor_bounds(
                kShape, holoscan::TensorBoundsOptions{.max_byte_span = kRows * kPitch,
                                                      .max_abs_byte_strides = kStrides}),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{5}})
        .complete_after(holoscan::ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::array<std::int64_t, 2> shape{kRows, kColumns};
    const std::array<std::int64_t, 2> strides{kPitch, 1};
    auto loan = tensor.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = shape, .dtype = kU8, .byte_strides = holoscan::TensorByteStridesSpan{strides}});
    if (!loan) return holoscan::make_unexpected(std::move(loan).error());
    // Each row's pixels, then padding of 0xEE.
    std::vector<std::uint8_t> bytes(kRows * kPitch, 0xEEU);
    for (std::int64_t r = 0; r < kRows; ++r) {
      for (std::int64_t c = 0; c < kColumns; ++c) {
        bytes[r * kPitch + c] = static_cast<std::uint8_t>(r * kColumns + c);
      }
    }
    holoscan::expected<void, holoscan::Error> written;
    if (device_) {
      written = loan->write_and_commit(
          context.cuda_stream(),
          [&](const holoscan::TensorOutputWriteGuard& writer)
              -> holoscan::expected<void, holoscan::Error> {
            if (cudaMemcpyAsync(writer.data(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice,
                                context.cuda_stream().get()) != cudaSuccess) {
              return holoscan::make_unexpected(
                  holoscan::Error{holoscan::ErrorCode::kFailure, "cudaMemcpyAsync"});
            }
            return {};
          });
    } else {
      written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer) {
        std::memcpy(writer.data(), bytes.data(), bytes.size());
      });
    }
    if (!written) return written;
    return tensor.emit(std::move(*loan),
                       holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<holoscan::Tensor> tensor;

 private:
  bool device_;
};

void padded_rows(bool device) {
  const std::filesystem::path folder = fresh_folder(device ? "padded_device" : "padded_host");
  holoscan::Graph graph{"padded"};
  auto source = graph.op<PaddedSourceOp>("source", device);
  auto stats = std::make_shared<RecorderStats>();
  auto recorder =
      graph.op<RecorderOp>("recorder", RecorderConfig{.directory = folder.string(),
                                                      .streams = {{"tensor", StreamKind::kTensor}},
                                                      .stats = stats});
  graph.add_flow(source->tensor, recorder->tensor("tensor"), kLossless);
  holoscan::CompileOptions options;
  if (device) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "source", .output_port = "tensor", .device = holoscan::DeviceId{0}});
  }
  const RunOutcome outcome = run_graph(graph, std::move(options));
  VERIFY_MSG(outcome.compiled && outcome.completed, outcome.diagnostics);
  VERIFY(stats->recorded == 1U);
  const RecordingReader reader(folder);
  VERIFY(reader.records().size() == 1U);
  if (reader.records().size() != 1U) return;
  const Record& record = reader.records().front();
  VERIFY(
      (record.shape == std::vector<std::int64_t>{PaddedSourceOp::kRows, PaddedSourceOp::kColumns}));
  std::vector<std::byte> data(*record.bytes());
  reader.read(record, data);
  for (std::size_t i = 0; i < data.size(); ++i) {
    VERIFY_MSG(data[i] == static_cast<std::byte>(i), "byte " + std::to_string(i));
  }
}

void rejects_configs() {
  const auto rejected = [](RecorderConfig config, const std::string& part) {
    const std::string message = [&]() -> std::string {
      try {
        validate_config(config);
      } catch (const std::invalid_argument& error) {
        return error.what();
      }
      return "";
    }();
    VERIFY_MSG(message.find(part) != std::string::npos,
               "expected \"" + part + "\", got \"" + message + "\"");
  };
  const RecorderConfig good{.directory = "x", .streams = {{"a", StreamKind::kImage}}};
  validate_config(good);
  RecorderConfig config = good;
  config.directory.clear();
  rejected(config, "directory");
  config = good;
  config.streams.clear();
  rejected(config, "at least one stream");
  config = good;
  config.streams = {{"a/b", StreamKind::kImage}};
  rejected(config, "stream name \"a/b\"");
  config.streams = {{".hidden", StreamKind::kImage}};
  rejected(config, "stream name \".hidden\"");
  config.streams = {{"a", StreamKind::kImage}, {"a", StreamKind::kTensor}};
  rejected(config, "two streams are named \"a\"");
  config = good;
  config.calibration = {{"camera", "{not json"}};
  rejected(config, "calibration \"camera\" is not JSON");
  config = good;
  config.queue_samples = 0;
  rejected(config, "queue_samples");
  config = good;
  config.input_queue_depth = 0;
  rejected(config, "input_queue_depth");
  VERIFY(throws<std::invalid_argument>([] { RecorderOp op(RecorderConfig{}); }));
  RecorderOp op(good);
  VERIFY(throws<std::out_of_range>([&] { (void)op.tensor("a"); }));
  VERIFY(throws<std::out_of_range>([&] { (void)op.image("b"); }));
}

// A folder that holds anything but a recording is never written into, and an
// existing recording only with `overwrite`: the run fails before it starts.
void refuses_folders() {
  const std::filesystem::path foreign = fresh_folder("foreign");
  std::filesystem::create_directories(foreign);
  std::ofstream(foreign / "notes.txt") << "mine\n";
  const auto run_into = [](const std::filesystem::path& folder, bool overwrite) {
    holoscan::Graph graph{"refuses"};
    const std::vector<Stream> streams{
        Stream{.name = "a", .kind = StreamKind::kTensor, .dtype = kU8, .shape = {4}}};
    auto source = graph.op<PatternSourceOp>(
        "source", SourceConfig{.streams = streams, .frames = 2, .calls = 20});
    auto recorder =
        graph.op<RecorderOp>("recorder", RecorderConfig{.directory = folder.string(),
                                                        .streams = {{"a", StreamKind::kTensor}},
                                                        .overwrite = overwrite});
    graph.add_flow(source->tensor("a"), recorder->tensor("a"), kLossless);
    return run_graph(graph);
  };
  RunOutcome outcome = run_into(foreign, true);
  VERIFY(outcome.compiled && !outcome.completed);
  VERIFY_MSG(outcome.diagnostics.find("not part of a recording") != std::string::npos,
             outcome.diagnostics);
  VERIFY(std::filesystem::exists(foreign / "notes.txt"));

  const std::filesystem::path existing = fresh_folder("existing");
  outcome = run_into(existing, false);
  VERIFY_MSG(outcome.completed, outcome.diagnostics);
  outcome = run_into(existing, false);
  VERIFY(!outcome.completed);
  VERIFY_MSG(outcome.diagnostics.find("already holds a recording") != std::string::npos,
             outcome.diagnostics);
  outcome = run_into(existing, true);
  VERIFY_MSG(outcome.completed, outcome.diagnostics);
  VERIFY(RecordingReader(existing).records().size() == 2U);
}

}  // namespace
}  // namespace holoscan::recorder::test

int main() {
  using holoscan::recorder::test::TestCase;
  namespace t = holoscan::recorder::test;
  return t::run({
      TestCase{"a slow disk drops and counts samples, never holding the source back",
               [] { t::slow_disk_drops(); }},
      TestCase{"padded host rows are recorded packed", [] { t::padded_rows(false); }},
      TestCase{"padded device rows are recorded packed", [] { t::padded_rows(true); }},
      TestCase{"rejects unusable configurations", [] { t::rejects_configs(); }},
      TestCase{"refuses foreign folders and existing recordings", [] { t::refuses_folders(); }},
  });
}
