// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StereoRectifyOp in a running graph, between a source and a consumer that
// reads each rectified pair on the GPU as soon as it arrives: on its own CUDA
// stream, ordered only by the pair's readiness, with no wait on the host
// before. The dots move from pair to pair, so a remap that had not finished
// when the operator sent its output, or an output that still held an earlier
// pair, misplaces them. The reader takes each pair's camera info as a
// consumer would (messages::camera_info_contract(), kEveryFrame), and checks
// it. For every backend this platform runs, from device, pinned host and
// pageable host memory; each also counts the host waits and host-to-device
// copies the operator makes per pair (remap_statistics()).

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/rectify/remapper.hpp>
#include <holoscan/rectify/stereo_rectify_op.hpp>
#include <holoscan/rectify/synthetic_stereo.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "test_support.hpp"

namespace holoscan::rectify {
namespace {

using std::chrono_literals::operator""ms;

constexpr int kWidth = 1280;  // The PVA takes widths that are multiples of 256.
constexpr int kHeight = 720;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

enum class Memory { kDevice, kPinned, kPageable };

const char* memory_name(Memory memory) {
  switch (memory) {
    case Memory::kDevice:
      return "device";
    case Memory::kPinned:
      return "pinned host";
    case Memory::kPageable:
      return "pageable host";
  }
  return "?";
}

holoscan::MemoryKind memory_kind(Memory memory) {
  switch (memory) {
    case Memory::kDevice:
      return holoscan::MemoryKind::kCudaDevice;
    case Memory::kPinned:
      return holoscan::MemoryKind::kPinnedHost;
    case Memory::kPageable:
      return holoscan::MemoryKind::kHost;
  }
  return holoscan::MemoryKind::kUnknown;
}

// The synthetic pair's camera_info files and frames, made once for every run.
struct Frames {
  SyntheticStereo scene = synthetic_stereo(kWidth, kHeight);
  test::TempDir dir;
  std::filesystem::path left_file = dir / "left.yaml";
  std::filesystem::path right_file = dir / "right.yaml";
  StereoRectification rectification;
  std::vector<std::vector<std::uint8_t>> left;
  std::vector<std::vector<std::uint8_t>> right;

  explicit Frames(std::uint32_t count) {
    messages::write_stereo_camera_info_yaml(scene.calibration, left_file, right_file);
    rectification =
        compute_stereo_rectification(messages::read_stereo_camera_info_yaml(left_file, right_file));
    const SyntheticView left_camera = left_view(scene);
    const SyntheticView right_camera = right_view(scene);
    for (std::uint32_t f = 0; f < count; ++f) {
      left.push_back(left_camera.render(f));
      right.push_back(right_camera.render(f));
    }
  }
};

// What the consumer saw, shared with the source for flow control.
struct Outcome {
  std::atomic<std::uint32_t> consumed{0};
  std::mutex mutex;
  std::vector<std::string> problems;
  std::uint32_t pairs{};
  double max_row_error{};
  double max_position_error{};

  void problem(std::string what) {
    const std::lock_guard lock(mutex);
    problems.push_back(std::move(what));
  }
};

// Sends `frames` pairs in `memory`, the next only once the consumer has the
// previous, so that the lossless edges never refuse one.
// What the source and the reader share. (Graph::op() takes no raw pointers.)
struct RunSetup {
  std::shared_ptr<const Frames> frames;
  std::uint32_t count{};
  Memory memory{Memory::kDevice};
  std::shared_ptr<Outcome> outcome;
};

class PairSourceOp final : public holoscan::Operator<> {
 public:
  explicit PairSourceOp(RunSetup setup)
      : frames_(std::move(setup.frames)),
        count_(setup.count),
        memory_(setup.memory),
        outcome_(std::move(setup.outcome)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    for (auto [port, name] : {std::pair{&left, "left"}, std::pair{&right, "right"}}) {
      spec.output(*port, name)
          .produces_tensor(holoscan::TensorOutputSpec{
              .representation =
                  holoscan::TensorRepresentation{
                      .memory_kind = memory_kind(memory_),
                      .dtype = kUInt8,
                      .rank = 2U,
                      .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                  },
              .bounds = holoscan::tensor_bounds(std::uint64_t{kWidth} * kHeight),
              .storage = holoscan::TensorOutputStorage::kRuntimePool,
          })
          .payload_options(holoscan::PayloadOptions{
              .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
          });
    }
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 2ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (sent_ >= count_ || outcome_->consumed.load() < sent_) return {};
    const holoscan::EmitOptions emit{.capture_time = context.activation_time(),
                                     .frame_id = std::uint64_t{sent_} + 1U};
    for (auto [port, pixels] :
         {std::pair{&left, &frames_->left[sent_]}, std::pair{&right, &frames_->right[sent_]}}) {
      if (auto sent = send(*port, *pixels, context, emit); !sent) {
        return sent;
      }
    }
    ++sent_;
    return {};
  }

  holoscan::Output<holoscan::schema::ImageT> left;
  holoscan::Output<holoscan::schema::ImageT> right;

 private:
  holoscan::expected<void, holoscan::Error> send(holoscan::Output<holoscan::schema::ImageT>& port,
                                                 const std::vector<std::uint8_t>& pixels,
                                                 holoscan::ExecutionContext& context,
                                                 const holoscan::EmitOptions& emit) {
    const std::array<std::int64_t, 2> shape{kHeight, kWidth};
    auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) return holoscan::make_unexpected(std::move(loan).error());
    if (memory_ == Memory::kDevice) {
      auto writer = loan->write(context.cuda_stream());
      if (!writer) return holoscan::make_unexpected(std::move(writer).error());
      if (const cudaError_t copied =
              cudaMemcpyAsync(writer->data(), pixels.data(), pixels.size(), cudaMemcpyHostToDevice,
                              context.cuda_stream().get());
          copied != cudaSuccess) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, cudaGetErrorString(copied)});
      }
      if (auto committed = std::move(*writer).commit(); !committed) {
        return holoscan::make_unexpected(std::move(committed).error());
      }
    } else {
      auto committed =
          loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                          -> holoscan::expected<void, holoscan::Error> {
            std::memcpy(writer.data(), pixels.data(), pixels.size());
            return {};
          });
      if (!committed) return holoscan::make_unexpected(std::move(committed).error());
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = kWidth;
    descriptor.height = kHeight;
    descriptor.encoding = holoscan::schema::ImageEncoding_MONO8;
    return port.emit_tensor(std::move(*loan), descriptor, emit);
  }

  std::shared_ptr<const Frames> frames_;
  std::uint32_t count_;
  Memory memory_;
  std::shared_ptr<Outcome> outcome_;
  std::uint32_t sent_{};
};

// Reads each rectified pair back on its own CUDA stream the moment it arrives
// and checks its dots against the geometry, and its camera info.
class GpuReaderOp final : public holoscan::Operator<> {
 public:
  explicit GpuReaderOp(RunSetup setup)
      : frames_(std::move(setup.frames)), count_(setup.count), outcome_(std::move(setup.outcome)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(rectified_left, "rectified_left").queue_depth(2U);
    spec.input(rectified_right, "rectified_right").queue_depth(2U);
    spec.input(camera_info, "camera_info").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    return messages::camera_info_contract(
        messages::CameraInfoInputOptions{.timing = messages::CameraInfoTiming::kEveryFrame},
        camera_info, rectified_left, rectified_right);
  }

  void start() override {
    const StereoRectification& r = frames_->rectification;
    bytes_ = static_cast<std::size_t>(r.width) * r.height;
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMallocHost(&left_, bytes_) != cudaSuccess ||
        cudaMallocHost(&right_, bytes_) != cudaSuccess) {
      outcome_->problem("the reader's stream or buffers could not be made");
    }
  }

  void stop() override {
    cudaFreeHost(left_);
    cudaFreeHost(right_);
    cudaStreamDestroy(stream_);
    left_ = right_ = nullptr;
    stream_ = nullptr;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto l = rectified_left.receive();
    if (!l) return holoscan::make_unexpected(std::move(l).error());
    auto r = rectified_right.receive();
    if (!r) return holoscan::make_unexpected(std::move(r).error());
    auto info = camera_info_.update(camera_info);
    if (!info) return holoscan::make_unexpected(std::move(info).error());
    const bool info_matches =
        *info != nullptr &&
        messages::same_calibration(**info, frames_->rectification.camera_info) &&
        (*info)->header.capture_timestamp_ns == l->metadata.capture_timestamp_ns;
    const auto frame = static_cast<std::uint32_t>(l->metadata.frame_id - 1U);
    if (!info_matches) outcome_->problem("pair " + std::to_string(frame) + ": camera info");
    // Both reads are queued before anything waits: only the readiness orders them.
    if (auto copied = l->data.data->copy_to_host_async(left_, bytes_, stream_); !copied) {
      return holoscan::make_unexpected(std::move(copied).error());
    }
    if (auto copied = r->data.data->copy_to_host_async(right_, bytes_, stream_); !copied) {
      return holoscan::make_unexpected(std::move(copied).error());
    }
    if (const cudaError_t status = cudaStreamSynchronize(stream_); status != cudaSuccess) {
      outcome_->problem(cudaGetErrorString(status));
    }
    const StereoRectification& geometry = frames_->rectification;
    const AlignmentReport report =
        check_rectified(frames_->scene, geometry, static_cast<const std::uint8_t*>(left_),
                        static_cast<const std::uint8_t*>(right_), geometry.width, frame);
    {
      const std::lock_guard lock(outcome_->mutex);
      ++outcome_->pairs;
      outcome_->max_row_error = std::max(outcome_->max_row_error, report.max_row_error);
      outcome_->max_position_error =
          std::max(outcome_->max_position_error, report.max_position_error);
      if (report.dots < 50 || report.missing != 0) {
        outcome_->problems.push_back("pair " + std::to_string(frame) + ": dots missing");
      }
    }
    if (outcome_->consumed.fetch_add(1) + 1 == count_) context.request_completion();
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> rectified_left;
  holoscan::Input<holoscan::schema::ImageT> rectified_right;
  holoscan::Input<messages::schema::StereoCameraInfoT> camera_info;

 private:
  std::shared_ptr<const Frames> frames_;
  std::uint32_t count_;
  std::shared_ptr<Outcome> outcome_;
  messages::LatestCameraInfo<messages::schema::StereoCameraInfoT> camera_info_;
  cudaStream_t stream_{};
  std::size_t bytes_{};
  void* left_{};
  void* right_{};
};

struct RunResult {
  bool ok{};
  RemapStatistics cost;  // Over the whole run.
};

RunResult run_pipeline(const std::shared_ptr<const Frames>& frames, Backend backend, Memory memory,
                       std::uint32_t count) {
  const std::string what = std::string(backend_name(backend)) + ", " + memory_name(memory) + ", " +
                           std::to_string(count) + " pairs";
  auto outcome = std::make_shared<Outcome>();
  holoscan::Graph graph{"rectify_pipeline"};
  const RunSetup setup{.frames = frames, .count = count, .memory = memory, .outcome = outcome};
  auto source = graph.op<PairSourceOp>("source", setup);
  auto rectify = graph.op<StereoRectifyOp>(
      "rectify", StereoRectifyConfig{.left_camera_info_path = frames->left_file,
                                     .right_camera_info_path = frames->right_file,
                                     .backend = backend});
  auto reader = graph.op<GpuReaderOp>("reader", setup);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->left, rectify->left, lossless);
  graph.add_flow(source->right, rectify->right, lossless);
  graph.add_flow(rectify->rectified_left, reader->rectified_left, lossless);
  graph.add_flow(rectify->rectified_right, reader->rectified_right, lossless);
  graph.add_flow(rectify->camera_info, reader->camera_info, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  for (const char* port : {"rectified_left", "rectified_right"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "rectify", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  if (memory == Memory::kDevice) {
    for (const char* port : {"left", "right"}) {
      options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = port, .device = holoscan::DeviceId{0}});
    }
  }
  const auto plan = holoscan::compile(graph, std::move(options));
  VERIFY_MSG(plan.ok(),
             what + ": " + holoscan::format_diagnostics(plan.diagnostics(), plan.graph()));
  if (!plan.ok()) return {};

  const RemapStatistics before = remap_statistics();
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  const bool timed_out = session.is_running();
  if (timed_out) session.request_stop();
  session.wait();
  const RemapStatistics after = remap_statistics();

  const bool completed = !timed_out && session.diagnostics().empty() &&
                         session.termination() == holoscan::RunTermination::kRequestedCompleted;
  VERIFY_MSG(completed, what + ": the run did not complete: " +
                            holoscan::format_diagnostics(session.diagnostics(), plan.graph()));
  VERIFY_MSG(outcome->pairs == count, what + ": " + std::to_string(outcome->pairs) + " pairs read");
  for (const std::string& problem : outcome->problems) VERIFY_MSG(false, what + ": " + problem);
  VERIFY_MSG(outcome->max_row_error < 0.1, what + ": rows differ");
  VERIFY_MSG(outcome->max_position_error < 0.15,
             what + ": dots misplaced, as by a stale or unfinished remap");
  std::cout << "  " << what << ": row error max " << outcome->max_row_error
            << " px, position error max " << outcome->max_position_error << " px\n";
  return RunResult{.ok = completed && outcome->problems.empty(),
                   .cost = {.host_waits = after.host_waits - before.host_waits,
                            .host_copies = after.host_copies - before.host_copies}};
}

bool integrated_gpu() {
  int device = 0;
  int integrated = 0;
  return cudaGetDevice(&device) == cudaSuccess &&
         cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device) == cudaSuccess &&
         integrated != 0;
}

// Per pair, the operator waits on the host once for VPI's work on the VIC or
// PVA and never for cuda's on device input; once for pinned input, which the
// runtime would hand back to its producer while the GPU still reads it (that
// wait also covers VPI's); for a pageable host producer and for CUDA to stage
// each pageable copy; and it copies host images unless they are pinned and the
// GPU reads them in place (an integrated GPU).
void test_backends() {
  constexpr std::uint32_t kShort = 3;
  constexpr std::uint32_t kLong = 9;
  const auto frames = std::make_shared<const Frames>(kLong);
  const bool integrated = integrated_gpu();
  std::cout << "  integrated GPU: " << (integrated ? "yes" : "no") << '\n';
  for (const Backend backend : available_backends()) {
    if (const auto why =
            try_backend(backend, PixelFormat::kMono8, Interpolation::kLinear, kWidth, kHeight)) {
      std::cout << "  " << backend_name(backend) << " cannot run here: " << *why << '\n';
      continue;
    }
    for (const Memory memory : {Memory::kDevice, Memory::kPinned, Memory::kPageable}) {
      const RunResult short_run = run_pipeline(frames, backend, memory, kShort);
      const RunResult long_run = run_pipeline(frames, backend, memory, kLong);
      if (!short_run.ok || !long_run.ok) continue;
      // The difference leaves out what is paid once, as building the remappers.
      const double pairs = kLong - kShort;
      const double waits = (long_run.cost.host_waits - short_run.cost.host_waits) / pairs;
      const double copies = (long_run.cost.host_copies - short_run.cost.host_copies) / pairs;
      const bool waits_for_stream = backend != Backend::kCuda || memory == Memory::kPinned;
      const double expected_waits =
          (waits_for_stream ? 1 : 0) + (memory == Memory::kPageable ? 4 : 0);
      const double expected_copies =
          memory == Memory::kDevice || (memory == Memory::kPinned && integrated) ? 0 : 2;
      std::cout << "  " << backend_name(backend) << ", " << memory_name(memory)
                << ": host waits per pair " << waits << ", host copies per pair " << copies << '\n';
      VERIFY_MSG(waits == expected_waits, std::string(backend_name(backend)) + ", " +
                                              memory_name(memory) + ": " + std::to_string(waits) +
                                              " host waits per pair, not " +
                                              std::to_string(expected_waits));
      VERIFY_MSG(copies == expected_copies,
                 std::string(backend_name(backend)) + ", " + memory_name(memory) + ": " +
                     std::to_string(copies) + " host copies per pair, not " +
                     std::to_string(expected_copies));
    }
  }
}

}  // namespace
}  // namespace holoscan::rectify

int main() {
  namespace rc = holoscan::rectify;
  return rc::test::run({
      {"rectified pairs read on the GPU at once, and the cost per pair", rc::test_backends},
  });
}
