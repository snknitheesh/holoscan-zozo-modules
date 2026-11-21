// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-stitching package: each of
// the four surround cameras' calibration is written as a ROS camera_info file
// and sent by a CameraInfoSourceOp; a source renders the synthetic world
// through the cameras of a moving rig, in CUDA device memory; StitchOp makes
// a bird's-eye view of them; a sink checks every view against the directly
// rendered ground truth. Every frame must arrive, so the graph runs on one
// worker.

#include <cuda_runtime_api.h>
#include <stdlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/stitching/remap.hpp>
#include <holoscan/stitching/stitch_op.hpp>
#include <holoscan/stitching/synthetic_world.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace fs = std::filesystem;
namespace hm = holoscan::messages;
namespace st = holoscan::stitching;
using std::chrono_literals::operator""ms;

constexpr int kWidth = 640;
constexpr int kHeight = 360;
constexpr std::size_t kCameras = 4;
constexpr std::uint32_t kFrames = 5;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// A busy consumer: an emit refused before anything is delivered, or an
// output pool whose every slot is still held. The source sends the same
// frame again on its next tick.
bool consumers_busy(const holoscan::Error& error) {
  return error.code == holoscan::ErrorCode::kBackpressured ||
         error.code == holoscan::ErrorCode::kResourceExhausted;
}

struct Tally {
  std::atomic<std::uint32_t> frames{};
  double worst_psnr{99.0};
  double worst_corner{};
  double worst_coverage{1.0};
};

// Frame sets of the surround rig, RGB8 on the GPU, until kFrames have been
// checked (at most twice as many), then idle. The first tick sends nothing,
// so that the camera info, sent when the run starts, comes first; should a
// frame set still come before it, StitchOp skips it and the next one counts.
class RigSourceOp final : public holoscan::Operator<> {
 public:
  explicit RigSourceOp(std::shared_ptr<const Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    for (std::size_t c = 0; c < images.size(); ++c) {
      spec.output(images[c], "image_" + std::to_string(c))
          .produces_tensor(holoscan::TensorOutputSpec{
              .representation =
                  holoscan::TensorRepresentation{
                      .memory_kind = holoscan::MemoryKind::kCudaDevice,
                      .dtype = kUInt8,
                      .rank = 3U,
                      .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                  },
              .bounds = holoscan::tensor_bounds(std::uint64_t{kWidth} * kHeight * 3U),
              .storage = holoscan::TensorOutputStorage::kRuntimePool,
          })
          .payload_options(holoscan::PayloadOptions{
              .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
          });
    }
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{20}});
    return result;
  }

  void start() override {
    sent_ = 0;
    delayed_ = false;
    pending_.reset();
    emitted_ = 0;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (!pending_ && (tally_->frames.load() >= kFrames || sent_ >= 2U * kFrames)) {
      return {};
    }
    if (!delayed_) {
      delayed_ = true;
      return {};
    }
    cudaStream_t stream = context.cuda_stream().get();
    if (!pending_) {
      pending_ =
          holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = sent_ + 1U};
      emitted_ = 0;
    }
    // Every image still to send is allocated and rendered before any is
    // sent, so that a full pool leaves nothing half sent.
    std::array<std::optional<holoscan::TensorOutputLoan>, kCameras> loans;
    for (std::size_t c = emitted_; c < kCameras; ++c) {
      auto loan = images[c].allocate_tensor(holoscan::TensorLoanRequest{
          .shape = std::array<std::int64_t, 3>{kHeight, kWidth, 3}, .dtype = kUInt8});
      if (!loan) {
        if (consumers_busy(loan.error())) return {};
        return holoscan::make_unexpected(std::move(loan).error());
      }
      auto writer = loan->write(stream);
      if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
      }
      if (const cudaError_t status =
              st::render_camera(world_, rig_.cameras[c], st::moving_pose(10U * sent_), 1.0F,
                                st::PixelFormat::kRgb8, writer->data(), kWidth * 3U, 3, stream);
          status != cudaSuccess) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, cudaGetErrorString(status)});
      }
      if (auto committed = std::move(*writer).commit(); !committed) {
        return holoscan::make_unexpected(std::move(committed).error());
      }
      loans[c].emplace(std::move(*loan));
    }
    // A refused image is sent again on the next tick, with the frame set's
    // capture time, after the ones that went.
    for (; emitted_ < kCameras; ++emitted_) {
      holoscan::schema::ImageT descriptor;
      descriptor.width = kWidth;
      descriptor.height = kHeight;
      descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
      auto sent = images[emitted_].emit_tensor(std::move(*loans[emitted_]), descriptor, *pending_);
      if (!sent) {
        if (consumers_busy(sent.error())) return {};
        return sent;
      }
    }
    pending_.reset();
    ++sent_;
    return {};
  }

  std::array<holoscan::Output<holoscan::schema::ImageT>, kCameras> images;

 private:
  std::shared_ptr<const Tally> tally_;
  st::SyntheticWorld world_;
  st::CameraRig rig_ = st::surround_rig(kWidth, kHeight);
  std::uint32_t sent_{};
  bool delayed_{};
  std::optional<holoscan::EmitOptions> pending_;
  std::size_t emitted_{};
};

// Checks each bird's-eye view against the truth of its frame: both copied to
// pinned buffers on the operator's stream, one wait per frame.
class CheckOp final : public holoscan::Operator<> {
 public:
  CheckOp(std::shared_ptr<Tally> tally, st::OutputGeometry geometry)
      : tally_(std::move(tally)), geometry_(geometry) {}
  ~CheckOp() override { release(); }
  CheckOp(const CheckOp&) = delete;
  CheckOp& operator=(const CheckOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(stitched, "stitched").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{stitched});
    return result;
  }

  void start() override {
    release();
    const std::size_t bytes = pitch() * geometry_.height;
    if (cudaMalloc(&truth_device_, bytes) != cudaSuccess ||
        cudaMallocHost(&got_, bytes) != cudaSuccess ||
        cudaMallocHost(&truth_, bytes) != cudaSuccess) {
      release();
      throw std::runtime_error("CheckOp: out of memory");
    }
    resolution_ =
        st::resolution_map(st::build_remap_tables(st::surround_rig(kWidth, kHeight), geometry_));
  }

  void stop() override { release(); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto image = stitched.receive();
    if (!image) {
      return holoscan::make_unexpected(std::move(image).error());
    }
    cudaStream_t stream = context.cuda_stream().get();
    const std::size_t bytes = pitch() * geometry_.height;
    if (auto copied = image->data.data->copy_to_host_async(got_, bytes, stream); !copied) {
      return copied;
    }
    const st::RigPose pose =
        st::moving_pose(10U * static_cast<std::uint32_t>(image->metadata.frame_id - 1U));
    if (st::render_truth(world_, geometry_, pose, truth_device_, pitch(), 3, stream) !=
            cudaSuccess ||
        cudaMemcpyAsync(truth_, truth_device_, bytes, cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "rendering the truth failed"});
    }
    const st::Comparison c = st::compare_with_truth(
        static_cast<const std::uint8_t*>(got_), pitch(), static_cast<const std::uint8_t*>(truth_),
        pitch(), geometry_.width, geometry_.height, st::checker_corners(world_, geometry_, pose), 2,
        resolution_);
    tally_->worst_psnr = std::min(tally_->worst_psnr, c.psnr_db);
    tally_->worst_corner = std::max(tally_->worst_corner, c.max_corner_error);
    tally_->worst_coverage = std::min(tally_->worst_coverage, c.coverage);
    if (tally_->frames.fetch_add(1) + 1U == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> stitched;

 private:
  [[nodiscard]] std::size_t pitch() const { return static_cast<std::size_t>(geometry_.width) * 4U; }

  void release() noexcept {
    if (truth_device_ != nullptr) cudaFree(truth_device_);
    if (got_ != nullptr) cudaFreeHost(got_);
    if (truth_ != nullptr) cudaFreeHost(truth_);
    truth_device_ = got_ = truth_ = nullptr;
  }

  std::shared_ptr<Tally> tally_;
  st::OutputGeometry geometry_;
  st::SyntheticWorld world_;
  std::vector<float> resolution_;
  void* truth_device_{};
  void* got_{};
  void* truth_{};
};

// A folder for the camera_info files that mkdtemp() names and makes private;
// removed with everything in it when main() returns, whichever way.
class TemporaryFolder {
 public:
  TemporaryFolder() {
    std::string pattern = (fs::temp_directory_path() / "stitching_consumer_XXXXXX").string();
    if (::mkdtemp(pattern.data()) == nullptr) {
      throw std::system_error(errno, std::generic_category(), "mkdtemp");
    }
    path_ = pattern;
  }
  ~TemporaryFolder() {
    std::error_code ignored;
    fs::remove_all(path_, ignored);
  }
  TemporaryFolder(const TemporaryFolder&) = delete;
  TemporaryFolder& operator=(const TemporaryFolder&) = delete;
  [[nodiscard]] const fs::path& path() const noexcept { return path_; }

 private:
  fs::path path_;
};

}  // namespace

int main() try {
  // The run constructs the operators again, reading the files then: they
  // stay until main() returns.
  const TemporaryFolder folder;
  const st::CameraRig rig = st::surround_rig(kWidth, kHeight);
  const st::OutputGeometry geometry = st::ground_view(256, 256, 5.5);
  st::StitchConfig config{.output = geometry};
  for (const st::Camera& camera : rig.cameras) {
    config.camera_from_rig.push_back(camera.camera_from_rig);
  }

  holoscan::Graph graph{"stitching_consumer"};
  auto tally = std::make_shared<Tally>();
  auto source = graph.op<RigSourceOp>("source", tally);
  auto stitch = graph.op<st::StitchOp>("stitch", config);
  auto check = graph.op<CheckOp>("check", tally, geometry);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  for (std::size_t c = 0; c < kCameras; ++c) {
    const fs::path file = folder.path() / (rig.cameras[c].info.header.frame_id + ".yaml");
    hm::write_camera_info_yaml(rig.cameras[c].info, file);
    auto info = graph.op<hm::CameraInfoSourceOp>(
        "camera_info_" + std::to_string(c), hm::CameraInfoSourceConfig{.camera_info_path = file});
    graph.add_flow(info->camera_info, stitch->camera_info[c], hm::latest_camera_info_connection());
    graph.add_flow(source->images[c], stitch->images[c], lossless);
  }
  graph.add_flow(stitch->stitched, check->stitched, lossless);
  // One worker: every frame is checked, and the camera info sources would
  // each get a worker of their own otherwise.
  graph.partition("stitching_consumer").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  for (std::size_t c = 0; c < kCameras; ++c) {
    options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = "source",
                                              .output_port = "image_" + std::to_string(c),
                                              .device = holoscan::DeviceId{0}});
  }
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "stitch", .output_port = "stitched", .device = holoscan::DeviceId{0}});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) return 1;
  const std::uint32_t workers = plan.scheduler_plan().worker_count;
  if (workers != 1U) {
    std::cerr << "stitching_consumer: the graph runs on " << workers << " workers, not 1\n";
    return 1;
  }
  // A lost frame set fails the run within a minute.
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  const bool timed_out = session.is_running();
  if (timed_out) session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
  }
  const bool passed = !timed_out &&
                      session.termination() == holoscan::RunTermination::kRequestedCompleted &&
                      tally->frames.load() == kFrames && tally->worst_psnr >= 24.0 &&
                      tally->worst_corner <= 1.0 && tally->worst_coverage >= 0.9;
  std::cout << "stitching_consumer: workers=" << workers << " frames=" << tally->frames.load()
            << '/' << kFrames << " worst_psnr=" << tally->worst_psnr
            << " dB worst_corner=" << tally->worst_corner
            << " px worst_coverage=" << tally->worst_coverage << (timed_out ? " timed out" : "")
            << '\n'
            << "stitching_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
} catch (const std::exception& error) {
  std::cerr << "stitching_consumer: " << error.what() << '\n';
  return 1;
}
