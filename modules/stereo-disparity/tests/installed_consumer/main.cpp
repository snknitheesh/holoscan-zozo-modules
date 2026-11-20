// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-stereo-disparity and
// holoscan-messages packages: a source sends frames of the synthetic stereo
// scene as MONO8 pairs in host memory, CameraInfoSourceOp sends the scene's
// camera info from its camera_info files, one StereoDisparityOp per backend
// this platform has matches them, and a sink per backend checks each
// disparity map against the scene's truth. The source holds its first frame
// until the camera info has come, so every matcher matches every frame.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/stereo_disparity/platform.hpp>
#include <holoscan/stereo_disparity/stereo_disparity_op.hpp>
#include <holoscan/stereo_disparity/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace sd = holoscan::stereo_disparity;
namespace hm = holoscan::messages;

constexpr int kWidth = 320;
constexpr int kHeight = 240;
constexpr std::uint32_t kFrames = 3;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// The scene's frame `frame` as MONO8 left and right images, on the host.
std::array<std::vector<std::uint8_t>, 2> render(std::uint32_t frame) {
  const std::size_t pixels = static_cast<std::size_t>(kWidth) * kHeight;
  std::array<std::vector<std::uint8_t>, 2> pair{std::vector<std::uint8_t>(pixels),
                                                std::vector<std::uint8_t>(pixels)};
  std::uint8_t* gpu = nullptr;
  if (cudaMalloc(&gpu, 2 * pixels) != cudaSuccess) return {};
  const bool ok =
      sd::render_scene(sd::make_scene(frame, kWidth, kHeight), 1, gpu, gpu + pixels, nullptr) ==
          cudaSuccess &&
      cudaMemcpy(pair[0].data(), gpu, pixels, cudaMemcpyDeviceToHost) == cudaSuccess &&
      cudaMemcpy(pair[1].data(), gpu + pixels, pixels, cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(gpu);
  return ok ? pair : std::array<std::vector<std::uint8_t>, 2>{};
}

// kFrames stereo pairs, MONO8 in host memory, the first once the camera info
// has come, then idle.
class PairSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(camera_info, "camera_info");
    for (auto* port : {&left, &right}) {
      spec.output(*port, port == &left ? "left" : "right")
          .produces_tensor(holoscan::TensorOutputSpec{
              .representation =
                  holoscan::TensorRepresentation{
                      .memory_kind = holoscan::MemoryKind::kHost,
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
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{50}});
    result.input(camera_info,
                 holoscan::LatestWithin{.max_age = hm::CameraInfoInputOptions{}.max_age});
    result.on_missing(camera_info, holoscan::SafeDefault{});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    // CameraInfoSourceOp sends to every matcher at once, and the graph runs
    // on one worker, so its emit reaches every queue before this tick: once
    // it has come here, the matchers have it too.
    auto info = info_.update(camera_info);
    if (!info) {
      return holoscan::make_unexpected(std::move(info).error());
    }
    if (emitted_ >= kFrames || *info == nullptr) {
      return {};
    }
    // A frame keeps its capture time until both eyes are out.
    if (!capture_) {
      capture_ = context.activation_time();
      pair_ = render(emitted_);
      if (pair_[0].empty()) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "rendering the scene failed"});
      }
    }
    for (int side = 0; side < 2; ++side) {
      if (sent_[side]) continue;
      holoscan::Output<holoscan::schema::ImageT>& port = side == 0 ? left : right;
      auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{
          .shape = std::array<std::int64_t, 2>{kHeight, kWidth}, .dtype = kUInt8});
      if (!loan) {
        return holoscan::make_unexpected(std::move(loan).error());
      }
      auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
        std::copy(pair_[side].begin(), pair_[side].end(), writer.data_as<std::uint8_t>());
        return {};
      });
      if (!written) {
        return written;
      }
      holoscan::schema::ImageT descriptor;
      descriptor.width = kWidth;
      descriptor.height = kHeight;
      descriptor.encoding = holoscan::schema::ImageEncoding_MONO8;
      auto emitted = port.emit_tensor(
          std::move(*loan), descriptor,
          holoscan::EmitOptions{.capture_time = *capture_, .frame_id = emitted_ + 1U});
      if (!emitted) {
        // A busy consumer refuses the frame; send it again on the next tick.
        return emitted.error().code == holoscan::ErrorCode::kBackpressured
                   ? holoscan::expected<void, holoscan::Error>{}
                   : emitted;
      }
      sent_[side] = true;
    }
    ++emitted_;
    capture_.reset();
    sent_ = {};
    return {};
  }

  holoscan::Input<hm::schema::StereoCameraInfoT> camera_info;
  holoscan::Output<holoscan::schema::ImageT> left;
  holoscan::Output<holoscan::schema::ImageT> right;

 private:
  hm::LatestCameraInfo<hm::schema::StereoCameraInfoT> info_;
  std::uint32_t emitted_{};
  std::optional<holoscan::TimePoint> capture_;
  std::array<std::vector<std::uint8_t>, 2> pair_;
  std::array<bool, 2> sent_{};
};

struct Tally {
  std::uint32_t frames{};
  std::uint64_t truth_pixels{};
  std::uint64_t within_1px{};
};

// Checks each disparity map against the frame it came from; the last sink to
// see frame kFrames ends the run.
class CheckOp final : public holoscan::Operator<> {
 public:
  CheckOp(std::shared_ptr<Tally> tally, std::shared_ptr<std::atomic<int>> remaining)
      : tally_(std::move(tally)), remaining_(std::move(remaining)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(disparity, "disparity").queue_depth(2U);
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{disparity});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto image = disparity.receive();
    if (!image) {
      return holoscan::make_unexpected(std::move(image).error());
    }
    std::vector<float> values(static_cast<std::size_t>(kWidth) * kHeight);
    if (auto copied = image->data.data->copy_to_host(values.data(), values.size() * sizeof(float),
                                                     context.cuda_stream());
        !copied) {
      return copied;
    }
    const sd::SyntheticScene scene =
        sd::make_scene(static_cast<std::uint32_t>(image->metadata.frame_id - 1U), kWidth, kHeight);
    for (int v = 0; v < kHeight; ++v) {
      for (int u = 0; u < kWidth; ++u) {
        const float truth = sd::true_disparity(scene, static_cast<float>(u), static_cast<float>(v));
        if (truth < 0.0F) continue;
        ++tally_->truth_pixels;
        const float d = values[static_cast<std::size_t>(v) * kWidth + u];
        tally_->within_1px += d > 0.0F && std::abs(d - truth) <= 1.0F;
      }
    }
    ++tally_->frames;
    if (image->metadata.frame_id == kFrames && remaining_->fetch_sub(1) == 1) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> disparity;

 private:
  std::shared_ptr<Tally> tally_;
  std::shared_ptr<std::atomic<int>> remaining_;
};

}  // namespace

int main() {
  // The scene's calibration as its camera_info files, which CameraInfoSourceOp
  // reads when it is built and again when the run starts.
  std::string folder_pattern =
      (std::filesystem::temp_directory_path() / "stereo_disparity_consumer_XXXXXX").string();
  if (::mkdtemp(folder_pattern.data()) == nullptr) {
    std::cerr << "stereo_disparity_consumer: cannot create a folder\n";
    return 1;
  }
  const std::filesystem::path folder = folder_pattern;
  struct RemoveFolder {
    std::filesystem::path path;
    ~RemoveFolder() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } const remove_folder{folder};
  hm::write_stereo_camera_info_yaml(sd::scene_camera_info(sd::make_scene(0, kWidth, kHeight)),
                                    folder / "left.yaml", folder / "right.yaml");

  const std::vector<sd::Backend> backends = sd::available_backends();
  holoscan::Graph graph{"stereo_disparity_consumer"};
  auto source = graph.op<PairSourceOp>("source");
  auto camera_info = graph.op<hm::CameraInfoSourceOp>(
      "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = folder / "left.yaml",
                                                .right_camera_info_path = folder / "right.yaml"});
  graph.add_flow(camera_info->stereo_camera_info, source->camera_info,
                 hm::latest_camera_info_connection());
  auto remaining = std::make_shared<std::atomic<int>>(static_cast<int>(backends.size()));
  std::vector<std::shared_ptr<Tally>> tallies;
  holoscan::CompileOptions options;
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  for (sd::Backend backend : backends) {
    const std::string name(sd::backend_name(backend));
    auto matcher = graph.op<sd::StereoDisparityOp>(
        "stereo_" + name,
        sd::StereoDisparityConfig{.backend = backend, .max_width = kWidth, .max_height = kHeight});
    tallies.push_back(std::make_shared<Tally>());
    auto check = graph.op<CheckOp>("check_" + name, tallies.back(), remaining);
    graph.add_flow(source->left, matcher->left, lossless);
    graph.add_flow(source->right, matcher->right, lossless);
    graph.add_flow(camera_info->stereo_camera_info, matcher->camera_info,
                   hm::latest_camera_info_connection());
    graph.add_flow(matcher->disparity, check->disparity, lossless);
    // The disparity is in device memory, on GPU 0.
    options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = "stereo_" + name,
                                              .output_port = "disparity",
                                              .device = holoscan::DeviceId{0}});
  }
  // One worker for the whole graph, as the module's README advises with a
  // CameraInfoSourceOp.
  graph.partition("stereo_disparity_consumer").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
    return 1;
  }
  // VPI, asked for the platform's backends above, has started threads of its
  // own, so the SDK's managed signal handling would refuse: this short test
  // leaves signals to the process.
  if (const int status = holoscan::run(plan, holoscan::RunSignalPolicy::kCallerManaged);
      status != 0) {
    std::cerr << "stereo_disparity_consumer: run failed with " << status << '\n';
    return 1;
  }
  bool passed = true;
  for (std::size_t i = 0; i < backends.size(); ++i) {
    const Tally& tally = *tallies[i];
    const double within =
        tally.truth_pixels == 0 ? 0.0 : static_cast<double>(tally.within_1px) / tally.truth_pixels;
    // Every frame: the edges are lossless, and the camera info came first.
    const bool ok = tally.frames == kFrames && within >= 0.8;
    passed = passed && ok;
    std::cout << "stereo_disparity_consumer: backend=" << sd::backend_name(backends[i])
              << " frames=" << tally.frames << " within_1px=" << within << '\n';
  }
  std::cout << "stereo_disparity_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
