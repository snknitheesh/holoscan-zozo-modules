// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Uses the installed holoscan-messages package: writes a camera_info file,
// and runs CameraInfoSourceOp into an operator that takes frames and the
// camera info with camera_info_contract() and LatestCameraInfo.

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/synthetic_calibration.hpp>
#include <holoscan/schemas/header_schema_traits.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace hm = holoscan::messages;
using Frame = holoscan::schema::HeaderT;
using std::chrono_literals::operator""ms;
constexpr std::uint32_t kFrames = 20;

class FrameSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(frame, "frame"); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 5ms});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (sent_ >= kFrames) return {};
    ++sent_;
    return frame.emit(Frame{}, holoscan::EmitOptions{.capture_time = context.activation_time()});
  }
  holoscan::Output<Frame> frame;

 private:
  std::uint32_t sent_{};
};

struct Result {
  std::uint32_t frames{};
  std::uint32_t with_info{};
  double fx{};
};

class ConsumerOp final : public holoscan::Operator<> {
 public:
  explicit ConsumerOp(std::shared_ptr<Result> result) : result_(std::move(result)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(frame, "frame");
    spec.input(camera_info, "camera_info");
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    return hm::camera_info_contract(camera_info, frame);
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto info = info_.update(camera_info);
    if (!info) return holoscan::make_unexpected(std::move(info).error());
    if (*info != nullptr) {
      ++result_->with_info;
      result_->fx = (*info)->fx;
    }
    if (++result_->frames == kFrames) context.request_completion();
    return {};
  }
  holoscan::Input<Frame> frame;
  holoscan::Input<hm::schema::CameraInfoT> camera_info;

 private:
  std::shared_ptr<Result> result_;
  hm::LatestCameraInfo<hm::schema::CameraInfoT> info_;
};

}  // namespace

int main() try {
  const std::filesystem::path file = std::filesystem::temp_directory_path() /
                                     ("messages_consumer_" + std::to_string(::getpid()) + ".yaml");
  const hm::CameraInfo truth =
      hm::synthetic_stereo_camera_info(hm::DistortionModel::kPlumbBob).left;
  hm::write_camera_info_yaml(truth, file);

  auto result = std::make_shared<Result>();
  holoscan::Graph graph{"messages_consumer"};
  auto frames = graph.op<FrameSourceOp>("frames");
  auto source = graph.op<hm::CameraInfoSourceOp>(
      "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = file, .period = 20ms});
  auto consumer = graph.op<ConsumerOp>("consumer", result);
  graph.add_flow(frames->frame, consumer->frame);
  graph.add_flow(source->camera_info, consumer->camera_info, hm::latest_camera_info_connection());
  // One worker for the whole graph, as the module's README advises with a
  // CameraInfoSourceOp.
  graph.partition("messages_consumer").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::filesystem::remove(file);
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
    return 1;
  }
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  if (session.is_running()) session.request_stop();
  session.wait();
  // The run constructs the operators again, reading the file then: remove it only now.
  std::filesystem::remove(file);
  const bool passed = session.termination() == holoscan::RunTermination::kRequestedCompleted &&
                      result->frames == kFrames && result->with_info + 1 >= kFrames &&
                      result->fx == truth.fx;
  std::cout << "messages_consumer: frames=" << result->frames
            << " with_camera_info=" << result->with_info << " fx=" << result->fx << '\n'
            << "messages_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
} catch (const std::exception& error) {
  std::cerr << "messages_consumer: " << error.what() << '\n';
  return 1;
}
