// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-fiducials package: two sources render
// synthetic markers and a ChArUco board, ArucoDetectorOp and CharucoDetectorOp find them on
// every backend, and sinks check that each frame has the scene's six markers, or all of the
// board's corners and its pose.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/fiducials/aruco_detector_op.hpp>
#include <holoscan/fiducials/charuco_detector_op.hpp>
#include <holoscan/fiducials/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace fd = holoscan::fiducials;

constexpr int kWidth = 960;
constexpr int kHeight = 540;
constexpr std::uint32_t kFrames = 4;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
const fd::CharucoBoardConfig kBoard{};

// kFrames RGB8 renderings of marker scenes, or of board scenes, then idle.
class SceneSourceOp final : public holoscan::Operator<> {
 public:
  explicit SceneSourceOp(bool board) : board_(board) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(image, "image")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
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

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{20}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_ >= kFrames) return {};
    const fd::SyntheticScene scene =
        board_ ? fd::make_board_scene(kBoard, emitted_ * 10, kWidth, kHeight)
               : fd::make_marker_scene("DICT_4X4_50", emitted_ * 10, kWidth, kHeight);
    const std::vector<std::uint8_t> pixels = fd::render_scene(scene, 3);
    const std::array<std::int64_t, 3> shape{kHeight, kWidth, 3};
    auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) return holoscan::make_unexpected(std::move(loan).error());
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      std::memcpy(writer.data_as<std::uint8_t>(), pixels.data(), pixels.size());
      return {};
    });
    if (!written) return holoscan::make_unexpected(std::move(written).error());
    holoscan::schema::ImageT descriptor;
    descriptor.width = kWidth;
    descriptor.height = kHeight;
    descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
    auto emitted =
        image.emit_tensor(std::move(*loan), descriptor,
                          holoscan::EmitOptions{.capture_time = context.activation_time()});
    if (emitted) {
      ++emitted_;
      return {};
    }
    // A busy consumer refuses the frame; send it again on the next tick.
    return emitted.error().code == holoscan::ErrorCode::kBackpressured
               ? holoscan::expected<void, holoscan::Error>{}
               : emitted;
  }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  bool board_;
  std::uint32_t emitted_{};
};

struct Tally {
  std::string what;
  std::uint32_t frames{};
  std::uint32_t complete{};  // Frames with every marker, or every corner and a pose.
};

// Counts complete frames; the last sink to finish ends the run.
class TallyOp final : public holoscan::Operator<> {
 public:
  TallyOp(std::shared_ptr<Tally> tally, std::shared_ptr<std::atomic<int>> remaining)
      : tally_(std::move(tally)), remaining_(std::move(remaining)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(detections, "detections").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{detections});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto message = detections.receive_data();
    if (!message) return holoscan::make_unexpected(std::move(message).error());
    tally_->what += tally_->frames == 0 ? message->backend : "";
    ++tally_->frames;
    const bool complete = message->board ? message->board->corners.size() == 24U &&
                                               message->board->has_pose
                                         : message->markers.size() == 6U;
    tally_->complete += complete ? 1U : 0U;
    if (tally_->frames == kFrames && remaining_->fetch_sub(1) == 1) context.request_completion();
    return {};
  }

  holoscan::Input<fd::schema::FiducialDetectionsT> detections;

 private:
  std::shared_ptr<Tally> tally_;
  std::shared_ptr<std::atomic<int>> remaining_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"fiducials_consumer"};
  auto markers = graph.op<SceneSourceOp>("markers", false);
  auto boards = graph.op<SceneSourceOp>("boards", true);
  const fd::SyntheticScene scene = fd::make_marker_scene("DICT_4X4_50", 0, kWidth, kHeight);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                             .queue_depth = 2U};
  const std::vector<fd::Backend> backends = fd::available_backends();
  auto remaining = std::make_shared<std::atomic<int>>(static_cast<int>(2 * backends.size()));
  std::vector<std::shared_ptr<Tally>> tallies;
  for (const fd::Backend backend : backends) {
    const std::string name(fd::backend_name(backend));
    auto aruco = graph.op<fd::ArucoDetectorOp>(
        "aruco_" + name, fd::ArucoDetectorConfig{.backend = backend,
                                                 .intrinsics = scene.intrinsics,
                                                 .marker_length = 0.1});
    auto charuco = graph.op<fd::CharucoDetectorOp>(
        "charuco_" + name, fd::CharucoDetectorConfig{
                               .backend = backend, .board = kBoard, .intrinsics = scene.intrinsics});
    auto aruco_tally = tallies.emplace_back(std::make_shared<Tally>(Tally{.what = "aruco "}));
    auto charuco_tally = tallies.emplace_back(std::make_shared<Tally>(Tally{.what = "charuco "}));
    auto aruco_sink = graph.op<TallyOp>("tally_aruco_" + name, aruco_tally, remaining);
    auto charuco_sink = graph.op<TallyOp>("tally_charuco_" + name, charuco_tally, remaining);
    graph.add_flow(markers->image, aruco->image, lossless);
    graph.add_flow(boards->image, charuco->image, lossless);
    graph.add_flow(aruco->detections, aruco_sink->detections, lossless);
    graph.add_flow(charuco->detections, charuco_sink->detections, lossless);
  }
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "fiducials_consumer: run failed with " << status << '\n';
    return 1;
  }
  bool passed = !tallies.empty();
  for (const auto& tally : tallies) {
    std::cout << "fiducials_consumer: " << tally->what << " frames=" << tally->frames
              << " complete=" << tally->complete << '\n';
    passed = passed && tally->frames == kFrames && tally->complete == kFrames;
  }
  std::cout << "fiducials_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
