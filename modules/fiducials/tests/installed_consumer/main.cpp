// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-fiducials package: the synthetic camera's
// calibration is written to a ROS camera_info file and sent by holoscan-messages'
// CameraInfoSourceOp; two sources render synthetic markers and a ChArUco board once it has
// come; ArucoDetectorOp and CharucoDetectorOp find them on every backend, with poses from the
// camera info; sinks check that each frame has the scene's six markers with their poses, or all
// of the board's corners and its pose. With the overlay component, FiducialOverlayOp draws the
// first detector's markers too. The graph is lossless and runs on one worker.

#include <stdlib.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/fiducials/aruco_detector_op.hpp>
#include <holoscan/fiducials/charuco_detector_op.hpp>
#include <holoscan/fiducials/synthetic_scene.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if FIDUCIALS_CONSUMER_WITH_OVERLAY
#include <holoscan/fiducials/fiducial_overlay_op.hpp>
#endif

namespace {

namespace fd = holoscan::fiducials;
namespace hm = holoscan::messages;

constexpr int kWidth = 960;
constexpr int kHeight = 540;
constexpr std::uint32_t kFrames = 4;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
const fd::CharucoBoardConfig kBoard{};

// A folder for the camera_info file, which mkdtemp() names unpredictably and makes private;
// removed with everything in it.
class TemporaryFolder {
 public:
  TemporaryFolder() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "fiducials_consumer_XXXXXX").string();
    if (::mkdtemp(pattern.data()) == nullptr) {
      throw std::system_error(errno, std::generic_category(), "mkdtemp");
    }
    path_ = pattern;
  }
  ~TemporaryFolder() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  TemporaryFolder(const TemporaryFolder&) = delete;
  TemporaryFolder& operator=(const TemporaryFolder&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

// Whether an emit or allocation failed only because the consumers have no room yet: the
// source sends the same frame again on its next tick.
bool backpressured(const holoscan::Error& error) {
  return error.code == holoscan::ErrorCode::kBackpressured ||
         error.code == holoscan::ErrorCode::kResourceExhausted;
}

// kFrames RGB8 renderings of marker scenes, or of board scenes, once the camera info has come
// (so that every frame finds it), then idle.
class SceneSourceOp final : public holoscan::Operator<> {
 public:
  explicit SceneSourceOp(bool board) : board_(board) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(camera_info, "camera_info");
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
    result.input(camera_info,
                 holoscan::LatestWithin{.max_age = hm::CameraInfoInputOptions{}.max_age});
    result.on_missing(camera_info, holoscan::SafeDefault{});
    return result;
  }

  void start() override {
    info_.reset();
    emitted_ = 0;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto info = info_.update(camera_info);
    if (!info) return holoscan::make_unexpected(std::move(info).error());
    if (*info == nullptr || emitted_ >= kFrames) return {};
    const fd::SyntheticScene scene =
        board_ ? fd::make_board_scene(kBoard, emitted_ * 10, kWidth, kHeight)
               : fd::make_marker_scene("DICT_4X4_50", emitted_ * 10, kWidth, kHeight);
    const std::vector<std::uint8_t> pixels = fd::render_scene(scene, 3);
    const std::array<std::int64_t, 3> shape{kHeight, kWidth, 3};
    auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      return backpressured(loan.error()) ? holoscan::expected<void, holoscan::Error>{}
                                         : holoscan::make_unexpected(std::move(loan).error());
    }
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
    // Busy consumers refuse the frame; send it again on the next tick.
    return backpressured(emitted.error()) ? holoscan::expected<void, holoscan::Error>{} : emitted;
  }

  holoscan::Input<hm::schema::CameraInfoT> camera_info;
  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  bool board_;
  hm::LatestCameraInfo<hm::schema::CameraInfoT> info_;
  std::uint32_t emitted_{};
};

struct Tally {
  std::string what;
  std::uint32_t frames{};
  // Frames with every marker and its pose, or every corner and the board's pose.
  std::uint32_t complete{};
};

// Counts complete frames; the last sink to finish ends the run.
class TallyOp final : public holoscan::Operator<> {
 public:
  TallyOp(std::shared_ptr<Tally> tally, std::shared_ptr<std::atomic<int>> remaining)
      : tally_(std::move(tally)), remaining_(std::move(remaining)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(detections, "detections"); }

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
    bool posed = true;
    for (const auto& marker : message->markers) posed = posed && marker.has_pose();
    const bool complete = message->board
                              ? message->board->corners.size() == 24U && message->board->has_pose
                              : message->markers.size() == 6U && posed;
    tally_->complete += complete ? 1U : 0U;
    if (tally_->frames == kFrames && remaining_->fetch_sub(1) == 1) context.request_completion();
    return {};
  }

  holoscan::Input<fd::schema::FiducialDetectionsT> detections;

 private:
  std::shared_ptr<Tally> tally_;
  std::shared_ptr<std::atomic<int>> remaining_;
};

#if FIDUCIALS_CONSUMER_WITH_OVERLAY
// Counts what arrives on one port: the overlay's images, or its label tensors.
template <typename T>
class CountOp final : public holoscan::Operator<> {
 public:
  explicit CountOp(std::shared_ptr<std::uint32_t> count) : count_(std::move(count)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto received = in.receive();
    if (!received) return holoscan::make_unexpected(std::move(received).error());
    ++*count_;
    return {};
  }

  holoscan::Input<T> in;

 private:
  std::shared_ptr<std::uint32_t> count_;
};
#endif

}  // namespace

int main() try {
  // The run constructs CameraInfoSourceOp again, reading the file then: it stays until main()
  // returns.
  const TemporaryFolder folder;
  const std::filesystem::path camera_file = folder.path() / "camera_info.yaml";
  const hm::CameraInfo camera = fd::make_marker_scene("DICT_4X4_50", 0, kWidth, kHeight).camera;
  hm::write_camera_info_yaml(camera, camera_file);

  holoscan::Graph graph{"fiducials_consumer"};
  auto camera_info = graph.op<hm::CameraInfoSourceOp>(
      "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = camera_file});
  auto markers = graph.op<SceneSourceOp>("markers", false);
  auto boards = graph.op<SceneSourceOp>("boards", true);
  graph.add_flow(camera_info->camera_info, markers->camera_info,
                 hm::latest_camera_info_connection());
  graph.add_flow(camera_info->camera_info, boards->camera_info,
                 hm::latest_camera_info_connection());
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  const std::vector<fd::Backend> backends = fd::available_backends();
  auto remaining = std::make_shared<std::atomic<int>>(static_cast<int>(2 * backends.size()));
  std::vector<std::shared_ptr<Tally>> tallies;
  const holoscan::Output<fd::schema::FiducialDetectionsT>* first = nullptr;
  for (const fd::Backend backend : backends) {
    const std::string name(fd::backend_name(backend));
    auto aruco = graph.op<fd::ArucoDetectorOp>(
        "aruco_" + name,
        fd::ArucoDetectorConfig{.backend = backend, .estimate_pose = true, .marker_length = 0.1});
    auto charuco = graph.op<fd::CharucoDetectorOp>(
        "charuco_" + name,
        fd::CharucoDetectorConfig{.backend = backend, .board = kBoard, .estimate_pose = true});
    auto aruco_tally = tallies.emplace_back(std::make_shared<Tally>(Tally{.what = "aruco "}));
    auto charuco_tally = tallies.emplace_back(std::make_shared<Tally>(Tally{.what = "charuco "}));
    auto aruco_sink = graph.op<TallyOp>("tally_aruco_" + name, aruco_tally, remaining);
    auto charuco_sink = graph.op<TallyOp>("tally_charuco_" + name, charuco_tally, remaining);
    graph.add_flow(markers->image, aruco->image, lossless);
    graph.add_flow(boards->image, charuco->image, lossless);
    graph.add_flow(camera_info->camera_info, aruco->camera_info,
                   hm::latest_camera_info_connection());
    graph.add_flow(camera_info->camera_info, charuco->camera_info,
                   hm::latest_camera_info_connection());
    graph.add_flow(aruco->detections, aruco_sink->detections, lossless);
    graph.add_flow(charuco->detections, charuco_sink->detections, lossless);
    if (first == nullptr) first = &aruco->detections;
  }
#if FIDUCIALS_CONSUMER_WITH_OVERLAY
  auto drawn = std::make_shared<std::uint32_t>(0U);
  auto labelled = std::make_shared<std::uint32_t>(0U);
  auto overlay = graph.op<fd::FiducialOverlayOp>(
      "overlay", fd::FiducialOverlayConfig{
                     .width = kWidth, .height = kHeight, .draw_axes = true, .marker_length = 0.1});
  auto images = graph.op<CountOp<holoscan::schema::ImageT>>("overlay_images", drawn);
  auto labels = graph.op<CountOp<holoscan::Tensor>>("overlay_labels", labelled);
  graph.add_flow(*first, overlay->detections, lossless);
  graph.add_flow(camera_info->camera_info, overlay->camera_info,
                 hm::latest_camera_info_connection());
  graph.add_flow(overlay->overlay, images->in, lossless);
  graph.add_flow(overlay->labels, labels->in, lossless);
#endif
  // One worker for the whole graph: every edge but the camera info's is lossless, and across
  // workers a full queue refuses what is sent to it.
  graph.partition("fiducials_consumer").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const auto plan = holoscan::compile(graph);
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) return 1;
  const std::uint32_t workers = plan.scheduler_plan().worker_count;
  // No signal handling of its own: a test that is killed should die.
  auto session = holoscan::run_async(plan);
  if (!session.wait_for(std::chrono::seconds{120})) {
    std::cerr << "fiducials_consumer: timed out\n";
    session.request_stop();
    session.wait();
  }
  if (!session.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
  }
  bool passed = workers == 1 && !tallies.empty() &&
                session.termination() == holoscan::RunTermination::kRequestedCompleted;
  std::cout << "fiducials_consumer: workers=" << workers << '\n';
  for (const auto& tally : tallies) {
    std::cout << "fiducials_consumer: " << tally->what << " frames=" << tally->frames
              << " complete=" << tally->complete << '\n';
    passed = passed && tally->frames == kFrames && tally->complete == kFrames;
  }
#if FIDUCIALS_CONSUMER_WITH_OVERLAY
  // The overlay may still be drawing the last frame when the last sink completes the run.
  std::cout << "fiducials_consumer: overlay images=" << *drawn << " labels=" << *labelled << '\n';
  passed = passed && *drawn + 1 >= kFrames && *labelled + 1 >= kFrames;
#endif
  std::cout << "fiducials_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
} catch (const std::exception& error) {
  std::cerr << "fiducials_consumer: " << error.what() << '\n';
  return 1;
}
