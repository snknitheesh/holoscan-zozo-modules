// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CameraInfoSourceOp: what it sends for one camera_info file and for a stereo
// pair's two (the files' calibration, stamped with the send time, `count`
// times on its period), and what it refuses when constructed.

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/messages/synthetic_calibration.hpp>

#include "graph_support.hpp"
#include "test_support.hpp"

namespace holoscan::messages {
namespace {

namespace fs = std::filesystem;
constexpr std::size_t kCount = 4;
constexpr auto kPeriod = std::chrono::milliseconds{15};

struct Folder {
  Folder() : path(fs::temp_directory_path() / ("messages_source_" + std::to_string(::getpid()))) {
    fs::create_directories(path);
  }
  ~Folder() { fs::remove_all(path); }
  Folder(const Folder&) = delete;
  Folder& operator=(const Folder&) = delete;
  fs::path path;
};

// The capture times step by the period (to a millisecond of scheduling) and
// equal the headers'.
template <typename Message>
void check_times(const std::vector<test::Received<Message>>& received) {
  for (std::size_t i = 0; i < received.size(); ++i) {
    VERIFY(received[i].data.header->capture_timestamp_ns == received[i].capture_timestamp_ns);
    if (i == 0) continue;
    const std::int64_t step =
        received[i].capture_timestamp_ns - received[i - 1].capture_timestamp_ns;
    VERIFY_MSG(step > 0 && std::abs(step - std::chrono::nanoseconds(kPeriod).count()) <
                               std::chrono::nanoseconds(std::chrono::milliseconds{5}).count(),
               "step " + std::to_string(step) + " ns");
  }
}

template <typename Message>
std::vector<test::Received<Message>> run_source(CameraInfoSourceConfig config,
                                                std::size_t expected) {
  auto received = std::make_shared<std::vector<test::Received<Message>>>();
  holoscan::Graph graph{"source"};
  auto source = graph.op<CameraInfoSourceOp>("camera_info", std::move(config));
  auto sink = graph.op<test::CaptureOp<Message>>("sink", received, expected);
  if constexpr (std::is_same_v<Message, schema::CameraInfoT>) {
    graph.add_flow(source->camera_info, sink->input, test::lossless());
  } else {
    graph.add_flow(source->stereo_camera_info, sink->input, test::lossless());
  }
  const test::RunOutcome outcome = test::run_graph(graph);
  VERIFY_MSG(outcome.ok(), outcome.diagnostics);
  return *received;
}

void test_mono() {
  const Folder folder;
  const CameraInfo truth = synthetic_stereo_camera_info(DistortionModel::kPlumbBob, 640, 480).left;
  write_camera_info_yaml(truth, folder.path / "camera.yaml");
  const auto received = run_source<schema::CameraInfoT>(
      {.camera_info_path = folder.path / "camera.yaml", .period = kPeriod, .count = kCount},
      kCount);
  VERIFY(received.size() == kCount);
  for (const auto& message : received) {
    CameraInfo info = from_message(message.data);
    VERIFY(info.header.frame_id == "left" && info.header.device_sequence == 0);
    info.header.capture_timestamp_ns = truth.header.capture_timestamp_ns;
    VERIFY(info == truth);
  }
  check_times(received);
}

void test_stereo() {
  const Folder folder;
  const StereoCameraInfo truth =
      synthetic_stereo_camera_info(DistortionModel::kEquidistant, 640, 480);
  write_stereo_camera_info_yaml(truth, folder.path / "left.yaml", folder.path / "right.yaml");
  const StereoCameraInfo from_files =
      read_stereo_camera_info_yaml(folder.path / "left.yaml", folder.path / "right.yaml");
  // Unlimited: the sink ends the run.
  const auto received =
      run_source<schema::StereoCameraInfoT>({.camera_info_path = folder.path / "left.yaml",
                                             .right_camera_info_path = folder.path / "right.yaml",
                                             .period = kPeriod},
                                            kCount);
  VERIFY(received.size() == kCount);
  for (const auto& message : received) {
    const StereoCameraInfo info = from_message(message.data);
    VERIFY(same_calibration(info, from_files));
    VERIFY(info.header.frame_id == "left" && info.right.header.frame_id == "right");
    VERIFY(info.left.header.capture_timestamp_ns == message.capture_timestamp_ns &&
           info.right.header.capture_timestamp_ns == message.capture_timestamp_ns);
  }
  check_times(received);
}

void test_refusals() {
  const Folder folder;
  const fs::path missing = folder.path / "missing.yaml";
  VERIFY(test::refuses([&] { CameraInfoSourceOp op({.camera_info_path = missing}); },
                       missing.string() + ": cannot read"));
  VERIFY(test::refuses([&] { CameraInfoSourceOp op({}); }, "camera_info_path is required"));
  write_camera_info_yaml(synthetic_stereo_camera_info(DistortionModel::kPlumbBob).left,
                         folder.path / "camera.yaml");
  VERIFY(test::refuses(
      [&] {
        CameraInfoSourceOp op({.camera_info_path = folder.path / "camera.yaml",
                               .period = std::chrono::nanoseconds{0}});
      },
      "the period must be > 0"));
  {
    std::ofstream bad(folder.path / "bad.yaml");
    bad << "image_width: 640\nimage_height: 480\nfocal: 3\n";
  }
  VERIFY(
      test::refuses([&] { CameraInfoSourceOp op({.camera_info_path = folder.path / "bad.yaml"}); },
                    "'focal' (line 3) is not a camera_info key"));
  // A single camera's file is not a stereo pair: its P has no baseline.
  VERIFY(test::refuses(
      [&] {
        CameraInfoSourceOp op({.camera_info_path = folder.path / "camera.yaml",
                               .right_camera_info_path = folder.path / "camera.yaml"});
      },
      "carries no baseline"));
}

}  // namespace
}  // namespace holoscan::messages

int main() {
  namespace m = holoscan::messages;
  return holoscan::messages::test::run({
      {"one camera_info file", m::test_mono},
      {"a stereo pair's files, until stopped", m::test_stereo},
      {"refusals at construction", m::test_refusals},
  });
}
