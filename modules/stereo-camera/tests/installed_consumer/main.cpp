// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-stereo-camera package: it
// writes a synthetic recording, plays it through StereoCameraOp into device
// memory, and checks that every frame's eyes, RGB image, depth and calibration
// arrive as recorded.

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <holoscan/stereo_camera/calibration_message.hpp>
#include <holoscan/stereo_camera/stereo_camera_op.hpp>
#include <holoscan/stereo_camera/synthetic.hpp>

#include "graph_support.hpp"

namespace {

namespace sc = holoscan::stereo_camera;

constexpr int kWidth = 128;
constexpr int kHeight = 96;
constexpr std::uint32_t kFrames = 6;

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path recording = argc > 1 ? argv[1] : "recording";
  std::filesystem::remove_all(recording);
  sc::write_synthetic_recording(recording, kWidth, kHeight, kFrames, true, true);

  holoscan::Graph graph{"stereo_camera_consumer"};
  sc::StereoCameraConfig config;
  config.source = sc::SourceConfig{.backend = sc::Backend::kPlayback,
                                   .rgb = true,
                                   .depth = true,
                                   .playback = sc::PlaybackOptions{.directory = recording}};
  config.period = std::chrono::milliseconds{5};
  config.memory = sc::OutputMemory::kDevice;
  auto camera = graph.op<sc::StereoCameraOp>("camera", config);
  auto received = std::make_shared<sc::test::Received>();
  auto sink = graph.op<sc::test::FrameSinkOp>(
      "sink", sc::test::FrameSinkConfig{
                  .rgb = true, .depth = true, .expected = kFrames, .received = received});
  sc::test::connect(graph, camera, sink, true, true,
                    holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kBlocking});
  const auto outcome = sc::test::run_graph(graph, sc::test::device_outputs(true, true));
  if (!outcome.completed) {
    std::cerr << outcome.diagnostics << '\n';
  }

  const std::string calibration =
      sc::format_calibration(sc::synthetic_calibration(kWidth, kHeight, true, true));
  std::uint32_t wrong = 0;
  sc::Image left;
  sc::Image right;
  sc::Image rgb;
  sc::Image depth;
  for (std::uint32_t i = 0; i < received->frames.size(); ++i) {
    const sc::test::ReceivedFrame& frame = received->frames[i];
    sc::render_synthetic_frame(kWidth, kHeight, i, left, right, &rgb, &depth);
    wrong += !frame.consistent || frame.frame_id != i + 1U;
    wrong += !sc::same_pixels(frame.left.view(sc::PixelFormat::kMono8), left.view());
    wrong += !sc::same_pixels(frame.right.view(sc::PixelFormat::kMono8), right.view());
    wrong += !sc::same_pixels(frame.rgb->view(sc::PixelFormat::kRgb8), rgb.view());
    wrong += !sc::same_pixels(frame.depth->view(sc::PixelFormat::kMono16), depth.view());
    wrong += sc::format_calibration(sc::from_message(frame.calibration)) != calibration;
  }
  const bool passed = outcome.completed && received->frames.size() == kFrames && wrong == 0;
  std::cout << "stereo_camera_consumer: frames=" << received->frames.size() << '/' << kFrames
            << " wrong=" << wrong << " realsense_backend=" << sc::realsense_available() << '\n'
            << "stereo_camera_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
