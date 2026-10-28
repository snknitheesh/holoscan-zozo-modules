// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The backends without a camera: the playback backend over a synthetic
// recording (every frame in order, once or in a loop, with and without rgb and
// depth), the names of the backends, and the configurations validate() refuses.

#include <cstdint>
#include <filesystem>
#include <string>

#include <holoscan/stereo_camera/source.hpp>
#include <holoscan/stereo_camera/synthetic.hpp>

#include "host_support.hpp"
#include "test_support.hpp"

namespace holoscan::stereo_camera {
namespace {

using test::contains;
using test::invalid_message;

constexpr int kWidth = 80;
constexpr int kHeight = 64;
constexpr std::uint32_t kFrames = 3;

SourceConfig playback(const std::filesystem::path& folder, bool rgb, bool depth, bool loop) {
  return SourceConfig{.backend = Backend::kPlayback,
                      .rgb = rgb,
                      .depth = depth,
                      .playback = PlaybackOptions{.directory = folder, .loop = loop}};
}

void check_playback(const std::filesystem::path& folder) {
  for (const bool loop : {false, true}) {
    for (const bool extras : {false, true}) {
      auto source = create_source(playback(folder, extras, extras, loop));
      VERIFY(source->backend() == Backend::kPlayback);
      const SourceLayout& layout = source->layout();
      VERIFY(!layout.live && layout.rgb.has_value() == extras &&
             layout.depth.has_value() == extras);
      VERIFY(test::throws<std::logic_error>([&] { (void)source->calibration(); }));
      VERIFY(test::throws<std::logic_error>([&] { (void)source->grab(); }));
      source->open();
      VERIFY(source->calibration().camera.name == "synthetic");
      Image left;
      Image right;
      Image rgb;
      Image depth;
      const std::uint32_t total = loop ? 2 * kFrames + 1 : kFrames;
      for (std::uint32_t n = 0; n < total; ++n) {
        const auto frame = source->grab();
        VERIFY_MSG(frame.has_value(), "frame " + std::to_string(n));
        if (!frame) break;
        const std::uint32_t index = n % kFrames;
        render_synthetic_frame(kWidth, kHeight, index, left, right, &rgb, &depth);
        VERIFY(frame->index == index && frame->timestamp_ns == synthetic_timestamp_ns(index));
        VERIFY(frame->device_sequence == index + 1U && !frame->age_ns);
        VERIFY(same_pixels(frame->left, left.view()) && same_pixels(frame->right, right.view()));
        VERIFY(frame->rgb.has_value() == extras && frame->depth.has_value() == extras);
        if (extras) {
          VERIFY(same_pixels(*frame->rgb, rgb.view()) && same_pixels(*frame->depth, depth.view()));
        }
      }
      VERIFY(source->finished() == !loop);
      VERIFY(source->grab().has_value() == loop);
      // Opening again starts over.
      source->open();
      VERIFY(!source->finished() && source->grab()->index == 0);
      source->close();
      source->close();
    }
  }
}

void check_names() {
  for (const Backend backend :
       {Backend::kRealSense, Backend::kV4l2SideBySide, Backend::kPlayback}) {
    VERIFY(parse_backend(backend_name(backend)) == backend);
  }
  VERIFY(!parse_backend("auto") && !parse_backend("zed"));
  const auto backends = available_backends();
  VERIFY(backends.size() == (realsense_available() ? 3U : 2U));
  VERIFY(backends.back() == Backend::kPlayback);
}

void check_rejections(const std::filesystem::path& work) {
  const auto expect = [](const SourceConfig& config, const std::string& part) {
    const std::string message = invalid_message([&] { (void)create_source(config); });
    VERIFY_MSG(contains(message, part), "expected \"" + part + "\", got \"" + message + "\"");
  };
  const auto plain = work / "source_plain";
  std::filesystem::remove_all(plain);
  write_synthetic_recording(plain, kWidth, kHeight, 2, false, false);
  expect(playback(plain, true, false, false), "the recording has no rgb images");
  expect(playback(plain, false, true, false), "the recording has no depth images");
  expect(playback("", false, false, false), "playback: needs a recording directory");
  expect(playback(work / "none", false, false, false), "none: not a recording directory");

  // v4l2-sbs, checked without opening the device.
  const auto folder = test::fresh_folder(work, "v4l2");
  write_calibration(folder / "zed.json", synthetic_calibration(640, 360, false, false));
  SourceConfig v4l2{.backend = Backend::kV4l2SideBySide,
                    .v4l2 = V4l2SideBySideOptions{.device = "/dev/video-none",
                                                  .width = 1280,
                                                  .height = 360,
                                                  .calibration_file = folder / "zed.json"}};
  auto source = create_source(v4l2);
  VERIFY((source->layout().left == StreamLayout{640, 360, PixelFormat::kMono8}));
  VERIFY(source->layout().live && !source->layout().rgb && !source->layout().depth);
  const std::string open_error = invalid_message([&] { source->open(); });
  VERIFY_MSG(contains(open_error, "v4l2-sbs: cannot open /dev/video-none"), open_error);
  SourceConfig bad = v4l2;
  bad.v4l2.width = 2560;
  expect(bad, "zed.json: its eyes are 640x360, the camera's 1280x360");
  bad = v4l2;
  bad.v4l2.width = 1282;
  expect(bad, "a multiple of 4 for YUYV");
  bad = v4l2;
  bad.v4l2.calibration_file.clear();
  expect(bad, "v4l2-sbs: needs a calibration file");
  bad = v4l2;
  bad.depth = true;
  expect(bad, "gives no rgb or depth image");
  bad = v4l2;
  bad.v4l2.fps = 0;
  expect(bad, "fps must be 1..1000");
  bad = v4l2;
  bad.v4l2.output = PixelFormat::kMono16;
  expect(bad, "the eyes are mono8 or rgb8");

  SourceConfig realsense{.backend = Backend::kRealSense};
  if (realsense_available()) {
    realsense.realsense.width = 0;
    expect(realsense, "realsense: width and height must be 1..8192");
  } else {
    expect(realsense, "this build has no librealsense");
  }
}

}  // namespace
}  // namespace holoscan::stereo_camera

int main(int argc, char** argv) {
  namespace sc = holoscan::stereo_camera;
  const std::filesystem::path work = argc > 1 ? argv[1] : std::filesystem::current_path();
  const std::filesystem::path recording = work / "source_recording";
  std::filesystem::remove_all(recording);
  sc::write_synthetic_recording(recording, sc::kWidth, sc::kHeight, sc::kFrames, true, true);
  return sc::test::run({
      {"playback backend", [&] { sc::check_playback(recording); }},
      {"backend names", sc::check_names},
      {"rejected configurations", [&] { sc::check_rejections(work); }},
  });
}
