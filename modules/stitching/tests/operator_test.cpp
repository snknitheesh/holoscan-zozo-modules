// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StitchOp through the SDK's operator test harness: the surround rig's images
// from device and host memory, stitched as the Stitcher does; the descriptor and provenance it
// publishes; a one-camera rig; and unusable images and configurations refused.

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/stitching/stitch_op.hpp>
#include <holoscan/stitching/stitcher.hpp>
#include <holoscan/stitching/synthetic_world.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "gpu_support.hpp"
#include "test_support.hpp"

namespace holoscan::stitching {
namespace {

namespace schema = holoscan::schema;
using ImageSnapshot = holoscan::testing::Snapshot<schema::ImageT>;

const SyntheticWorld kWorld{};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

schema::ImageEncoding encoding_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kMono8:
      return schema::ImageEncoding_MONO8;
    case PixelFormat::kRgb8:
      return schema::ImageEncoding_RGB8;
    case PixelFormat::kRgba8:
      return schema::ImageEncoding_RGBA8;
    case PixelFormat::kBgr8:
      return schema::ImageEncoding_BGR8;
    case PixelFormat::kBgra8:
      return schema::ImageEncoding_BGRA8;
  }
  return schema::ImageEncoding_RGB8;
}

// An image message of `bytes` (rows `pitch` bytes apart), in host or device
// memory, its tensor [H, W, C] (or [H, W] for mono when `rank2`).
schema::ImageT message(const std::vector<std::uint8_t>& bytes, int width, int height,
                       std::size_t pitch, PixelFormat format, bool device, bool rank2 = false) {
  const int ch = channels(format);
  std::vector<std::int64_t> shape{height, width};
  std::vector<std::int64_t> strides{static_cast<std::int64_t>(pitch), ch};
  if (!rank2) {
    shape.push_back(ch);
    strides.push_back(1);
  }
  auto tensor = holoscan::make_tensor(
      shape, kUInt8, TensorByteStridesSpan(strides),
      device ? holoscan::MemoryStorageType::kDevice : holoscan::MemoryStorageType::kHost,
      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) {
    throw std::runtime_error(tensor.error().message);
  }
  auto destination = tensor->mutable_data();
  if (!destination) {
    throw std::runtime_error(destination.error().message);
  }
  // The last row ends at its last pixel: the tensor holds no padding after it.
  const std::size_t extent = pitch * (height - 1) + static_cast<std::size_t>(width) * ch;
  test::cuda_check(cudaMemcpy(*destination, bytes.data(), extent, cudaMemcpyDefault), "cudaMemcpy");
  schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding_of(format);
  image.data = std::make_shared<holoscan::Tensor>(std::move(*tensor));
  return image;
}

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

struct Run {
  bool ok{};
  std::string failure;
  std::vector<std::vector<std::uint8_t>> outputs;
  std::vector<schema::ImageT> descriptors;
  std::vector<holoscan::SampleMetadata> metadata;
};

// Runs StitchOp on `frames` (per frame, one image per camera); camera c's
// image of frame f has capture time capture_time(f) + skew[c].
Run run_op(const StitchConfig& config, const std::vector<std::vector<schema::ImageT>>& frames,
           bool device, const std::vector<std::chrono::nanoseconds>& skew = {}) {
  auto h = holoscan::testing::create_operator_test<StitchOp>(config);
  for (std::size_t c = 0; c < config.rig.cameras.size(); ++c) {
    std::vector<holoscan::testing::TestSample<schema::ImageT>> samples;
    for (std::uint32_t f = 0; f < frames.size(); ++f) {
      const auto offset = skew.empty() ? std::chrono::nanoseconds{0} : skew[c];
      auto sample = holoscan::testing::sample_at(frames[f][c], capture_time(f) + offset);
      sample.metadata.frame_id = 7U + f;
      samples.push_back(std::move(sample));
    }
    const std::string port = "image_" + std::to_string(c);
    h.push_samples(port, std::move(samples));
    if (device) {
      h.input_source_device(port, holoscan::DeviceId{0});
    }
  }
  h.output_device("stitched", holoscan::DeviceId{0});
  h.capture_snapshot<schema::ImageT>("stitched");
  const auto result = h.run();
  Run run;
  run.ok = result.ok();
  if (!run.ok) {
    run.failure = result.describe_failure();
    return run;
  }
  for (const ImageSnapshot& snapshot : result.snapshots<schema::ImageT>("stitched")) {
    const auto& tensor = snapshot.tensor(schema::ImageTensorFields::data);
    std::vector<std::uint8_t> bytes(tensor.bytes().size());
    std::memcpy(bytes.data(), tensor.bytes().data(), bytes.size());
    run.outputs.push_back(std::move(bytes));
    run.descriptors.push_back(snapshot.metadata());
    run.metadata.push_back(snapshot.sample_metadata());
  }
  return run;
}

// The rig's frames as messages, and what the Stitcher makes of them.
struct Expected {
  std::vector<std::vector<schema::ImageT>> frames;
  std::vector<std::vector<std::uint8_t>> stitched;
};

Expected expected_frames(const StitchConfig& config, PixelFormat format, bool device,
                         std::size_t pad, int count) {
  Expected e;
  Stitcher stitcher(config.rig, config.output, config.blend);
  for (int f = 0; f < count; ++f) {
    const test::RigFrame frame =
        test::render_frame(kWorld, config.rig, moving_pose(10U * f), format, {}, 2, pad);
    std::vector<schema::ImageT> images;
    for (std::size_t c = 0; c < config.rig.cameras.size(); ++c) {
      const Camera& camera = config.rig.cameras[c];
      images.push_back(message(frame.host[c], camera.width, camera.height, frame.images[c].pitch,
                               format, device));
    }
    e.frames.push_back(std::move(images));
    e.stitched.push_back(test::stitch_frame(stitcher, frame));
  }
  return e;
}

StitchConfig surround_config(PixelFormat output = PixelFormat::kRgba8) {
  StitchConfig config;
  config.rig = surround_rig(320, 180);
  config.output = ground_view(200, 160, 5.0);
  config.blend.output_format = output;
  config.blend.exposure_compensation = true;
  config.frame_id = "vehicle";
  return config;
}

void check_outputs(const std::string& what, const Run& run, const Expected& expected,
                   const StitchConfig& config) {
  VERIFY_MSG(run.ok, what + ": " + run.failure);
  VERIFY_MSG(run.outputs.size() == expected.stitched.size(), what + ": output count");
  const bool rgba = config.blend.output_format == PixelFormat::kRgba8;
  for (std::size_t f = 0; f < run.outputs.size() && f < expected.stitched.size(); ++f) {
    VERIFY_MSG(run.outputs[f] == expected.stitched[f],
               what + ": frame " + std::to_string(f) + " differs from the Stitcher's");
    const schema::ImageT& d = run.descriptors[f];
    VERIFY(d.width == config.output.width && d.height == config.output.height);
    VERIFY(d.encoding == (rgba ? schema::ImageEncoding_RGBA8 : schema::ImageEncoding_RGB8));
    VERIFY(d.header && d.header->frame_id == config.frame_id);
    VERIFY(d.header && d.header->capture_timestamp_ns == capture_time(f).count());
    VERIFY(run.metadata[f].capture_timestamp_ns == capture_time(f).count());
    VERIFY(run.metadata[f].frame_id == 7U + f);
  }
}

void device_images() {
  const StitchConfig config = surround_config();
  const Expected e = expected_frames(config, PixelFormat::kRgba8, true, 0, 3);
  check_outputs("device rgba8", run_op(config, e.frames, true), e, config);
}

void host_images() {
  // Host memory and RGB8 out: copied to the GPU each frame. (The harness
  // stages packed tensors only; the Stitcher's test covers padded rows.)
  const StitchConfig config = surround_config(PixelFormat::kRgb8);
  const Expected e = expected_frames(config, PixelFormat::kBgr8, false, 0, 2);
  check_outputs("host bgr8", run_op(config, e.frames, false), e, config);
  const Expected mono = expected_frames(config, PixelFormat::kMono8, false, 0, 1);
  check_outputs("host mono8", run_op(config, mono.frames, false), mono, config);
}

void capture_time_window() {
  // Images up to the window apart make one frame set, with the earliest time.
  StitchConfig config = surround_config();
  config.sync_window = std::chrono::milliseconds(5);
  const Expected e = expected_frames(config, PixelFormat::kRgba8, true, 0, 2);
  const std::vector<std::chrono::nanoseconds> skew{
      std::chrono::milliseconds(0), std::chrono::milliseconds(3), std::chrono::milliseconds(5),
      std::chrono::milliseconds(1)};
  const Run run = run_op(config, e.frames, true, skew);
  check_outputs("within the window", run, e, config);
}

void one_camera() {
  StitchConfig config;
  config.rig = ring_rig(1, 320, 180, 90.0, 0.0);
  config.output = wall_panorama(kWorld, Projection::kSpherical, 256, 128);
  config.output.panorama.horizontal_fov_deg = 120.0;
  const Expected e = expected_frames(config, PixelFormat::kRgb8, true, 0, 2);
  check_outputs("one camera", run_op(config, e.frames, true), e, config);
}

void rejects(const std::string& what, const StitchConfig& config,
             std::vector<schema::ImageT> images) {
  const Run run = run_op(config, {std::move(images)}, true);
  VERIFY_MSG(!run.ok, what + " is rejected");
  VERIFY_MSG(run.failure.find(what) != std::string::npos,
             "failure names '" + what + "': " + run.failure);
}

void rejected_inputs() {
  const StitchConfig config = surround_config();
  const Expected e = expected_frames(config, PixelFormat::kRgb8, true, 0, 1);
  {
    // An image of another size than its camera's.
    auto images = e.frames[0];
    const std::vector<std::uint8_t> small(64 * 3 * 48);
    images[2] = message(small, 64, 48, 64 * 3, PixelFormat::kRgb8, true);
    rejects("the camera's calibration is 320x180", config, images);
  }
  {
    auto images = e.frames[0];
    images[1].encoding = schema::ImageEncoding_MONO16;
    rejects("unsupported encoding MONO16", config, images);
  }
  {
    // A mono image as a 3-channel tensor (an RGBA8 one the SDK's image schema
    // already refuses before it is sent).
    auto images = e.frames[0];
    const std::vector<std::uint8_t> rgb(320 * 3 * 180);
    images[0] = message(rgb, 320, 180, 320 * 3, PixelFormat::kRgb8, true);
    images[0].encoding = schema::ImageEncoding_MONO8;
    rejects("the tensor must be uint8 [H, W, 1] or [H, W]", config, images);
  }
}

void rejected_configs() {
  StitchConfig good = surround_config();
  VERIFY(!test::throws<std::invalid_argument>([&] { StitchOp op(good); }));
  StitchConfig empty = good;
  empty.rig.cameras.clear();
  VERIFY(test::throws<std::invalid_argument>([&] { StitchOp op(empty); }));
  StitchConfig negative = good;
  negative.sync_window = std::chrono::nanoseconds(-1);
  VERIFY(test::throws<std::invalid_argument>([&] { StitchOp op(negative); }));
  StitchConfig mono_out = good;
  mono_out.blend.output_format = PixelFormat::kMono8;
  VERIFY(test::throws<std::invalid_argument>([&] { StitchOp op(mono_out); }));
  StitchConfig bad_view = good;
  bad_view.output.birds_eye.x_max = bad_view.output.birds_eye.x_min;
  VERIFY(test::throws<std::invalid_argument>([&] { StitchOp op(bad_view); }));
  StitchConfig twins = good;
  twins.rig.cameras[1].name = twins.rig.cameras[0].name;
  VERIFY(test::throws<std::invalid_argument>([&] { StitchOp op(twins); }));
}

}  // namespace
}  // namespace holoscan::stitching

int main() {
  namespace st = holoscan::stitching;
  if (!st::test::have_gpu()) {
    std::cout << "no CUDA device; skipped\n";
    return 77;
  }
  return st::test::run({
      {"device_images", st::device_images},
      {"host_images", st::host_images},
      {"capture_time_window", st::capture_time_window},
      {"one_camera", st::one_camera},
      {"rejected_inputs", st::rejected_inputs},
      {"rejected_configs", st::rejected_configs},
  });
}
