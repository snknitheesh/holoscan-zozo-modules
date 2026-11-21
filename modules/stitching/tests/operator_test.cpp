// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StitchOp through the SDK's operator test harness, each camera's calibration
// arriving as CameraInfo on its own port: the surround rig's images from
// device, pinned host and pageable host memory, stitched as the Stitcher
// does; the descriptor and provenance it publishes; a one-camera rig; camera
// info with every frame set; frame sets before every camera's calibration has
// arrived; a calibration that changes; and unusable images, camera info and
// configurations refused.

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
#include <holoscan/memory/memory_storage_type.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/stitching/stitch_op.hpp>
#include <holoscan/stitching/stitcher.hpp>
#include <holoscan/stitching/synthetic_world.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "gpu_support.hpp"
#include "test_support.hpp"

namespace holoscan::stitching {
namespace {

namespace schema = holoscan::schema;
namespace hm = holoscan::messages;
using ImageSnapshot = holoscan::testing::Snapshot<schema::ImageT>;
using holoscan::MemoryStorageType;
using std::chrono_literals::operator""ms;

const SyntheticWorld kWorld{};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
// Records each capture keeps.
constexpr std::size_t kCaptures = 8;

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

// An image message of `bytes` (rows `pitch` bytes apart), in device, pinned
// host (kHost) or pageable host (kSystem) memory, its tensor [H, W, C] (or
// [H, W] for mono when `rank2`).
schema::ImageT message(const std::vector<std::uint8_t>& bytes, int width, int height,
                       std::size_t pitch, PixelFormat format, MemoryStorageType memory,
                       bool rank2 = false) {
  const int ch = channels(format);
  std::vector<std::int64_t> shape{height, width};
  std::vector<std::int64_t> strides{static_cast<std::int64_t>(pitch), ch};
  if (!rank2) {
    shape.push_back(ch);
    strides.push_back(1);
  }
  auto tensor = holoscan::make_tensor(shape, kUInt8, TensorByteStridesSpan(strides), memory,
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

// Adds one sample of `value` at `time` on `port` to a staged frame.
template <typename T>
void add(holoscan::testing::TestFrame& frame, const std::string& port, T value,
         std::chrono::nanoseconds time, std::uint64_t frame_id = 0U) {
  auto sample = holoscan::testing::sample_at(std::move(value), time);
  sample.metadata.frame_id = frame_id;
  frame.fields.push_back(holoscan::testing::TestFrame::Field{
      port, holoscan::testing::staged_samples<T>(
                std::vector<holoscan::testing::TestSample<T>>{std::move(sample)})});
}

// The camera info of every camera of `rig`, captured at `time`.
holoscan::testing::TestFrame camera_info_at(const CameraRig& rig, std::chrono::nanoseconds time) {
  holoscan::testing::TestFrame frame;
  for (std::size_t c = 0; c < rig.cameras.size(); ++c) {
    add(frame, "camera_info_" + std::to_string(c), hm::to_message(rig.cameras[c].info), time);
  }
  return frame;
}

// The configuration of `rig`'s poses.
StitchConfig config_of(const CameraRig& rig, const OutputGeometry& output,
                       PixelFormat out = PixelFormat::kRgba8) {
  StitchConfig config;
  for (const Camera& camera : rig.cameras) config.camera_from_rig.push_back(camera.camera_from_rig);
  config.output = output;
  config.blend.output_format = out;
  config.blend.exposure_compensation = true;
  config.frame_id = "vehicle";
  return config;
}

// The rig's frames as messages, and what the Stitcher makes of them.
struct Expected {
  std::vector<std::vector<schema::ImageT>> frames;
  std::vector<std::vector<std::uint8_t>> stitched;
};

// The images are rendered through `rig`'s cameras and stitched with
// `stitch_rig`'s calibration when given, else `rig`'s.
Expected expected_frames(const CameraRig& rig, const StitchConfig& config, PixelFormat format,
                         MemoryStorageType memory, int count,
                         const CameraRig* stitch_rig = nullptr) {
  Expected e;
  Stitcher stitcher(stitch_rig != nullptr ? *stitch_rig : rig, config.output, config.blend);
  for (int f = 0; f < count; ++f) {
    const test::RigFrame frame =
        test::render_frame(kWorld, rig, moving_pose(10U * f), format, {}, 2, 0);
    std::vector<schema::ImageT> images;
    for (std::size_t c = 0; c < rig.cameras.size(); ++c) {
      const hm::CameraInfo& camera = rig.cameras[c].info;
      images.push_back(message(frame.host[c], static_cast<int>(camera.width),
                               static_cast<int>(camera.height), frame.images[c].pitch, format,
                               memory));
    }
    e.frames.push_back(std::move(images));
    e.stitched.push_back(test::stitch_frame(stitcher, frame));
  }
  return e;
}

// How a run sends the camera info.
struct InfoPlan {
  // kLatest: before the frame set of this index, and `changed` before
  // `change_at`.
  std::uint32_t first_frame{0};
  std::optional<CameraRig> changed;
  std::uint32_t change_at{0};
  // A camera whose camera info comes only before the frame set `late_at`.
  std::optional<std::size_t> late_camera;
  std::uint32_t late_at{0};
};

struct Run {
  bool ok{};
  std::string failure;
  std::vector<std::vector<std::uint8_t>> outputs;
  std::vector<schema::ImageT> descriptors;
  std::vector<holoscan::SampleMetadata> metadata;
};

// Runs StitchOp on `frames` (per frame, one image per camera); camera c's
// image of frame f has capture time capture_time(f) + skew[c]. The camera
// info comes as `plan` says, or with every frame set at its capture time for
// kEveryFrame.
Run run_op(const StitchConfig& config, const CameraRig& rig,
           const std::vector<std::vector<schema::ImageT>>& frames, bool device,
           const InfoPlan& plan = {}, const std::vector<std::chrono::nanoseconds>& skew = {}) {
  auto h = holoscan::testing::create_operator_test<StitchOp>(config);
  const bool every_frame = config.camera_info.timing == hm::CameraInfoTiming::kEveryFrame;
  for (std::uint32_t f = 0; f < frames.size(); ++f) {
    if (!every_frame && (f == plan.first_frame || (plan.changed && f == plan.change_at))) {
      const CameraRig& calibration = plan.changed && f >= plan.change_at ? *plan.changed : rig;
      holoscan::testing::TestFrame info = camera_info_at(calibration, capture_time(f) - 1ms);
      if (plan.late_camera && f == plan.first_frame) {
        info.fields.erase(info.fields.begin() + static_cast<std::ptrdiff_t>(*plan.late_camera));
      }
      h.push_frame(info);
    }
    if (!every_frame && plan.late_camera && f == plan.late_at) {
      holoscan::testing::TestFrame info;
      add(info, "camera_info_" + std::to_string(*plan.late_camera),
          hm::to_message(rig.cameras[*plan.late_camera].info), capture_time(f) - 1ms);
      h.push_frame(info);
    }
    holoscan::testing::TestFrame frame;
    for (std::size_t c = 0; c < rig.cameras.size(); ++c) {
      const auto offset = skew.empty() ? std::chrono::nanoseconds{0} : skew[c];
      add(frame, "image_" + std::to_string(c), frames[f][c], capture_time(f) + offset, 7U + f);
      if (every_frame) {
        add(frame, "camera_info_" + std::to_string(c), hm::to_message(rig.cameras[c].info),
            capture_time(f) + offset);
      }
    }
    h.push_frame(frame);
  }
  if (device) {
    for (std::size_t c = 0; c < rig.cameras.size(); ++c) {
      h.input_source_device("image_" + std::to_string(c), holoscan::DeviceId{0});
    }
  }
  h.output_device("stitched", holoscan::DeviceId{0});
  h.capture_snapshot<schema::ImageT>("stitched", kCaptures);
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

const CameraRig& surround() {
  static const CameraRig rig = surround_rig(320, 180);
  return rig;
}

StitchConfig surround_config(PixelFormat output = PixelFormat::kRgba8) {
  return config_of(surround(), ground_view(200, 160, 5.0), output);
}

// Checks that `run` gave `expected`'s frames from `first` on, in order.
void check_outputs(const std::string& what, const Run& run, const Expected& expected,
                   const StitchConfig& config, std::uint32_t first = 0) {
  VERIFY_MSG(run.ok, what + ": " + run.failure);
  VERIFY_MSG(run.outputs.size() == expected.stitched.size() - first,
             what + ": " + std::to_string(run.outputs.size()) + " outputs");
  const bool rgba = config.blend.output_format == PixelFormat::kRgba8;
  for (std::size_t i = 0; i < run.outputs.size() && first + i < expected.stitched.size(); ++i) {
    const auto f = static_cast<std::uint32_t>(first + i);
    VERIFY_MSG(run.outputs[i] == expected.stitched[f],
               what + ": frame " + std::to_string(f) + " differs from the Stitcher's");
    const schema::ImageT& d = run.descriptors[i];
    VERIFY(d.width == config.output.width && d.height == config.output.height);
    VERIFY(d.encoding == (rgba ? schema::ImageEncoding_RGBA8 : schema::ImageEncoding_RGB8));
    VERIFY(d.header && d.header->frame_id == config.frame_id);
    VERIFY(d.header && d.header->capture_timestamp_ns == capture_time(f).count());
    VERIFY(run.metadata[i].capture_timestamp_ns == capture_time(f).count());
    VERIFY(run.metadata[i].frame_id == 7U + f);
  }
}

void device_images() {
  const StitchConfig config = surround_config();
  const Expected e =
      expected_frames(surround(), config, PixelFormat::kRgba8, MemoryStorageType::kDevice, 3);
  check_outputs("device rgba8", run_op(config, surround(), e.frames, true), e, config);
}

void pinned_host_images() {
  // Pinned host memory and RGB8 out: copied to the GPU each frame.
  const StitchConfig config = surround_config(PixelFormat::kRgb8);
  const Expected e =
      expected_frames(surround(), config, PixelFormat::kBgr8, MemoryStorageType::kHost, 2);
  check_outputs("pinned bgr8", run_op(config, surround(), e.frames, false), e, config);
  const Expected mono =
      expected_frames(surround(), config, PixelFormat::kMono8, MemoryStorageType::kHost, 1);
  check_outputs("pinned mono8", run_op(config, surround(), mono.frames, false), mono, config);
}

void pageable_host_images() {
  // Pageable host memory: waited for, then copied (the harness stages packed
  // tensors only; the Stitcher's test covers padded rows).
  const StitchConfig config = surround_config();
  const Expected e =
      expected_frames(surround(), config, PixelFormat::kRgba8, MemoryStorageType::kSystem, 2);
  check_outputs("pageable rgba8", run_op(config, surround(), e.frames, false), e, config);
  const Expected mono =
      expected_frames(surround(), config, PixelFormat::kMono8, MemoryStorageType::kSystem, 1);
  check_outputs("pageable mono8", run_op(config, surround(), mono.frames, false), mono, config);
}

void capture_time_window() {
  // Images up to the window apart make one frame set, with the earliest time.
  StitchConfig config = surround_config();
  config.sync_window = 5ms;
  const Expected e =
      expected_frames(surround(), config, PixelFormat::kRgba8, MemoryStorageType::kDevice, 2);
  const std::vector<std::chrono::nanoseconds> skew{0ms, 3ms, 5ms, 1ms};
  check_outputs("within the window", run_op(config, surround(), e.frames, true, {}, skew), e,
                config);
  // A window of half the frame period or more could pair two frame sets.
  config.sync_window = 50ms;
  const Run wide = run_op(config, surround(), e.frames, true);
  VERIFY_MSG(!wide.ok, "a window of half the period is refused");
  VERIFY_MSG(wide.failure.find("within 2 x sync_window") != std::string::npos, wide.failure);
}

void one_camera() {
  const CameraRig rig = ring_rig(1, 320, 180, 90.0, 0.0);
  OutputGeometry output = wall_panorama(kWorld, Projection::kSpherical, 256, 128);
  output.panorama.horizontal_fov_deg = 120.0;
  const StitchConfig config = config_of(rig, output, PixelFormat::kRgb8);
  const Expected e =
      expected_frames(rig, config, PixelFormat::kRgb8, MemoryStorageType::kDevice, 2);
  check_outputs("one camera", run_op(config, rig, e.frames, true), e, config);
}

void camera_info_every_frame() {
  // Each frame set with its cameras' info at its own capture time.
  StitchConfig config = surround_config();
  config.camera_info.timing = hm::CameraInfoTiming::kEveryFrame;
  const Expected e =
      expected_frames(surround(), config, PixelFormat::kRgba8, MemoryStorageType::kDevice, 3);
  check_outputs("every frame", run_op(config, surround(), e.frames, true), e, config);
  // And within a capture time window.
  config.sync_window = 5ms;
  const std::vector<std::chrono::nanoseconds> skew{0ms, 3ms, 5ms, 1ms};
  check_outputs("every frame, within the window",
                run_op(config, surround(), e.frames, true, {}, skew), e, config);
}

void frames_before_camera_info() {
  // The frame sets before the camera info gives nothing, the ones after all
  // stitch with it.
  const StitchConfig config = surround_config();
  const Expected e =
      expected_frames(surround(), config, PixelFormat::kRgba8, MemoryStorageType::kDevice, 3);
  check_outputs("from the second frame",
                run_op(config, surround(), e.frames, true, InfoPlan{.first_frame = 1}), e, config,
                1);
  // One camera's camera info comes later than the others': the frame sets
  // wait for it.
  check_outputs(
      "from the third frame",
      run_op(config, surround(), e.frames, true, InfoPlan{.late_camera = 2, .late_at = 2}), e,
      config, 2);
}

void calibration_change() {
  // A new calibration for one camera takes effect from the next frame set:
  // the tables are built again.
  CameraRig changed = surround();
  changed.cameras[1].info.fx *= 1.05;
  changed.cameras[1].info.cx += 4.0;
  const StitchConfig config = surround_config();
  const Expected before =
      expected_frames(surround(), config, PixelFormat::kRgba8, MemoryStorageType::kDevice, 3);
  const Expected after = expected_frames(surround(), config, PixelFormat::kRgba8,
                                         MemoryStorageType::kDevice, 3, &changed);
  VERIFY(before.stitched[2] != after.stitched[2]);
  const Run run =
      run_op(config, surround(), before.frames, true, InfoPlan{.changed = changed, .change_at = 2});
  VERIFY_MSG(run.ok, run.failure);
  VERIFY_MSG(run.outputs.size() == 3, std::to_string(run.outputs.size()) + " outputs");
  if (run.outputs.size() != 3) return;
  VERIFY(run.outputs[0] == before.stitched[0]);
  VERIFY(run.outputs[1] == before.stitched[1]);
  VERIFY_MSG(run.outputs[2] == after.stitched[2], "the third frame uses the new calibration");
}

void rejects(const std::string& what, const Run& run) {
  VERIFY_MSG(!run.ok, what + " is rejected");
  VERIFY_MSG(run.failure.find(what) != std::string::npos,
             "failure names '" + what + "': " + run.failure);
}

void rejected_inputs() {
  const StitchConfig config = surround_config();
  const Expected e =
      expected_frames(surround(), config, PixelFormat::kRgb8, MemoryStorageType::kDevice, 1);
  {
    // An image of another size than its camera's.
    auto frames = e.frames;
    const std::vector<std::uint8_t> small(64 * 3 * 48);
    frames[0][2] = message(small, 64, 48, 64 * 3, PixelFormat::kRgb8, MemoryStorageType::kDevice);
    rejects("the camera's calibration is 320x180", run_op(config, surround(), frames, true));
  }
  {
    auto frames = e.frames;
    frames[0][1].encoding = schema::ImageEncoding_MONO16;
    rejects("unsupported encoding MONO16", run_op(config, surround(), frames, true));
  }
  {
    // A mono image as a 3-channel tensor (an RGBA8 one the SDK's image schema
    // already refuses before it is sent).
    auto frames = e.frames;
    const std::vector<std::uint8_t> rgb(320 * 3 * 180);
    frames[0][0] = message(rgb, 320, 180, 320 * 3, PixelFormat::kRgb8, MemoryStorageType::kDevice);
    frames[0][0].encoding = schema::ImageEncoding_MONO8;
    rejects("the tensor must be uint8 [H, W, 1] or [H, W]",
            run_op(config, surround(), frames, true));
  }
  {
    // A camera info the stitcher cannot model.
    CameraRig skewed = surround();
    skewed.cameras[1].info.skew = 0.5;
    rejects("a camera with skew is not supported", run_op(config, skewed, e.frames, true));
  }
}

void rejected_configs() {
  const StitchConfig good = surround_config();
  VERIFY(!test::throws<std::invalid_argument>([&] { StitchOp op(good); }));
  StitchConfig empty = good;
  empty.camera_from_rig.clear();
  VERIFY(test::throws<std::invalid_argument>([&] { StitchOp op(empty); }));
  StitchConfig nine = good;
  nine.camera_from_rig.resize(9);
  VERIFY(test::throws<std::invalid_argument>([&] { StitchOp op(nine); }));
  StitchConfig reflected = good;
  reflected.camera_from_rig[1].rotation = {1, 0, 0, 0, 1, 0, 0, 0, -1};
  VERIFY(test::throws<std::invalid_argument>([&] { StitchOp op(reflected); }));
  StitchConfig negative = good;
  negative.sync_window = std::chrono::nanoseconds(-1);
  VERIFY(test::throws<std::invalid_argument>([&] { StitchOp op(negative); }));
  StitchConfig ageless = good;
  ageless.camera_info.max_age = std::chrono::nanoseconds(0);
  VERIFY(test::throws<std::invalid_argument>([&] { StitchOp op(ageless); }));
  StitchConfig mono_out = good;
  mono_out.blend.output_format = PixelFormat::kMono8;
  VERIFY(test::throws<std::invalid_argument>([&] { StitchOp op(mono_out); }));
  StitchConfig bad_view = good;
  bad_view.output.birds_eye.x_max = bad_view.output.birds_eye.x_min;
  VERIFY(test::throws<std::invalid_argument>([&] { StitchOp op(bad_view); }));
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
      {"pinned_host_images", st::pinned_host_images},
      {"pageable_host_images", st::pageable_host_images},
      {"capture_time_window", st::capture_time_window},
      {"one_camera", st::one_camera},
      {"camera_info_every_frame", st::camera_info_every_frame},
      {"frames_before_camera_info", st::frames_before_camera_info},
      {"calibration_change", st::calibration_change},
      {"rejected_inputs", st::rejected_inputs},
      {"rejected_configs", st::rejected_configs},
  });
}
