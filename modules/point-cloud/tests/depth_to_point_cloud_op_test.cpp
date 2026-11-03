// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// DepthToPointCloudOp on its own, through the SDK's operator test harness:
// float32 and uint16 depth from host and device memory with camera info,
// aligned color in every encoding with camera info sent with every frame and
// changing from frame to frame, every point and color checked against the
// synthetic camera; the cloud message's header, layout and provenance; frames
// before the first camera info and frames whose camera info has another
// capture time; and unusable cameras, inputs and configurations rejected.
// Through graphs: CameraInfoSourceOp reading a camera_info file, how host
// input is held until its copy is done, and misaligned data rejected.

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
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
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/point_cloud/depth_to_point_cloud_op.hpp>
#include <holoscan/point_cloud/synthetic_depth.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "host_input.hpp"
#include "tensor_support.hpp"
#include "test_support.hpp"

namespace holoscan::point_cloud {
namespace {

namespace hm = holoscan::messages;
using test::make;
using test::stall;
using test::TestMemory;
using test::wrap;

using std::chrono_literals::operator""ms;
using CloudSnapshot = holoscan::testing::Snapshot<hm::schema::PointCloudT>;

constexpr int kWidth = 96;
constexpr int kHeight = 72;
// Clouds a test captures at most: the harness gives each capture a pool slot
// per tensor, and a channel holds at most 4096 buffers, so the default 1024
// clouds of four tensors would not fit.
constexpr std::size_t kCaptures = 16;

holoscan::schema::ImageT image(holoscan::Tensor tensor, int width, int height,
                               holoscan::schema::ImageEncoding encoding) {
  holoscan::schema::ImageT result;
  result.width = width;
  result.height = height;
  result.encoding = encoding;
  result.data = std::make_shared<holoscan::Tensor>(std::move(tensor));
  return result;
}

holoscan::schema::ImageT depth_image(const SyntheticFrame& f, bool millimeters, bool device) {
  const std::size_t n = static_cast<std::size_t>(f.width) * f.height;
  if (millimeters) {
    std::vector<std::uint16_t> mm(n);
    for (int v = 0; v < f.height; ++v)
      for (int u = 0; u < f.width; ++u) mm[v * f.width + u] = synthetic_depth_mm(f, u, v);
    return image(make({f.height, f.width}, {kDLUInt, 16, 1}, mm.data(), n * 2, device), f.width,
                 f.height, holoscan::schema::ImageEncoding_MONO16);
  }
  std::vector<float> m(n);
  for (int v = 0; v < f.height; ++v)
    for (int u = 0; u < f.width; ++u) m[v * f.width + u] = synthetic_depth(f, u, v);
  return image(make({f.height, f.width, 1}, {kDLFloat, 32, 1}, m.data(), n * 4, device), f.width,
               f.height, holoscan::schema::ImageEncoding_MONO32F);
}

// The synthetic camera's color in `encoding`: RGB8, BGR8, RGBA8 or BGRA8.
holoscan::schema::ImageT color_image(const SyntheticFrame& f,
                                     holoscan::schema::ImageEncoding encoding, bool device) {
  const bool bgr = encoding == holoscan::schema::ImageEncoding_BGR8 ||
                   encoding == holoscan::schema::ImageEncoding_BGRA8;
  const int channels = encoding == holoscan::schema::ImageEncoding_RGBA8 ||
                               encoding == holoscan::schema::ImageEncoding_BGRA8
                           ? 4
                           : 3;
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(f.width) * f.height * channels);
  for (int v = 0; v < f.height; ++v) {
    for (int u = 0; u < f.width; ++u) {
      std::uint8_t rgb[3];
      synthetic_color(f, u, v, rgb);
      std::uint8_t* p = pixels.data() + (static_cast<std::size_t>(v) * f.width + u) * channels;
      p[0] = bgr ? rgb[2] : rgb[0];
      p[1] = rgb[1];
      p[2] = bgr ? rgb[0] : rgb[2];
      if (channels == 4) {
        p[3] = 255;
      }
    }
  }
  return image(
      make({f.height, f.width, channels}, {kDLUInt, 8, 1}, pixels.data(), pixels.size(), device),
      f.width, f.height, encoding);
}

// The camera info of frame `f`'s camera, as a message.
hm::schema::CameraInfoT camera_info(const SyntheticFrame& f) {
  return hm::to_message(synthetic_camera_info(f, "depth_optical"));
}

// The bytes of one present field of a captured cloud.
template <typename T>
std::vector<T> field(const CloudSnapshot& cloud,
                     holoscan::TensorFieldSelector<hm::schema::PointCloudT> selector) {
  const auto& tensor = cloud.tensor(selector);
  std::vector<T> values(tensor.bytes().size() / sizeof(T));
  std::memcpy(values.data(), tensor.bytes().data(), values.size() * sizeof(T));
  return values;
}

// Checks a captured cloud against frame `f`: an organized [H, W, 3] cloud,
// NaN where the depth is not valid, colors when `colors`, absent normals and
// intensities, not counted.
void check_cloud(const CloudSnapshot& cloud, bool colors, const SyntheticFrame& f, bool millimeters,
                 const DepthToPointCloudConfig& config) {
  const hm::schema::PointCloudT metadata = cloud.metadata();
  VERIFY(metadata.organized);
  VERIFY(metadata.valid_count == -1);
  const std::vector<std::int64_t> shape{f.height, f.width, 3};
  VERIFY(cloud.tensor(hm::PointCloudFields::points).shape() == shape);
  VERIFY(cloud.tensor(hm::PointCloudFields::colors).shape() ==
         (colors ? shape : std::vector<std::int64_t>{0}));
  VERIFY(cloud.tensor(hm::PointCloudFields::normals).shape() == std::vector<std::int64_t>{0});
  VERIFY(cloud.tensor(hm::PointCloudFields::intensity).shape() == std::vector<std::int64_t>{0});
  const std::vector<float> xyz = field<float>(cloud, hm::PointCloudFields::points);
  VERIFY(xyz.size() == static_cast<std::size_t>(f.width) * f.height * 3);
  if (xyz.size() != static_cast<std::size_t>(f.width) * f.height * 3) return;
  std::vector<std::uint8_t> rgb;
  if (colors) {
    rgb = field<std::uint8_t>(cloud, hm::PointCloudFields::colors);
    VERIFY(rgb.size() == xyz.size());
    if (rgb.size() != xyz.size()) return;
  }
  int wrong = 0;
  int wrong_colors = 0;
  for (int v = 0; v < f.height; ++v) {
    for (int u = 0; u < f.width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * f.width + u;
      const float z = millimeters ? synthetic_depth_mm(f, u, v) * config.depth_scale
                                  : synthetic_depth(f, u, v) * config.depth_scale;
      const bool valid = z > 0 && z >= config.depth_min && z <= config.depth_max;
      const float* p = xyz.data() + 3 * i;
      if (!valid) {
        wrong += !(std::isnan(p[0]) && std::isnan(p[1]) && std::isnan(p[2]));
      } else {
        const CameraIntrinsics& k = f.intrinsics;
        wrong +=
            std::hypot(p[0] - (u - k.cx) * z / k.fx, p[1] - (v - k.cy) * z / k.fy, p[2] - z) > 1e-4;
      }
      if (colors) {
        std::uint8_t want[3];
        synthetic_color(f, u, v, want);
        const std::uint8_t* got = rgb.data() + 3 * i;
        wrong_colors += got[0] != want[0] || got[1] != want[1] || got[2] != want[2];
      }
    }
  }
  VERIFY_MSG(wrong == 0, std::to_string(wrong) + " points differ from the truth");
  VERIFY_MSG(wrong_colors == 0, std::to_string(wrong_colors) + " colors differ from the truth");
}

DepthToPointCloudConfig base_config(bool millimeters) {
  DepthToPointCloudConfig config;
  config.depth_scale = millimeters ? 0.001F : 1.0F;
  config.depth_min = 0.1F;
  config.depth_max = 10.0F;
  config.max_width = kWidth;
  config.max_height = kHeight;
  return config;
}

DepthToPointCloudConfig every_frame_config() {
  DepthToPointCloudConfig config = base_config(true);
  config.camera_info.timing = hm::CameraInfoTiming::kEveryFrame;
  return config;
}

// The capture time of frame `frame` in the tests' frames.
std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

// Checks that a cloud carries frame `frame`'s capture time and `frame_id`, in
// its sample and in its header with the camera info's frame.
void check_provenance(const CloudSnapshot& cloud, std::uint32_t frame, std::uint64_t frame_id) {
  const holoscan::SampleMetadata& metadata = cloud.sample_metadata();
  VERIFY_MSG(metadata.capture_timestamp_ns == capture_time(frame).count(),
             "frame " + std::to_string(frame) + " carries capture time " +
                 std::to_string(metadata.capture_timestamp_ns));
  VERIFY_MSG(
      metadata.frame_id == frame_id,
      "frame " + std::to_string(frame) + " carries frame id " + std::to_string(metadata.frame_id));
  const hm::schema::PointCloudT message = cloud.metadata();
  VERIFY(message.header != nullptr);
  if (message.header == nullptr) return;
  VERIFY(message.header->frame_id == "depth_optical");
  VERIFY(message.header->capture_timestamp_ns == capture_time(frame).count());
}

// Depth with camera info at the same capture time, without color.
void check_depth_only(bool millimeters, bool device) {
  const DepthToPointCloudConfig config = base_config(millimeters);
  auto h = holoscan::testing::create_operator_test<DepthToPointCloudOp>(config);
  const SyntheticFrame first = synthetic_frame(kWidth, kHeight, 0, false);
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, false);
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame), holoscan::testing::input("camera_info", camera_info(first)),
        holoscan::testing::input("depth", depth_image(f, millimeters, device))));
  }
  if (device) {
    h.input_source_device("depth", holoscan::DeviceId{0});
  }
  h.output_device("point_cloud", holoscan::DeviceId{0});
  h.capture_snapshot<hm::schema::PointCloudT>("point_cloud", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto clouds = result.snapshots<hm::schema::PointCloudT>("point_cloud");
  VERIFY_MSG(clouds.size() == 2, std::to_string(clouds.size()) + " clouds from 2 frames");
  for (std::size_t frame = 0; frame < clouds.size(); ++frame) {
    const auto number = static_cast<std::uint32_t>(frame);
    check_cloud(clouds[frame], false, synthetic_frame(kWidth, kHeight, number, false), millimeters,
                config);
    check_provenance(clouds[frame], number, 0U);
  }
}

// Color, and camera info sent with every frame, its intrinsics changing from
// frame to frame.
void check_color_and_changing_camera(holoscan::schema::ImageEncoding encoding, bool device) {
  DepthToPointCloudConfig config = every_frame_config();
  config.color = true;
  auto h = holoscan::testing::create_operator_test<DepthToPointCloudOp>(config);
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, true);
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame), holoscan::testing::input("depth", depth_image(f, true, device)),
        holoscan::testing::input("color", color_image(f, encoding, device)),
        holoscan::testing::input("camera_info", camera_info(f))));
  }
  if (device) {
    h.input_source_device("depth", holoscan::DeviceId{0});
    h.input_source_device("color", holoscan::DeviceId{0});
  }
  h.output_device("point_cloud", holoscan::DeviceId{0});
  h.capture_snapshot<hm::schema::PointCloudT>("point_cloud", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto clouds = result.snapshots<hm::schema::PointCloudT>("point_cloud");
  VERIFY(clouds.size() == 3);
  for (std::size_t frame = 0; frame < clouds.size(); ++frame) {
    const auto number = static_cast<std::uint32_t>(frame);
    check_cloud(clouds[frame], true, synthetic_frame(kWidth, kHeight, number, true), true, config);
    check_provenance(clouds[frame], number, 0U);
  }
}

// With camera info at its own rate, a frame that comes before the first one
// gives no cloud; the frames after it all do, with the newest camera info.
void check_frames_before_camera_info() {
  const DepthToPointCloudConfig config = base_config(true);
  auto h = holoscan::testing::create_operator_test<DepthToPointCloudOp>(config);
  const SyntheticFrame first = synthetic_frame(kWidth, kHeight, 0, false);
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, false);
    if (frame == 1) {
      h.push_frame(holoscan::testing::frame_at(
          capture_time(frame) - 1ms, holoscan::testing::input("camera_info", camera_info(first))));
    }
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame), holoscan::testing::input("depth", depth_image(f, true, false))));
  }
  h.output_device("point_cloud", holoscan::DeviceId{0});
  h.capture_snapshot<hm::schema::PointCloudT>("point_cloud", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto clouds = result.snapshots<hm::schema::PointCloudT>("point_cloud");
  VERIFY_MSG(clouds.size() == 2, std::to_string(clouds.size()) +
                                     " clouds from 2 frames after "
                                     "the camera info");
  for (std::size_t i = 0; i < clouds.size(); ++i) {
    const auto number = static_cast<std::uint32_t>(i + 1);
    check_cloud(clouds[i], false, synthetic_frame(kWidth, kHeight, number, false), true, config);
    check_provenance(clouds[i], number, 0U);
  }
}

// Camera info sent with every frame pairs with it by exact capture time: a
// frame whose camera info carries another time gives no cloud, while the
// frames around it still do.
void check_unmatched_camera_info() {
  const DepthToPointCloudConfig config = every_frame_config();
  auto h = holoscan::testing::create_operator_test<DepthToPointCloudOp>(config);
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, true);
    // Frame 1's camera info is a millisecond late.
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame), holoscan::testing::input("depth", depth_image(f, true, false))));
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame) + std::chrono::milliseconds(frame == 1 ? 1 : 0),
        holoscan::testing::input("camera_info", camera_info(f))));
  }
  h.output_device("point_cloud", holoscan::DeviceId{0});
  h.capture_snapshot<hm::schema::PointCloudT>("point_cloud", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto clouds = result.snapshots<hm::schema::PointCloudT>("point_cloud");
  VERIFY_MSG(clouds.size() == 2, std::to_string(clouds.size()) + " clouds from 2 matched frames");
  const std::array<std::uint32_t, 2> matched{0U, 2U};
  for (std::size_t i = 0; i < clouds.size() && i < matched.size(); ++i) {
    check_cloud(clouds[i], false, synthetic_frame(kWidth, kHeight, matched[i], true), true, config);
    check_provenance(clouds[i], matched[i], 0U);
  }
}

// A cloud carries its depth image's frame id and device sequence.
void check_frame_ids() {
  const DepthToPointCloudConfig config = base_config(false);
  auto h = holoscan::testing::create_operator_test<DepthToPointCloudOp>(config);
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 0, false);
  holoscan::schema::ImageT depth = depth_image(f, false, false);
  depth.header = std::make_shared<holoscan::schema::HeaderT>();
  depth.header->device_sequence = 1234U;
  holoscan::testing::TestFrame staged = holoscan::testing::frame_at(
      capture_time(0), holoscan::testing::input("camera_info", camera_info(f)),
      holoscan::testing::input("depth", depth));
  h.push_frame(staged);
  h.output_device("point_cloud", holoscan::DeviceId{0});
  h.capture_snapshot<hm::schema::PointCloudT>("point_cloud", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto clouds = result.snapshots<hm::schema::PointCloudT>("point_cloud");
  VERIFY(clouds.size() == 1);
  if (clouds.empty()) return;
  const hm::schema::PointCloudT message = clouds[0].metadata();
  VERIFY(message.header != nullptr && message.header->device_sequence == 1234U);
}

// Host input is held until the stream that copies it is done: by its allocator
// when that can wait for a stream, without blocking; otherwise by waiting.
void check_host_input_held() {
  cudaStream_t stream{};
  VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  {
    auto memory = std::make_shared<TestMemory>(64, false, true);
    VERIFY(stall(stream));
    VERIFY(detail::hold_until_done(wrap(memory, 0, {64}, {kDLUInt, 8, 1}, false), stream, "test")
               .has_value());
    VERIFY_MSG(memory->streams() == 1, "the allocator is asked to wait for the stream");
    VERIFY_MSG(cudaStreamQuery(stream) == cudaErrorNotReady, "the host does not wait");
    VERIFY(cudaStreamSynchronize(stream) == cudaSuccess);
  }
  {
    auto memory = std::make_shared<TestMemory>(64, false, false);
    VERIFY(stall(stream));
    VERIFY(detail::hold_until_done(wrap(memory, 0, {64}, {kDLUInt, 8, 1}, false), stream, "test")
               .has_value());
    VERIFY_MSG(cudaStreamQuery(stream) == cudaSuccess,
               "the stream is done when the allocator cannot wait for it");
  }
  VERIFY(cudaStreamDestroy(stream) == cudaSuccess);
}

// The test harness copies staged inputs into its own storage, so a tensor whose
// memory the test controls reaches the operator only through a graph: this
// source sends the given frames as they are, one per tick, each with its
// color image when there are any, and with its camera info unless a separate
// camera info source is used.
struct GivenFramesConfig {
  std::vector<holoscan::schema::ImageT> depth;
  std::vector<holoscan::schema::ImageT> color;
  bool camera_info{true};
};

class GivenFramesSourceOp final : public holoscan::Operator<> {
 public:
  explicit GivenFramesSourceOp(GivenFramesConfig config) : config_(std::move(config)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    const holoscan::PayloadOptions image_payload{
        .schema = holoscan::schema_identity<holoscan::schema::ImageT>()};
    spec.output(depth, "depth").payload_options(image_payload);
    if (!config_.color.empty()) {
      spec.output(color, "color").payload_options(image_payload);
    }
    if (config_.camera_info) {
      spec.output(camera_info, "camera_info");
    }
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(5)});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_ >= config_.depth.size()) {
      return {};
    }
    // A frame keeps its capture time until all its parts are out.
    if (!capture_) {
      capture_ = context.activation_time();
    }
    const holoscan::EmitOptions options{.capture_time = *capture_, .frame_id = next_ + 1U};
    // Sends one part unless sent already; false when the consumer is busy.
    const auto send = [&](bool& sent, auto&& emit) -> holoscan::expected<bool, holoscan::Error> {
      if (sent) {
        return true;
      }
      auto result = emit();
      if (!result) {
        if (result.error().code == holoscan::ErrorCode::kBackpressured) {
          return false;
        }
        return holoscan::make_unexpected(std::move(result).error());
      }
      sent = true;
      return true;
    };
    const std::size_t i = next_;
    const SyntheticFrame f = synthetic_frame(config_.depth[i].width, config_.depth[i].height,
                                             static_cast<std::uint32_t>(i), false);
    for (auto part : {0, 1, 2}) {
      holoscan::expected<bool, holoscan::Error> done = true;
      if (part == 0 && config_.camera_info) {
        done = send(sent_camera_info_, [&] {
          hm::CameraInfo value = synthetic_camera_info(f, "depth_optical");
          value.header.capture_timestamp_ns = capture_->timestamp_ns;
          return camera_info.emit(hm::to_message(value), options);
        });
      } else if (part == 1 && !config_.color.empty()) {
        done = send(sent_color_, [&] { return color.emit(config_.color[i], options); });
      } else if (part == 2) {
        done = send(sent_depth_, [&] { return depth.emit(config_.depth[i], options); });
      }
      if (!done) {
        return holoscan::make_unexpected(std::move(done).error());
      }
      if (!*done) {
        return {};
      }
    }
    ++next_;
    capture_.reset();
    sent_camera_info_ = sent_color_ = sent_depth_ = false;
    return {};
  }

  holoscan::Output<holoscan::schema::ImageT> depth;
  holoscan::Output<holoscan::schema::ImageT> color;
  holoscan::Output<hm::schema::CameraInfoT> camera_info;

 private:
  GivenFramesConfig config_;
  std::uint64_t next_{};
  std::optional<holoscan::TimePoint> capture_;
  bool sent_camera_info_{};
  bool sent_color_{};
  bool sent_depth_{};
};

// What the sink below saw.
struct CloudCount {
  std::uint32_t clouds{};
  std::uint64_t first_frame_id{};
  std::uint64_t last_frame_id{};
  std::string frame_id;
};

// Counts the clouds it receives, and completes the run at the cloud of frame
// id `last_frame_id`.
class CloudCountOp final : public holoscan::Operator<> {
 public:
  CloudCountOp(std::shared_ptr<CloudCount> count, std::uint64_t last_frame_id)
      : count_(std::move(count)), last_frame_id_(last_frame_id) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(point_cloud, "point_cloud").queue_depth(2U);
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{point_cloud});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto cloud = point_cloud.receive();
    if (!cloud) {
      return holoscan::make_unexpected(std::move(cloud).error());
    }
    if (auto checked = hm::check_point_cloud(cloud->data); !checked) {
      return checked;
    }
    if (count_->clouds++ == 0) {
      count_->first_frame_id = cloud->metadata.frame_id;
    }
    count_->last_frame_id = cloud->metadata.frame_id;
    count_->frame_id = cloud->data.header ? cloud->data.header->frame_id : std::string{};
    if (cloud->metadata.frame_id == last_frame_id_) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<hm::schema::PointCloudT> point_cloud;

 private:
  std::shared_ptr<CloudCount> count_;
  std::uint64_t last_frame_id_{};
};

// Runs `frames` through the operator into a counting sink, with the camera
// info of the source or, given a file, of a CameraInfoSourceOp reading it.
// Returns what the sink saw, and in `failure` the run's diagnostics when it
// did not complete.
std::shared_ptr<CloudCount> run_graph(const DepthToPointCloudConfig& config,
                                      GivenFramesConfig frames, std::string& failure,
                                      const std::filesystem::path& camera_info_file = {}) {
  holoscan::Graph graph{"depth_to_point_cloud_test"};
  const auto count = static_cast<std::uint64_t>(frames.depth.size());
  frames.camera_info = camera_info_file.empty();
  auto source = graph.op<GivenFramesSourceOp>("source", std::move(frames));
  auto cloud = graph.op<DepthToPointCloudOp>("deproject", config);
  auto report = std::make_shared<CloudCount>();
  auto sink = graph.op<CloudCountOp>("count", report, count);
  graph.add_flow(source->depth, cloud->depth);
  if (config.color) {
    graph.add_flow(source->color, cloud->color);
  }
  if (camera_info_file.empty()) {
    graph.add_flow(source->camera_info, cloud->camera_info);
  } else {
    auto info = graph.op<hm::CameraInfoSourceOp>(
        "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = camera_info_file,
                                                  .period = std::chrono::milliseconds(20)});
    graph.add_flow(info->camera_info, cloud->camera_info, hm::latest_camera_info_connection());
  }
  graph.add_flow(cloud->point_cloud, sink->point_cloud);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "deproject", .output_port = "point_cloud", .device = holoscan::DeviceId{0}});
  const auto plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    failure = holoscan::format_diagnostics(plan.diagnostics(), plan.graph());
    return report;
  }
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (session.is_running()) {
    session.request_stop();
  }
  session.wait();
  if (session.termination() != holoscan::RunTermination::kRequestedCompleted) {
    failure = "run did not complete: " +
              holoscan::format_diagnostics(session.diagnostics(), plan.graph());
  }
  return report;
}

// The synthetic frame's depth in millimeters, at `offset` bytes into new
// pinned host or device memory.
std::pair<holoscan::schema::ImageT, std::shared_ptr<TestMemory>> given_depth(
    const SyntheticFrame& f, bool device, std::size_t offset = 0) {
  std::vector<std::uint16_t> mm(static_cast<std::size_t>(f.width) * f.height);
  for (int v = 0; v < f.height; ++v)
    for (int u = 0; u < f.width; ++u) mm[v * f.width + u] = synthetic_depth_mm(f, u, v);
  // Pinned depth is host-written: the runtime does not order its release.
  auto memory = std::make_shared<TestMemory>(mm.size() * 2U + offset, device, true, device);
  if (cudaMemcpy(memory->base() + offset, mm.data(), mm.size() * 2U, cudaMemcpyDefault) !=
      cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return {image(wrap(memory, offset, {f.height, f.width}, {kDLUInt, 16, 1}, device), f.width,
                f.height, holoscan::schema::ImageEncoding_MONO16),
          memory};
}

// A camera_info file of the synthetic camera, removed when it goes.
class CameraInfoFile {
 public:
  explicit CameraInfoFile(const hm::CameraInfo& info)
      : path_(std::filesystem::temp_directory_path() /
              ("point_cloud_test_" + std::to_string(::getpid()) + "_" + std::to_string(next_id()) +
               ".yaml")) {
    hm::write_camera_info_yaml(info, path_);
  }
  ~CameraInfoFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }
  CameraInfoFile(const CameraInfoFile&) = delete;
  CameraInfoFile& operator=(const CameraInfoFile&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

 private:
  static int next_id() {
    static int id = 0;
    return id++;
  }
  std::filesystem::path path_;
};

// CameraInfoSourceOp reads the camera from its file and sends it at its own
// rate; every frame from the first camera info on gives a cloud in the file's
// camera frame.
void check_camera_info_source() {
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 0, false);
  const CameraInfoFile file(synthetic_camera_info(f, "depth_from_file"));
  GivenFramesConfig frames;
  for (std::uint32_t frame = 0; frame < 10; ++frame) {
    frames.depth.push_back(depth_image(synthetic_frame(kWidth, kHeight, frame, false), true, true));
  }
  std::string failure;
  const auto report = run_graph(base_config(true), std::move(frames), failure, file.path());
  VERIFY_MSG(failure.empty(), failure);
  VERIFY_MSG(report->clouds >= 1 && report->clouds + report->first_frame_id - 1U == 10U &&
                 report->last_frame_id == 10U,
             std::to_string(report->clouds) + " clouds from frame " +
                 std::to_string(report->first_frame_id) + " to " +
                 std::to_string(report->last_frame_id));
  VERIFY(report->frame_id == "depth_from_file");
}

// Depth in pinned host memory reaches the operator as it is; the operator
// asks the memory to wait for the stream that copies it to the GPU.
void check_pinned_depth_held_by_operator() {
  const auto [depth, memory] = given_depth(synthetic_frame(kWidth, kHeight, 0, false), false);
  std::string failure;
  const auto report = run_graph(every_frame_config(), GivenFramesConfig{.depth = {depth}}, failure);
  VERIFY_MSG(failure.empty(), failure);
  VERIFY(report->clouds == 1U);
  VERIFY_MSG(memory->streams() > 0, "the operator asks the pinned depth's memory to wait");
}

// Device data the kernel cannot load its elements from in place is rejected.
void check_misaligned_device_data() {
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 0, false);
  const auto expect_failure = [](const DepthToPointCloudConfig& config, GivenFramesConfig frames,
                                 const std::string& what) {
    std::string failure;
    static_cast<void>(run_graph(config, std::move(frames), failure));
    VERIFY_MSG(failure.find(what) != std::string::npos, "rejected with '" + what + "': " + failure);
  };
  {
    const auto [depth, memory] = given_depth(f, true, 1);
    expect_failure(every_frame_config(), GivenFramesConfig{.depth = {depth}},
                   "depth data must be aligned");
  }
  {
    const auto [depth, depth_memory] = given_depth(f, true);
    holoscan::schema::ImageT bgra = color_image(f, holoscan::schema::ImageEncoding_BGRA8, false);
    const std::size_t bytes = static_cast<std::size_t>(f.width) * f.height * 4U;
    auto memory = std::make_shared<TestMemory>(bytes + 2U, true, true);
    VERIFY(cudaMemcpy(memory->base() + 2, bgra.data->data(), bytes, cudaMemcpyDefault) ==
           cudaSuccess);
    bgra.data = std::make_shared<holoscan::Tensor>(
        wrap(memory, 2, {f.height, f.width, 4}, {kDLUInt, 8, 1}, true));
    DepthToPointCloudConfig config = every_frame_config();
    config.color = true;
    expect_failure(config, GivenFramesConfig{.depth = {depth}, .color = {bgra}},
                   "color data must be aligned");
  }
}

// Runs the operator on one frame and checks that its failure names `what`.
void check_rejects(const DepthToPointCloudConfig& config, holoscan::schema::ImageT depth,
                   std::optional<holoscan::schema::ImageT> color, const hm::CameraInfo& info,
                   const std::string& what) {
  auto h = holoscan::testing::create_operator_test<DepthToPointCloudOp>(config);
  if (color) {
    h.push_frame(holoscan::testing::frame_at(
        100ms, holoscan::testing::input("camera_info", hm::to_message(info)),
        holoscan::testing::input("depth", depth), holoscan::testing::input("color", *color)));
  } else {
    h.push_frame(holoscan::testing::frame_at(
        100ms, holoscan::testing::input("camera_info", hm::to_message(info)),
        holoscan::testing::input("depth", depth)));
  }
  h.output_device("point_cloud", holoscan::DeviceId{0});
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

void check_rejected_inputs() {
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 0, false);
  const hm::CameraInfo info = synthetic_camera_info(f, "depth_optical");
  const DepthToPointCloudConfig config = base_config(false);
  // A color image where depth belongs.
  check_rejects(config, color_image(f, holoscan::schema::ImageEncoding_RGB8, false), std::nullopt,
                info, "unsupported depth encoding");
  // A depth image larger than the output is sized for.
  DepthToPointCloudConfig small = config;
  small.max_width = kWidth / 2;
  check_rejects(small, depth_image(f, false, false), std::nullopt, info, "exceeds the configured");
  // A color image of another size.
  DepthToPointCloudConfig with_color = config;
  with_color.color = true;
  const SyntheticFrame other = synthetic_frame(kWidth / 2, kHeight / 2, 0, false);
  check_rejects(with_color, depth_image(f, false, false),
                color_image(other, holoscan::schema::ImageEncoding_RGB8, false), info,
                "color image size must match");
  // Camera info for images of another size.
  check_rejects(config, depth_image(f, false, false), std::nullopt,
                synthetic_camera_info(other, "depth_optical"),
                "the depth image is 96x72 but camera info 'depth_optical' is for 48x36 images");
}

// Cameras the kernel cannot model fail the frame, naming what is wrong.
void check_rejected_cameras() {
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 0, false);
  const DepthToPointCloudConfig config = base_config(false);
  hm::CameraInfo raw = synthetic_camera_info(f, "depth_optical");
  raw.state = hm::CameraState::kRaw;
  raw.distortion_model = hm::DistortionModel::kPlumbBob;
  raw.coefficients = {0.1, -0.05, 0.0, 0.0, 0.0};
  check_rejects(config, depth_image(f, false, false), std::nullopt, raw,
                "camera info 'depth_optical' describes raw images, with plumb_bob distortion");
  hm::CameraInfo skewed = synthetic_camera_info(f, "depth_optical");
  skewed.skew = 0.5;
  check_rejects(config, depth_image(f, false, false), std::nullopt, skewed,
                "camera info 'depth_optical' has a skew");
  // Undistorted with a distortion model whose coefficients are all zero, and
  // rectified: both deproject.
  for (const hm::CameraState state : {hm::CameraState::kUndistorted, hm::CameraState::kRectified}) {
    hm::CameraInfo usable = synthetic_camera_info(f, "depth_optical");
    usable.state = state;
    usable.distortion_model = hm::DistortionModel::kPlumbBob;
    usable.coefficients = {0.0, 0.0, 0.0, 0.0, 0.0};
    auto h = holoscan::testing::create_operator_test<DepthToPointCloudOp>(config);
    h.push_frame(holoscan::testing::frame_at(
        capture_time(0), holoscan::testing::input("camera_info", hm::to_message(usable)),
        holoscan::testing::input("depth", depth_image(f, false, false))));
    h.output_device("point_cloud", holoscan::DeviceId{0});
    h.capture_snapshot<hm::schema::PointCloudT>("point_cloud", kCaptures);
    const auto result = h.run();
    VERIFY_MSG(result.ok(), result.describe_failure());
    if (!result.ok()) continue;
    const auto clouds = result.snapshots<hm::schema::PointCloudT>("point_cloud");
    VERIFY(clouds.size() == 1);
    if (clouds.size() == 1) check_cloud(clouds[0], false, f, false, config);
  }
}

void check_rejected_config() {
  const auto rejects = [](const auto& change, const std::string& what) {
    DepthToPointCloudConfig config = base_config(false);
    change(config);
    try {
      DepthToPointCloudOp op(config);
      VERIFY_MSG(false, what + " is rejected");
    } catch (const std::invalid_argument& error) {
      VERIFY_MSG(std::string(error.what()).find(what) != std::string::npos,
                 "'" + std::string(error.what()) + "' names '" + what + "'");
    }
  };
  rejects([](auto& c) { c.depth_max = c.depth_min; }, "depth range");
  rejects([](auto& c) { c.depth_scale = 0.0F; }, "depth_scale");
  rejects([](auto& c) { c.max_width = 0; }, "max_width and max_height");
  rejects([](auto& c) { c.camera_info.max_age = std::chrono::nanoseconds::zero(); },
          "camera_info.max_age");
}

}  // namespace
}  // namespace holoscan::point_cloud

int main() {
  namespace dpc = holoscan::point_cloud;
  return dpc::test::run({
      {"float32 depth from host memory", [] { dpc::check_depth_only(false, false); }},
      {"float32 depth from device memory", [] { dpc::check_depth_only(false, true); }},
      {"uint16 depth from host memory", [] { dpc::check_depth_only(true, false); }},
      {"uint16 depth from device memory", [] { dpc::check_depth_only(true, true); }},
      {"RGB8 color and a changing camera with every frame, host",
       [] { dpc::check_color_and_changing_camera(holoscan::schema::ImageEncoding_RGB8, false); }},
      {"BGR8 color and a changing camera with every frame, device",
       [] { dpc::check_color_and_changing_camera(holoscan::schema::ImageEncoding_BGR8, true); }},
      {"RGBA8 color and a changing camera with every frame, host",
       [] { dpc::check_color_and_changing_camera(holoscan::schema::ImageEncoding_RGBA8, false); }},
      {"BGRA8 color and a changing camera with every frame, device",
       [] { dpc::check_color_and_changing_camera(holoscan::schema::ImageEncoding_BGRA8, true); }},
      {"frames before the first camera info give no cloud", dpc::check_frames_before_camera_info},
      {"camera info with another capture time gives no cloud", dpc::check_unmatched_camera_info},
      {"a cloud carries its depth image's device sequence", dpc::check_frame_ids},
      {"CameraInfoSourceOp reads the camera from its file", dpc::check_camera_info_source},
      {"host input is held until its copy is done", dpc::check_host_input_held},
      {"pinned depth is held by the operator", dpc::check_pinned_depth_held_by_operator},
      {"misaligned device data is rejected", dpc::check_misaligned_device_data},
      {"unusable inputs are rejected", dpc::check_rejected_inputs},
      {"raw and skewed cameras are rejected", dpc::check_rejected_cameras},
      {"unusable configurations are rejected", dpc::check_rejected_config},
  });
}
