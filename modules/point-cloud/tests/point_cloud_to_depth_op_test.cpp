// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// PointCloudToDepthOp on its own, through the SDK's operator test harness:
// organized clouds from host and device memory, which must give back the
// depth image they were made from exactly; unorganized synthetic clouds with
// colors in the body frame, from host and device memory, with configured and
// per-frame intrinsics, checked against the synthetic camera; the capture
// time and frame id carried to the outputs; clouds whose intrinsics do not
// match their capture time; and unusable inputs and configurations rejected.
// Through a graph: pinned host input held until its copy is done, and
// misaligned or foreign-device input rejected. Also the image headers.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/point_cloud/depth_truth.hpp>
#include <holoscan/point_cloud/point_cloud_to_depth_op.hpp>
#include <holoscan/point_cloud/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "host_input.hpp"
#include "tensor_support.hpp"
#include "test_support.hpp"

namespace holoscan::point_cloud {
namespace {

using test::make;
using test::stall;
using test::TestMemory;
using test::wrap;

using std::chrono_literals::operator""ms;
using ImageSnapshot = holoscan::testing::Snapshot<holoscan::schema::ImageT>;

constexpr int kWidth = 96;
constexpr int kHeight = 72;
constexpr DLDataType kFloat32{kDLFloat, 32, 1};
constexpr DLDataType kUInt8{kDLUInt, 8, 1};

holoscan::Tensor points_tensor(const std::vector<float>& xyz, bool device) {
  return make({static_cast<std::int64_t>(xyz.size() / 3), 3}, kFloat32, xyz.data(),
              xyz.size() * sizeof(float), device);
}

holoscan::Tensor colors_tensor(const std::vector<std::uint8_t>& rgb, bool device) {
  return make({static_cast<std::int64_t>(rgb.size() / 3), 3}, kUInt8, rgb.data(), rgb.size(),
              device);
}

holoscan::Tensor intrinsics_tensor(const CameraIntrinsics& k, bool device) {
  const std::array<float, 4> values{k.fx, k.fy, k.cx, k.cy};
  return make({4}, kFloat32, values.data(), sizeof(values), device);
}

// `bytes` at `offset` into new pinned host or device memory, as a tensor.
std::pair<holoscan::Tensor, std::shared_ptr<TestMemory>> given_tensor(
    const void* bytes, std::size_t size, const std::vector<std::int64_t>& shape, DLDataType dtype,
    bool device, std::size_t offset = 0) {
  // Pinned data is host-written: the runtime does not order its release.
  auto memory = std::make_shared<TestMemory>(size + offset, device, true, device);
  if (cudaMemcpy(memory->base() + offset, bytes, size, cudaMemcpyDefault) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return {wrap(memory, offset, shape, dtype, device), memory};
}

// Frame `f`'s depth image deprojected into the camera frame, one point per
// pixel in image order: an organized cloud. Holes give points at z = 0.
std::vector<float> organized_cloud(const SyntheticFrame& f, bool millimeters) {
  std::vector<float> xyz;
  for (int v = 0; v < f.height; ++v) {
    for (int u = 0; u < f.width; ++u) {
      const float z = synthetic_scene_depth(f, millimeters, u, v);
      xyz.insert(xyz.end(), {(u - f.intrinsics.cx) * z / f.intrinsics.fx,
                             (v - f.intrinsics.cy) * z / f.intrinsics.fy, z});
    }
  }
  return xyz;
}

PointCloudToDepthConfig base_config(DepthFormat format) {
  PointCloudToDepthConfig config;
  config.width = kWidth;
  config.height = kHeight;
  config.intrinsics = synthetic_frame(kWidth, kHeight, 0, false).intrinsics;
  config.depth_min = 0.1F;
  config.depth_max = 10.0F;
  config.depth_format = format;
  return config;
}

// The synthetic clouds' settings: body frame, and colors.
PointCloudToDepthConfig synthetic_config(DepthFormat format) {
  PointCloudToDepthConfig config = base_config(format);
  config.extrinsics = rigid_inverse(synthetic_body_from_camera());
  config.color = true;
  return config;
}

// The capture time of frame `frame` in the tests' frames.
std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

// Checks that an image carries frame `frame`'s capture time and `frame_id`.
void check_provenance(const ImageSnapshot& image, std::uint32_t frame, std::uint64_t frame_id) {
  const holoscan::SampleMetadata& metadata = image.sample_metadata();
  VERIFY_MSG(metadata.capture_timestamp_ns == capture_time(frame).count(),
             "frame " + std::to_string(frame) + " carries capture time " +
                 std::to_string(metadata.capture_timestamp_ns));
  VERIFY_MSG(
      metadata.frame_id == frame_id,
      "frame " + std::to_string(frame) + " carries frame id " + std::to_string(metadata.frame_id));
}

// The bytes of a captured image after checking its descriptor and shape.
template <typename T>
std::vector<T> image_data(const ImageSnapshot& image, holoscan::schema::ImageEncoding encoding,
                          std::int64_t channels) {
  const holoscan::schema::ImageT descriptor = image.metadata();
  VERIFY(descriptor.encoding == encoding);
  VERIFY(descriptor.width == kWidth && descriptor.height == kHeight);
  const auto& tensor = image.tensor(holoscan::schema::ImageTensorFields::data);
  const std::vector<std::int64_t> shape =
      channels == 1 ? std::vector<std::int64_t>{kHeight, kWidth}
                    : std::vector<std::int64_t>{kHeight, kWidth, channels};
  VERIFY(tensor.shape() == shape);
  std::vector<T> values(static_cast<std::size_t>(kWidth) * kHeight * channels);
  VERIFY(tensor.bytes().size() == values.size() * sizeof(T));
  if (tensor.bytes().size() == values.size() * sizeof(T)) {
    std::memcpy(values.data(), tensor.bytes().data(), tensor.bytes().size());
  }
  return values;
}

// Checks a depth image (and colors) against frame `f` of the synthetic camera.
void check_truth(const ImageSnapshot& depth, const ImageSnapshot* color, const SyntheticFrame& f,
                 const PointCloudToDepthConfig& config) {
  const bool uint16 = config.depth_format == DepthFormat::kUint16;
  const SceneTruth truth{.millimeter_depth = uint16,
                         .format = config.depth_format,
                         .depth_scale = config.depth_scale,
                         .depth_min = config.depth_min,
                         .depth_max = config.depth_max,
                         .point_size = config.point_size};
  std::vector<std::uint8_t> rgb;
  if (color != nullptr) {
    rgb = image_data<std::uint8_t>(*color, holoscan::schema::ImageEncoding_RGB8, 3);
  }
  const std::uint8_t* rgb_data = color != nullptr ? rgb.data() : nullptr;
  TruthCheck check;
  if (uint16) {
    const auto counts = image_data<std::uint16_t>(depth, holoscan::schema::ImageEncoding_MONO16, 1);
    check = check_against_scene(f, truth, counts.data(), rgb_data);
  } else {
    const auto meters = image_data<float>(depth, holoscan::schema::ImageEncoding_MONO32F, 1);
    check = check_against_scene(f, truth, meters.data(), rgb_data);
  }
  VERIFY_MSG(check.wrong_depths == 0,
             std::to_string(check.wrong_depths) + " depths differ from the truth");
  VERIFY_MSG(check.wrong_colors == 0,
             std::to_string(check.wrong_colors) + " colors differ from the truth");
}

// An organized cloud made from a depth image gives that image back exactly.
void check_organized_round_trip(bool millimeters, bool device) {
  const PointCloudToDepthConfig config =
      base_config(millimeters ? DepthFormat::kUint16 : DepthFormat::kFloat32);
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 2, false);
  const std::vector<float> xyz = organized_cloud(f, millimeters);
  h.push("points", make({kHeight, kWidth, 3}, kFloat32, xyz.data(), xyz.size() * 4, device));
  if (device) {
    h.input_source_device("points", holoscan::DeviceId{0});
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto images = result.snapshots<holoscan::schema::ImageT>("depth");
  VERIFY(images.size() == 1);
  if (images.empty()) return;
  int wrong = 0;
  int filled = 0;
  std::vector<float> meters;
  std::vector<std::uint16_t> counts;
  if (millimeters) {
    counts = image_data<std::uint16_t>(images[0], holoscan::schema::ImageEncoding_MONO16, 1);
  } else {
    meters = image_data<float>(images[0], holoscan::schema::ImageEncoding_MONO32F, 1);
  }
  for (int v = 0; v < kHeight; ++v) {
    for (int u = 0; u < kWidth; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * kWidth + u;
      const float z = synthetic_scene_depth(f, millimeters, u, v);
      const bool valid = z >= config.depth_min && z <= config.depth_max;
      filled += valid;
      if (millimeters) {
        wrong += counts[i] != (valid ? synthetic_depth_mm(f, u, v) : 0U);
      } else {
        wrong += meters[i] != (valid ? z : 0.0F);
      }
    }
  }
  VERIFY_MSG(wrong == 0, std::to_string(wrong) + " pixels differ from the image");
  VERIFY(filled > kWidth * kHeight / 2);
}

// Unorganized synthetic clouds with colors, in the body frame.
void check_synthetic_clouds(DepthFormat format, int point_size, bool device) {
  PointCloudToDepthConfig config = synthetic_config(format);
  config.point_size = point_size;
  const bool millimeters = format == DepthFormat::kUint16;
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    std::vector<float> xyz;
    std::vector<std::uint8_t> rgb;
    synthetic_cloud(synthetic_frame(kWidth, kHeight, frame, false), millimeters, xyz, rgb);
    holoscan::testing::TestFrame staged = holoscan::testing::frame_at(
        capture_time(frame), holoscan::testing::input("points", points_tensor(xyz, device)),
        holoscan::testing::input("colors", colors_tensor(rgb, device)));
    h.push_frame(staged);
  }
  if (device) {
    h.input_source_device("points", holoscan::DeviceId{0});
    h.input_source_device("colors", holoscan::DeviceId{0});
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.output_device("color", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth");
  h.capture_snapshot<holoscan::schema::ImageT>("color");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  const auto colors = result.snapshots<holoscan::schema::ImageT>("color");
  VERIFY(depths.size() == 2 && colors.size() == 2);
  for (std::size_t frame = 0; frame < depths.size() && frame < colors.size(); ++frame) {
    const auto number = static_cast<std::uint32_t>(frame);
    check_truth(depths[frame], &colors[frame], synthetic_frame(kWidth, kHeight, number, false),
                config);
    check_provenance(depths[frame], number, 0U);
    check_provenance(colors[frame], number, 0U);
  }
}

// A cloud's frame id travels to its image.
void check_frame_ids() {
  const PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  std::vector<holoscan::testing::TestSample<holoscan::Tensor>> samples;
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    const std::vector<float> xyz =
        organized_cloud(synthetic_frame(kWidth, kHeight, frame, false), false);
    holoscan::testing::TestSample<holoscan::Tensor> sample =
        holoscan::testing::sample_at(points_tensor(xyz, false), capture_time(frame));
    sample.metadata.frame_id = 41U + frame;
    samples.push_back(std::move(sample));
  }
  h.push_samples("points", std::move(samples));
  h.output_device("depth", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto images = result.snapshots<holoscan::schema::ImageT>("depth");
  VERIFY(images.size() == 2);
  for (std::size_t frame = 0; frame < images.size(); ++frame) {
    check_provenance(images[frame], static_cast<std::uint32_t>(frame), 41U + frame);
  }
}

PointCloudToDepthConfig intrinsics_input_config() {
  PointCloudToDepthConfig config = synthetic_config(DepthFormat::kUint16);
  config.intrinsics.reset();
  config.intrinsics_input = true;
  return config;
}

// Clouds made with per-frame intrinsics, which arrive aligned with them.
void check_per_frame_intrinsics(bool device) {
  const PointCloudToDepthConfig config = intrinsics_input_config();
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, true);
    std::vector<float> xyz;
    std::vector<std::uint8_t> rgb;
    synthetic_cloud(f, true, xyz, rgb);
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame), holoscan::testing::input("points", points_tensor(xyz, device)),
        holoscan::testing::input("colors", colors_tensor(rgb, device)),
        holoscan::testing::input("intrinsics", intrinsics_tensor(f.intrinsics, device))));
  }
  if (device) {
    for (const char* port : {"points", "colors", "intrinsics"}) {
      h.input_source_device(port, holoscan::DeviceId{0});
    }
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.output_device("color", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth");
  h.capture_snapshot<holoscan::schema::ImageT>("color");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  const auto colors = result.snapshots<holoscan::schema::ImageT>("color");
  VERIFY(depths.size() == 3 && colors.size() == 3);
  for (std::size_t frame = 0; frame < depths.size() && frame < colors.size(); ++frame) {
    const auto number = static_cast<std::uint32_t>(frame);
    check_truth(depths[frame], &colors[frame], synthetic_frame(kWidth, kHeight, number, true),
                config);
    check_provenance(depths[frame], number, 0U);
  }
}

// Intrinsics pair with clouds by exact capture time: a cloud whose intrinsics
// carry another time never completes, and gives no image, while the clouds
// around it still do.
void check_unmatched_intrinsics() {
  const PointCloudToDepthConfig config = intrinsics_input_config();
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, true);
    std::vector<float> xyz;
    std::vector<std::uint8_t> rgb;
    synthetic_cloud(f, true, xyz, rgb);
    // Frame 1's intrinsics are a millisecond late.
    const auto intrinsics_time =
        capture_time(frame) + std::chrono::milliseconds(frame == 1 ? 1 : 0);
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame), holoscan::testing::input("points", points_tensor(xyz, false)),
        holoscan::testing::input("colors", colors_tensor(rgb, false))));
    h.push_frame(holoscan::testing::frame_at(
        intrinsics_time,
        holoscan::testing::input("intrinsics", intrinsics_tensor(f.intrinsics, false))));
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.output_device("color", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth");
  h.capture_snapshot<holoscan::schema::ImageT>("color");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  const auto colors = result.snapshots<holoscan::schema::ImageT>("color");
  VERIFY_MSG(depths.size() == 2 && colors.size() == 2,
             std::to_string(depths.size()) + " depth and " + std::to_string(colors.size()) +
                 " color images from 2 matched clouds");
  const std::array<std::uint32_t, 2> matched{0U, 2U};
  for (std::size_t i = 0; i < depths.size() && i < colors.size() && i < matched.size(); ++i) {
    check_truth(depths[i], &colors[i], synthetic_frame(kWidth, kHeight, matched[i], true), config);
    check_provenance(depths[i], matched[i], 0U);
  }
}

// A cloud with no points gives an empty image.
void check_empty_cloud() {
  PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  config.color = true;
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  h.push_frame(holoscan::testing::frame_at(
      100ms, holoscan::testing::input("points", points_tensor({}, false)),
      holoscan::testing::input("colors", colors_tensor({}, false))));
  h.output_device("depth", holoscan::DeviceId{0});
  h.output_device("color", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth");
  h.capture_snapshot<holoscan::schema::ImageT>("color");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  const auto colors = result.snapshots<holoscan::schema::ImageT>("color");
  VERIFY(depths.size() == 1 && colors.size() == 1);
  if (depths.size() != 1 || colors.size() != 1) return;
  for (const float z : image_data<float>(depths[0], holoscan::schema::ImageEncoding_MONO32F, 1)) {
    VERIFY(z == 0.0F);
  }
  for (const std::uint8_t c :
       image_data<std::uint8_t>(colors[0], holoscan::schema::ImageEncoding_RGB8, 3)) {
    VERIFY(c == 0U);
  }
}

// Host input is held until the stream that copies it is done: by its allocator
// when that can wait for a stream, without blocking; otherwise by waiting.
void check_host_input_held() {
  cudaStream_t stream{};
  VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  {
    auto memory = std::make_shared<TestMemory>(64, false, true);
    VERIFY(stall(stream));
    VERIFY(
        detail::hold_until_done(wrap(memory, 0, {64}, kUInt8, false), stream, "test").has_value());
    VERIFY_MSG(memory->streams() == 1, "the allocator is asked to wait for the stream");
    VERIFY_MSG(cudaStreamQuery(stream) == cudaErrorNotReady, "the host does not wait");
    VERIFY(cudaStreamSynchronize(stream) == cudaSuccess);
  }
  {
    auto memory = std::make_shared<TestMemory>(64, false, false);
    VERIFY(stall(stream));
    VERIFY(
        detail::hold_until_done(wrap(memory, 0, {64}, kUInt8, false), stream, "test").has_value());
    VERIFY_MSG(cudaStreamQuery(stream) == cudaSuccess,
               "the stream is done when the allocator cannot wait for it");
  }
  VERIFY(cudaStreamDestroy(stream) == cudaSuccess);
}

// The test harness copies staged inputs into its own storage, so a tensor whose
// memory the test controls reaches the operator only through a graph: this
// source sends the given clouds as they are, one per tick, each with the given
// colors and intrinsics when there are any.
struct GivenCloudsConfig {
  std::vector<holoscan::Tensor> points;
  std::vector<holoscan::Tensor> colors;
  std::vector<holoscan::Tensor> intrinsics;
};

class GivenCloudsSourceOp final : public holoscan::Operator<> {
 public:
  explicit GivenCloudsSourceOp(GivenCloudsConfig config) : config_(std::move(config)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(points, "points");
    if (!config_.colors.empty()) {
      spec.output(colors, "colors");
    }
    if (!config_.intrinsics.empty()) {
      spec.output(intrinsics, "intrinsics");
    }
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(5)});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_ >= config_.points.size()) {
      return {};
    }
    // A cloud keeps its capture time until all its parts are out.
    if (!capture_) {
      capture_ = context.activation_time();
    }
    const holoscan::EmitOptions options{.capture_time = *capture_, .frame_id = next_ + 1U};
    const std::size_t i = next_;
    // Sends one part unless sent already; false when the consumer is busy.
    const auto send = [&](bool& sent, holoscan::Output<holoscan::Tensor>& port,
                          const std::vector<holoscan::Tensor>& parts)
        -> holoscan::expected<bool, holoscan::Error> {
      if (sent || parts.empty()) {
        return true;
      }
      auto result = port.emit(parts[i], options);
      if (!result) {
        if (result.error().code == holoscan::ErrorCode::kBackpressured) {
          return false;
        }
        return holoscan::make_unexpected(std::move(result).error());
      }
      sent = true;
      return true;
    };
    for (auto [sent, port, parts] :
         {std::tuple{&sent_intrinsics_, &intrinsics, &config_.intrinsics},
          std::tuple{&sent_colors_, &colors, &config_.colors},
          std::tuple{&sent_points_, &points, &config_.points}}) {
      auto done = send(*sent, *port, *parts);
      if (!done) {
        return holoscan::make_unexpected(std::move(done).error());
      }
      if (!*done) {
        return {};
      }
    }
    ++next_;
    capture_.reset();
    sent_intrinsics_ = sent_colors_ = sent_points_ = false;
    return {};
  }

  holoscan::Output<holoscan::Tensor> points;
  holoscan::Output<holoscan::Tensor> colors;
  holoscan::Output<holoscan::Tensor> intrinsics;

 private:
  GivenCloudsConfig config_;
  std::uint64_t next_{};
  std::optional<holoscan::TimePoint> capture_;
  bool sent_intrinsics_{};
  bool sent_colors_{};
  bool sent_points_{};
};

// What the sink below saw.
struct DepthCount {
  std::uint32_t frames{};
  std::uint64_t filled_pixels{};
};

// Counts the depth images (with their color images) it receives and their
// filled pixels, and completes the run after `expected` of them.
class DepthCountOp final : public holoscan::Operator<> {
 public:
  DepthCountOp(std::shared_ptr<DepthCount> count, std::uint32_t expected, DepthFormat format,
               bool color)
      : count_(std::move(count)), expected_(expected), format_(format), color_(color) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(depth, "depth").queue_depth(2U);
    if (color_) {
      spec.input(color, "color").queue_depth(2U);
    }
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    if (!color_) {
      result.trigger(holoscan::OnEach{depth});
      return result;
    }
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::PortList{std::vector<holoscan::PortRef>{depth, color}},
        .window = std::chrono::nanoseconds{0}});
    result.input(depth, holoscan::Exact{});
    result.input(color, holoscan::Exact{});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto image = depth.receive();
    if (!image) {
      return holoscan::make_unexpected(std::move(image).error());
    }
    if (color_) {
      if (auto rgb = color.receive(); !rgb) {
        return holoscan::make_unexpected(std::move(rgb).error());
      }
    }
    if (!image->data.data) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "DepthCountOp: image has no tensor"});
    }
    const holoscan::Tensor& tensor = *image->data.data;
    const auto pixels = static_cast<std::size_t>(image->data.width) * image->data.height;
    if (format_ == DepthFormat::kUint16) {
      std::vector<std::uint16_t> counts(pixels);
      if (auto copied = tensor.copy_to_host(counts.data(), counts.size() * sizeof(std::uint16_t),
                                            context.cuda_stream());
          !copied) {
        return copied;
      }
      count_->filled_pixels += pixels - std::count(counts.begin(), counts.end(), 0U);
    } else {
      std::vector<float> meters(pixels);
      if (auto copied = tensor.copy_to_host(meters.data(), meters.size() * sizeof(float),
                                            context.cuda_stream());
          !copied) {
        return copied;
      }
      count_->filled_pixels += pixels - std::count(meters.begin(), meters.end(), 0.0F);
    }
    if (++count_->frames == expected_) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> depth;
  holoscan::Input<holoscan::schema::ImageT> color;

 private:
  std::shared_ptr<DepthCount> count_;
  std::uint32_t expected_{};
  DepthFormat format_{};
  bool color_{};
};

// Runs `clouds` through the operator into a counting sink. Returns what the
// sink saw, and in `failure` the run's diagnostics when it did not complete.
std::shared_ptr<DepthCount> run_graph(const PointCloudToDepthConfig& config,
                                      GivenCloudsConfig clouds, std::string& failure) {
  holoscan::Graph graph{"point_cloud_to_depth_test"};
  const auto count = static_cast<std::uint32_t>(clouds.points.size());
  auto source = graph.op<GivenCloudsSourceOp>("source", std::move(clouds));
  auto project = graph.op<PointCloudToDepthOp>("project", config);
  auto report = std::make_shared<DepthCount>();
  auto sink = graph.op<DepthCountOp>("count", report, count, config.depth_format, config.color);
  graph.add_flow(source->points, project->points);
  if (config.color) {
    graph.add_flow(source->colors, project->colors);
    graph.add_flow(project->color, sink->color);
  }
  if (config.intrinsics_input) {
    graph.add_flow(source->intrinsics, project->intrinsics);
  }
  graph.add_flow(project->depth, sink->depth);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  for (const char* port : {"depth", "color"}) {
    if (std::string(port) == "depth" || config.color) {
      options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = "project", .output_port = port, .device = holoscan::DeviceId{0}});
    }
  }
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

// An organized cloud of frame 0 at `offset` bytes into new pinned host or
// device memory.
std::pair<holoscan::Tensor, std::shared_ptr<TestMemory>> given_points(bool device,
                                                                      std::size_t offset = 0) {
  const std::vector<float> xyz = organized_cloud(synthetic_frame(kWidth, kHeight, 0, false), false);
  return given_tensor(xyz.data(), xyz.size() * sizeof(float), {kHeight, kWidth, 3}, kFloat32,
                      device, offset);
}

// Points in pinned host memory reach the operator as they are; the operator
// asks the memory to wait for the stream that copies them to the GPU.
void check_pinned_points_held_by_operator() {
  const auto [points, memory] = given_points(false);
  std::string failure;
  const auto report =
      run_graph(base_config(DepthFormat::kFloat32), GivenCloudsConfig{.points = {points}}, failure);
  VERIFY_MSG(failure.empty(), failure);
  VERIFY(report->frames == 1 && report->filled_pixels > 0U);
  VERIFY_MSG(memory->streams() > 0, "the operator asks the pinned points' memory to wait");
}

// Intrinsics in pinned host memory are read by the host, so the operator waits
// for their producer on the host, not only on its stream.
void check_pinned_intrinsics_waited_for() {
  const auto [points, points_memory] = given_points(false);
  const CameraIntrinsics k = synthetic_frame(kWidth, kHeight, 0, false).intrinsics;
  const std::array<float, 4> values{k.fx, k.fy, k.cx, k.cy};
  const auto [intrinsics, k_memory] =
      given_tensor(values.data(), sizeof(values), {4}, kFloat32, false);
  PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  config.intrinsics.reset();
  config.intrinsics_input = true;
  std::string failure;
  const auto report =
      run_graph(config, GivenCloudsConfig{.points = {points}, .intrinsics = {intrinsics}}, failure);
  VERIFY_MSG(failure.empty(), failure);
  VERIFY(report->frames == 1 && report->filled_pixels > 0U);
  VERIFY_MSG(k_memory->host_waits() > 0, "the host waits for pinned intrinsics before reading");
}

// Device data the kernel cannot load its elements from in place is rejected.
void check_misaligned_device_data() {
  const auto expect_failure = [](const PointCloudToDepthConfig& config, GivenCloudsConfig clouds,
                                 const std::string& what) {
    std::string failure;
    static_cast<void>(run_graph(config, std::move(clouds), failure));
    VERIFY_MSG(failure.find(what) != std::string::npos, "rejected with '" + what + "': " + failure);
  };
  {
    const auto [points, memory] = given_points(true, 2);
    expect_failure(base_config(DepthFormat::kFloat32), GivenCloudsConfig{.points = {points}},
                   "point data must be aligned");
  }
  {
    const auto [points, points_memory] = given_points(true);
    const CameraIntrinsics k = synthetic_frame(kWidth, kHeight, 0, false).intrinsics;
    const std::array<float, 4> values{k.fx, k.fy, k.cx, k.cy};
    const auto [intrinsics, k_memory] =
        given_tensor(values.data(), sizeof(values), {4}, kFloat32, true, 2);
    PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
    config.intrinsics.reset();
    config.intrinsics_input = true;
    expect_failure(config, GivenCloudsConfig{.points = {points}, .intrinsics = {intrinsics}},
                   "intrinsics data must be aligned");
  }
}

// Device data on another GPU than the operator's is rejected rather than
// read: here memory of this GPU, labeled as another one's.
void check_foreign_device_points() {
  const std::vector<float> xyz = organized_cloud(synthetic_frame(kWidth, kHeight, 0, false), false);
  const std::size_t bytes = xyz.size() * sizeof(float);
  auto memory = std::make_shared<TestMemory>(bytes, true, true);
  VERIFY(cudaMemcpy(memory->base(), xyz.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  std::string failure;
  static_cast<void>(run_graph(
      base_config(DepthFormat::kFloat32),
      GivenCloudsConfig{.points = {wrap(memory, 0, {kHeight, kWidth, 3}, kFloat32, true, 1)}},
      failure));
  VERIFY_MSG(failure.find("points are on another GPU") != std::string::npos,
             "rejected as on another GPU: " + failure);
}

// The images' descriptors carry the cloud's capture time and the camera's
// frame in their header.
void check_image_headers() {
  PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  config.color = true;
  config.frame_id = "camera_optical";
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    const std::vector<float> xyz =
        organized_cloud(synthetic_frame(kWidth, kHeight, frame, false), false);
    const std::vector<std::uint8_t> rgb(xyz.size(), 9U);
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame),
        holoscan::testing::input("points", make({kHeight, kWidth, 3}, kFloat32, xyz.data(),
                                                xyz.size() * sizeof(float), false)),
        holoscan::testing::input(
            "colors", make({kHeight, kWidth, 3}, kUInt8, rgb.data(), rgb.size(), false))));
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.output_device("color", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth");
  h.capture_snapshot<holoscan::schema::ImageT>("color");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  for (const char* port : {"depth", "color"}) {
    const auto images = result.snapshots<holoscan::schema::ImageT>(port);
    VERIFY(images.size() == 2);
    for (std::size_t frame = 0; frame < images.size(); ++frame) {
      const holoscan::schema::ImageT descriptor = images[frame].metadata();
      VERIFY_MSG(descriptor.header != nullptr, std::string(port) + " image has a header");
      if (descriptor.header == nullptr) continue;
      VERIFY_MSG(descriptor.header->capture_timestamp_ns ==
                     capture_time(static_cast<std::uint32_t>(frame)).count(),
                 std::string(port) + " header carries capture time " +
                     std::to_string(descriptor.header->capture_timestamp_ns));
      VERIFY(descriptor.header->frame_id == "camera_optical");
    }
  }
}

// Runs the operator on one unusable input and checks the failure names `what`.
void check_rejects(const PointCloudToDepthConfig& config, holoscan::Tensor points,
                   std::optional<holoscan::Tensor> colors,
                   std::optional<holoscan::Tensor> intrinsics, const std::string& what) {
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  if (colors) {
    h.push_frame(holoscan::testing::frame_at(100ms, holoscan::testing::input("points", points),
                                             holoscan::testing::input("colors", *colors)));
  } else if (intrinsics) {
    h.push_frame(holoscan::testing::frame_at(100ms, holoscan::testing::input("points", points),
                                             holoscan::testing::input("intrinsics", *intrinsics)));
  } else {
    h.push("points", points);
  }
  h.output_device("depth", holoscan::DeviceId{0});
  if (config.color) {
    h.output_device("color", holoscan::DeviceId{0});
  }
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

void check_rejected_inputs() {
  const PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  const std::vector<float> xyz(12, 1.0F);
  const std::vector<double> xyz64(12, 1.0);
  const char* kShape = "points must be float32 [N, 3] or [H, W, 3]";
  check_rejects(config, make({4, 3}, {kDLFloat, 64, 1}, xyz64.data(), 96, false), {}, {}, kShape);
  check_rejects(config, make({3, 4}, kFloat32, xyz.data(), 48, false), {}, {}, kShape);
  check_rejects(config, make({12}, kFloat32, xyz.data(), 48, false), {}, {}, kShape);
  check_rejects(config, make({1, 2, 2, 3}, kFloat32, xyz.data(), 48, false), {}, {}, kShape);
  PointCloudToDepthConfig small = config;
  small.max_points = 3;
  check_rejects(small, points_tensor(xyz, false), {}, {}, "exceed the configured maximum of 3");

  PointCloudToDepthConfig with_color = config;
  with_color.color = true;
  const std::vector<std::uint8_t> rgb(12, 7U);
  const char* kColors = "colors must be uint8 RGB of the points' shape";
  check_rejects(with_color, points_tensor(xyz, false), make({3, 3}, kUInt8, rgb.data(), 9, false),
                {}, kColors);
  check_rejects(with_color, points_tensor(xyz, false),
                make({2, 2, 3}, kUInt8, rgb.data(), 12, false), {}, kColors);
  check_rejects(with_color, points_tensor(xyz, false),
                make({4, 3}, kFloat32, xyz.data(), 48, false), {}, kColors);

  PointCloudToDepthConfig per_frame = config;
  per_frame.intrinsics.reset();
  per_frame.intrinsics_input = true;
  check_rejects(per_frame, points_tensor(xyz, false), {},
                intrinsics_tensor({0.0F, 50.0F, 10.0F, 10.0F}, false),
                "intrinsics need finite, positive");
  const std::array<float, 3> three{50.0F, 50.0F, 10.0F};
  check_rejects(per_frame, points_tensor(xyz, false), {},
                make({3}, kFloat32, three.data(), sizeof(three), false),
                "intrinsics must be float32 [fx, fy, cx, cy]");
}

void check_rejected_config() {
  const auto rejects = [](const auto& change, const std::string& what) {
    PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
    change(config);
    try {
      PointCloudToDepthOp op(config);
      VERIFY_MSG(false, what + " is rejected");
    } catch (const std::invalid_argument& error) {
      VERIFY_MSG(std::string(error.what()).find(what) != std::string::npos,
                 "'" + std::string(error.what()) + "' names '" + what + "'");
    }
  };
  rejects([](auto& c) { c.width = 0; }, "width and height must be 1..16384");
  rejects([](auto& c) { c.height = 16385; }, "width and height must be 1..16384");
  rejects([](auto& c) { c.intrinsics.reset(); }, "give intrinsics or enable the intrinsics input");
  rejects([](auto& c) { c.intrinsics_input = true; }, "not both");
  rejects([](auto& c) { c.intrinsics->fy = -1.0F; }, "intrinsics need finite, positive");
  rejects([](auto& c) { c.intrinsics->cx = std::nanf(""); }, "intrinsics need finite, positive");
  rejects(
      [](auto& c) {
        c.extrinsics = Extrinsics{};
        c.extrinsics->matrix[5] = std::numeric_limits<float>::infinity();
      },
      "extrinsics must be finite");
  rejects([](auto& c) { c.depth_max = c.depth_min; }, "depth range");
  rejects([](auto& c) { c.depth_min = -1.0F; }, "depth range");
  rejects([](auto& c) { c.depth_scale = 0.0F; }, "depth_scale");
  rejects([](auto& c) { c.depth_format = static_cast<DepthFormat>(7); }, "depth_format");
  rejects([](auto& c) { c.point_size = 2; }, "point_size must be 1, 3 or 5");
  rejects([](auto& c) { c.point_size = 7; }, "point_size must be 1, 3 or 5");
  rejects([](auto& c) { c.max_points = 0; }, "max_points");
  rejects([](auto& c) { c.max_points = kMaxPointsLimit + 1U; }, "max_points");
}

}  // namespace
}  // namespace holoscan::point_cloud

int main() {
  namespace pcd = holoscan::point_cloud;
  using pcd::DepthFormat;
  return pcd::test::run({
      {"organized float32 cloud from host memory gives its image back",
       [] { pcd::check_organized_round_trip(false, false); }},
      {"organized uint16 cloud from device memory gives its image back",
       [] { pcd::check_organized_round_trip(true, true); }},
      {"synthetic clouds with colors, float32, host",
       [] { pcd::check_synthetic_clouds(DepthFormat::kFloat32, 1, false); }},
      {"synthetic clouds with colors, uint16, device, point size 3",
       [] { pcd::check_synthetic_clouds(DepthFormat::kUint16, 3, true); }},
      {"synthetic clouds with colors, float32, device, point size 5",
       [] { pcd::check_synthetic_clouds(DepthFormat::kFloat32, 5, true); }},
      {"frame ids reach the images", pcd::check_frame_ids},
      {"per-frame intrinsics in host memory", [] { pcd::check_per_frame_intrinsics(false); }},
      {"per-frame intrinsics in device memory", [] { pcd::check_per_frame_intrinsics(true); }},
      {"intrinsics with another capture time give no image", pcd::check_unmatched_intrinsics},
      {"a cloud without points gives an empty image", pcd::check_empty_cloud},
      {"host input is held until its copy is done", pcd::check_host_input_held},
      {"pinned points are held by the operator", pcd::check_pinned_points_held_by_operator},
      {"pinned intrinsics are waited for on the host", pcd::check_pinned_intrinsics_waited_for},
      {"misaligned device data is rejected", pcd::check_misaligned_device_data},
      {"points on another GPU are rejected", pcd::check_foreign_device_points},
      {"image headers carry the capture time and frame", pcd::check_image_headers},
      {"unusable inputs are rejected", pcd::check_rejected_inputs},
      {"unusable configurations are rejected", pcd::check_rejected_config},
  });
}
