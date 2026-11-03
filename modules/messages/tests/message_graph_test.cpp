// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The three messages through a graph, sent by one operator and received by
// another: CameraInfo and StereoCameraInfo with and without their optional
// fields and with the longest frame ids; PointCloud in device and host
// memory, organized and not, empty, with optional tensors present and
// absent, every value checked. And the conversions' refusals.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/messages/conversion.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/messages/synthetic_calibration.hpp>

#include "graph_support.hpp"
#include "test_support.hpp"

namespace holoscan::messages {
namespace {

constexpr auto kPeriod = std::chrono::milliseconds{5};

std::string longest_frame_id(char fill) { return std::string(kMaxFrameIdBytes, fill); }

template <typename Value>
std::vector<Value> round_trip(const std::vector<Value>& values) {
  using Message = decltype(to_message(values.front()));
  std::vector<Message> messages;
  for (const Value& value : values) messages.push_back(to_message(value));
  auto received = std::make_shared<std::vector<test::Received<Message>>>();
  holoscan::Graph graph{"round_trip"};
  auto source = graph.op<test::ListSourceOp<Message>>("source", messages, kPeriod);
  auto sink = graph.op<test::CaptureOp<Message>>("sink", received, messages.size());
  graph.add_flow(source->output, sink->input, test::lossless());
  const test::RunOutcome outcome = test::run_graph(graph);
  VERIFY_MSG(outcome.ok(), outcome.diagnostics);
  std::vector<Value> back;
  for (const auto& message : *received) {
    VERIFY(message.capture_timestamp_ns != 0);
    back.push_back(from_message(message.data));
  }
  return back;
}

void test_camera_info() {
  StereoCameraInfo pair = synthetic_stereo_camera_info(DistortionModel::kRationalPolynomial);
  CameraInfo full = pair.right;
  full.header = Header{longest_frame_id('r'), 1234567890123, 42};
  full.skew = 0.5;
  full.roi = RegionOfInterest{16, 8, 1200, 700};
  full.binning_x = 2;
  full.binning_y = 3;
  CameraInfo minimal;
  minimal.width = 64;
  minimal.height = 48;
  minimal.state = CameraState::kUndistorted;
  minimal.fx = minimal.fy = 50.0;
  minimal.cx = 31.5;
  minimal.cy = 23.5;
  const std::vector<CameraInfo> sent{full, minimal};
  const std::vector<CameraInfo> back = round_trip(sent);
  VERIFY(back.size() == 2);
  if (back.size() != 2) return;
  VERIFY(back[0] == full);
  VERIFY(back[1] == minimal);
  VERIFY(!back[1].rectification && !back[1].roi && back[1].coefficients.empty());
}

void test_stereo_camera_info() {
  std::vector<StereoCameraInfo> sent;
  for (const DistortionModel model :
       {DistortionModel::kNone, DistortionModel::kPlumbBob, DistortionModel::kRationalPolynomial,
        DistortionModel::kEquidistant}) {
    sent.push_back(synthetic_stereo_camera_info(model, 640, 480));
  }
  sent[1].header = Header{longest_frame_id('s'), 77, 3};
  sent[1].left.header.frame_id = longest_frame_id('l');
  sent[1].right.header.frame_id = longest_frame_id('r');
  StereoCameraInfo rectified;
  for (CameraInfo* camera : {&rectified.left, &rectified.right}) {
    camera->width = 640;
    camera->height = 480;
    camera->state = CameraState::kRectified;
    camera->fx = camera->fy = 410.0;
    camera->cx = 320.0;
    camera->cy = 240.0;
  }
  rectified.right.cx = 330.0;
  rectified.right_from_left.translation = {-0.05, 0.0, 0.0};
  rectified.baseline = 0.05;
  rectified.doffs = 10.0;
  sent.push_back(rectified);
  const std::vector<StereoCameraInfo> back = round_trip(sent);
  VERIFY(back.size() == sent.size());
  for (std::size_t i = 0; i < back.size() && i < sent.size(); ++i) {
    VERIFY_MSG(back[i] == sent[i], "pair " + std::to_string(i));
  }
}

void test_conversion_refusals() {
  schema::StereoCameraInfoT stereo =
      to_message(synthetic_stereo_camera_info(DistortionModel::kPlumbBob));
  stereo.right.reset();
  VERIFY(test::refuses([&] { (void)from_message(stereo); },
                       "left, right and right_from_left must all be present"));
  schema::CameraInfoT camera =
      to_message(synthetic_stereo_camera_info(DistortionModel::kPlumbBob).left);
  camera.fx = -1.0;
  VERIFY(test::refuses([&] { (void)from_message(camera); }, "camera_info.fx"));
  camera.fx = 500.0;
  camera.binning_x = 0;  // ROS's 0 is 1.
  VERIFY(from_message(camera).binning_x == 1);
  camera.state = static_cast<schema::CameraState>(9);
  VERIFY(test::refuses([&] { (void)from_message(camera); }, "camera_info.state"));
  CameraInfo invalid = synthetic_stereo_camera_info(DistortionModel::kPlumbBob).left;
  invalid.coefficients.clear();
  VERIFY(test::refuses([&] { (void)to_message(invalid); }, "camera_info.coefficients"));
}

// --- point clouds ------------------------------------------------------------

struct CloudCase {
  const char* name;
  PointCloudShape shape;
  PointCloudContents contents;
  holoscan::MemoryKind memory;
  std::string frame_id;
};

std::int64_t points_of(const PointCloudShape& shape) {
  return shape.organized ? shape.height * shape.width : shape.width;
}

// Every value of frame `frame`, NaN for every 5th point and some normals.
struct CloudValues {
  std::vector<float> points;
  std::vector<std::uint8_t> colors;
  std::vector<float> normals;
  std::vector<float> intensity;
  std::int64_t valid{};
};

CloudValues cloud_values(std::int64_t n, std::uint32_t frame) {
  CloudValues v;
  for (std::int64_t i = 0; i < n; ++i) {
    const bool invalid = i % 5 == 3;
    v.valid += invalid ? 0 : 1;
    for (int k = 0; k < 3; ++k) {
      v.points.push_back(invalid ? NAN : static_cast<float>(frame * 1000.0 + i * 0.5 + k * 0.125));
      v.colors.push_back(static_cast<std::uint8_t>((i * 7 + k * 3 + frame) & 255));
      v.normals.push_back(i % 7 == 2 ? NAN : (k == 2 ? 1.0F : 0.0F));
    }
    v.intensity.push_back(static_cast<float>(i) * 0.25F + static_cast<float>(frame));
  }
  return v;
}

class CloudSourceOp final : public holoscan::Operator<> {
 public:
  CloudSourceOp(CloudCase cloud, std::uint32_t frames) : case_(std::move(cloud)), frames_(frames) {}

  void setup(holoscan::OperatorSpec& spec) override {
    declare_point_cloud_output(spec, cloud, "cloud",
                               {.max_points = static_cast<std::uint64_t>(
                                    std::max<std::int64_t>(points_of(case_.shape), 1)),
                                .colors = case_.contents.colors,
                                .normals = case_.contents.normals,
                                .intensity = case_.contents.intensity,
                                .memory_kind = case_.memory});
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kPeriod});
    return result;
  }
  void start() override { sent_ = 0; }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (sent_ >= frames_) return {};
    const CloudValues values = cloud_values(points_of(case_.shape), sent_);
    auto loan = allocate_point_cloud(cloud, case_.shape, case_.contents, context.cuda_stream());
    if (!loan) return holoscan::make_unexpected(std::move(loan).error());
    const auto put = [&](auto field, const void* data,
                         std::size_t bytes) -> holoscan::expected<void, holoscan::Error> {
      const auto fill = [&](const holoscan::TensorOutputWriteGuard& writer)
          -> holoscan::expected<void, holoscan::Error> {
        if (bytes == 0) return {};
        if (case_.memory == holoscan::MemoryKind::kHost) {
          std::memcpy(writer.data(), data, bytes);
        } else if (cudaMemcpyAsync(writer.data(), data, bytes, cudaMemcpyHostToDevice,
                                   context.cuda_stream().get()) != cudaSuccess) {
          return holoscan::make_unexpected(
              holoscan::Error{holoscan::ErrorCode::kFailure, "copy to the device failed"});
        }
        return {};
      };
      if (case_.memory == holoscan::MemoryKind::kHost)
        return loan->write_host_and_commit(field, fill);
      return loan->write_and_commit(field, context.cuda_stream(), fill);
    };
    if (auto r = put(PointCloudFields::points, values.points.data(), values.points.size() * 4); !r)
      return r;
    if (case_.contents.colors) {
      if (auto r = put(PointCloudFields::colors, values.colors.data(), values.colors.size()); !r)
        return r;
    }
    if (case_.contents.normals) {
      if (auto r = put(PointCloudFields::normals, values.normals.data(), values.normals.size() * 4);
          !r)
        return r;
    }
    if (case_.contents.intensity) {
      if (auto r = put(PointCloudFields::intensity, values.intensity.data(),
                       values.intensity.size() * 4);
          !r)
        return r;
    }
    const holoscan::TimePoint now = context.activation_time();
    auto sent = emit_point_cloud(cloud, std::move(*loan),
                                 Header{case_.frame_id, now.timestamp_ns, sent_ + 1U}, case_.shape,
                                 values.valid, holoscan::EmitOptions{.capture_time = now});
    if (!sent) {
      // A checker without room: this cloud again on the next tick.
      if (sent.error().code == holoscan::ErrorCode::kBackpressured) return {};
      return sent;
    }
    ++sent_;
    return {};
  }

  holoscan::Output<schema::PointCloudT> cloud;

 private:
  CloudCase case_;
  std::uint32_t frames_;
  std::uint32_t sent_{};
};

struct CloudReport {
  std::uint32_t frames{};
  std::uint32_t wrong{};
};

class CloudCheckOp final : public holoscan::Operator<> {
 public:
  CloudCheckOp(CloudCase cloud, std::uint32_t frames, std::shared_ptr<CloudReport> report)
      : case_(std::move(cloud)), frames_(frames), report_(std::move(report)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(cloud, "cloud").queue_depth(2U); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{cloud});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = cloud.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    const schema::PointCloudT& c = sample->data;
    const std::uint32_t frame = report_->frames;
    bool ok = static_cast<bool>(check_point_cloud(c));
    const std::int64_t n = points_of(case_.shape);
    const CloudValues expected = cloud_values(n, frame);
    ok = ok && c.header && c.header->frame_id == case_.frame_id &&
         c.header->device_sequence == frame + 1U &&
         c.header->capture_timestamp_ns == sample->metadata.capture_timestamp_ns &&
         c.organized == case_.shape.organized && c.valid_count == expected.valid;
    ok = ok && is_present(c.colors) == case_.contents.colors &&
         is_present(c.normals) == case_.contents.normals &&
         (n == 0 || is_present(c.intensity) == case_.contents.intensity);
    const auto memory = case_.memory == holoscan::MemoryKind::kHost ? kDLCPU : kDLCUDA;
    ok = ok && c.points->device().device_type == memory;
    ok = ok && same(*c.points, expected.points.data(), expected.points.size() * 4);
    if (case_.contents.colors)
      ok = ok && same(*c.colors, expected.colors.data(), expected.colors.size());
    if (case_.contents.normals) {
      ok = ok && same(*c.normals, expected.normals.data(), expected.normals.size() * 4);
    }
    if (case_.contents.intensity) {
      ok = ok && same(*c.intensity, expected.intensity.data(), expected.intensity.size() * 4);
    }
    report_->wrong += ok ? 0U : 1U;
    if (++report_->frames == frames_) context.request_completion();
    return {};
  }

  holoscan::Input<schema::PointCloudT> cloud;

 private:
  // Bit for bit, NaN included.
  static bool same(const holoscan::Tensor& tensor, const void* expected, std::size_t bytes) {
    if (static_cast<std::size_t>(tensor.nbytes()) != bytes) return false;
    if (bytes == 0) return true;
    std::vector<std::uint8_t> host(bytes);
    if (!tensor.copy_to_host(host.data(), bytes)) return false;
    return std::memcmp(host.data(), expected, bytes) == 0;
  }

  CloudCase case_;
  std::uint32_t frames_;
  std::shared_ptr<CloudReport> report_;
};

void check_cloud(const CloudCase& cloud) {
  constexpr std::uint32_t kFrames = 4;
  auto report = std::make_shared<CloudReport>();
  holoscan::Graph graph{"cloud"};
  auto source = graph.op<CloudSourceOp>("source", cloud, kFrames);
  auto check = graph.op<CloudCheckOp>("check", cloud, kFrames, report);
  graph.add_flow(source->cloud, check->cloud, test::lossless());
  holoscan::CompileOptions options;
  if (cloud.memory == holoscan::MemoryKind::kCudaDevice) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "source", .output_port = "cloud", .device = holoscan::DeviceId{0}});
  }
  const test::RunOutcome outcome = test::run_graph(graph, std::move(options));
  VERIFY_MSG(outcome.ok(), std::string(cloud.name) + ": " + outcome.diagnostics);
  VERIFY_MSG(report->frames == kFrames && report->wrong == 0,
             std::string(cloud.name) + ": " + std::to_string(report->wrong) + " of " +
                 std::to_string(report->frames) + " clouds wrong");
}

void test_point_clouds() {
  check_cloud({"device, organized, every field",
               organized_cloud(4, 6),
               {.colors = true, .normals = true, .intensity = true},
               holoscan::MemoryKind::kCudaDevice,
               longest_frame_id('p')});
  check_cloud({"device, unorganized, points only",
               unorganized_cloud(37),
               {},
               holoscan::MemoryKind::kCudaDevice,
               "lidar"});
  check_cloud({"host, organized, colors and intensity",
               organized_cloud(3, 5),
               {.colors = true, .intensity = true},
               holoscan::MemoryKind::kHost,
               "camera"});
  check_cloud({"device, empty",
               unorganized_cloud(0),
               {.colors = true},
               holoscan::MemoryKind::kCudaDevice,
               "empty"});
}

}  // namespace
}  // namespace holoscan::messages

int main() {
  namespace m = holoscan::messages;
  return holoscan::messages::test::run({
      {"CameraInfo through a graph", m::test_camera_info},
      {"StereoCameraInfo through a graph", m::test_stereo_camera_info},
      {"conversions refuse invalid messages", m::test_conversion_refusals},
      {"PointCloud through a graph", m::test_point_clouds},
  });
}
