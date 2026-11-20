// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The three messages through a graph, sent by one operator and received by
// another: CameraInfo and StereoCameraInfo with and without their optional
// fields and with the longest frame ids; PointCloud in device, host and
// pinned host memory, organized and not, empty, with optional tensors present
// and absent, every value checked. And the conversions' refusals, the point
// cloud helpers' refusals, and check_point_cloud() on clouds made by hand.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
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
    const bool device = case_.memory == holoscan::MemoryKind::kCudaDevice;
    const auto put = [&](auto field, const void* data,
                         std::size_t bytes) -> holoscan::expected<void, holoscan::Error> {
      const auto fill = [&](const holoscan::TensorOutputWriteGuard& writer)
          -> holoscan::expected<void, holoscan::Error> {
        if (bytes == 0) return {};
        if (!device) {
          std::memcpy(writer.data(), data, bytes);
        } else if (cudaMemcpyAsync(writer.data(), data, bytes, cudaMemcpyHostToDevice,
                                   context.cuda_stream().get()) != cudaSuccess) {
          return holoscan::make_unexpected(
              holoscan::Error{holoscan::ErrorCode::kFailure, "copy to the device failed"});
        }
        return {};
      };
      if (!device) return loan->write_host_and_commit(field, fill);
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
    // A copy from pageable memory may return before it has read `values`:
    // wait for the copies before `values` goes.
    if (device) {
      if (const cudaError_t done = cudaStreamSynchronize(context.cuda_stream().get());
          done != cudaSuccess) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, cudaGetErrorString(done)});
      }
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
    const auto memory = case_.memory == holoscan::MemoryKind::kHost         ? kDLCPU
                        : case_.memory == holoscan::MemoryKind::kPinnedHost ? kDLCUDAHost
                                                                            : kDLCUDA;
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
  check_cloud({"pinned host, unorganized, normals",
               unorganized_cloud(11),
               {.normals = true},
               holoscan::MemoryKind::kPinnedHost,
               "pinned"});
}

// --- the point cloud helpers' refusals --------------------------------------

struct Refusals {
  std::optional<holoscan::Error> empty_organized;  // allocate_point_cloud(), H = 0.
  std::optional<holoscan::Error> long_frame_id;    // emit_point_cloud(), 257 bytes.
  std::optional<holoscan::Error> other_shape;      // emit_point_cloud(), not as allocated.
  std::optional<holoscan::Error> too_many_valid;   // emit_point_cloud(), valid_count > N.
  bool sent{};                                     // The cloud that is right.
  std::uint32_t received{};
};

// On its first tick, tries every refusal, then sends one right cloud.
class RefusalProbeOp final : public holoscan::Operator<> {
 public:
  explicit RefusalProbeOp(std::shared_ptr<Refusals> refusals) : refusals_(std::move(refusals)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    declare_point_cloud_output(spec, cloud, "cloud",
                               {.max_points = 16, .memory_kind = holoscan::MemoryKind::kHost});
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kPeriod});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (done_) return {};
    done_ = true;
    Refusals& r = *refusals_;
    const ::cuda::stream_ref stream = context.cuda_stream();
    const PointCloudShape shape = unorganized_cloud(4);
    const auto error_of = [](holoscan::expected<void, holoscan::Error> result) {
      return result ? std::nullopt : std::optional<holoscan::Error>(std::move(result).error());
    };
    if (auto loan = allocate_point_cloud(cloud, organized_cloud(0, 5), {}, stream); !loan) {
      r.empty_organized = std::move(loan).error();
    }
    // Each emit consumes its loan, sent or not.
    const auto attempt = [&](const Header& header, const PointCloudShape& as,
                             std::int64_t valid) -> std::optional<holoscan::Error> {
      auto loan = allocate_point_cloud(cloud, shape, {}, stream);
      if (!loan) return std::move(loan).error();
      auto written = loan->write_host_and_commit(
          PointCloudFields::points, [](const holoscan::TensorOutputWriteGuard& writer) {
            std::memset(writer.data(), 0, 4 * 3 * sizeof(float));
          });
      if (!written) return std::move(written).error();
      return error_of(
          emit_point_cloud(cloud, std::move(*loan), header, as, valid,
                           holoscan::EmitOptions{.capture_time = context.activation_time()}));
    };
    r.long_frame_id = attempt(Header{std::string(kMaxFrameIdBytes + 1, 'x')}, shape, -1);
    r.other_shape = attempt(Header{"probe"}, organized_cloud(2, 2), -1);
    r.too_many_valid = attempt(Header{"probe"}, shape, 5);
    r.sent = !attempt(Header{"probe"}, shape, 4).has_value();
    return {};
  }

  holoscan::Output<schema::PointCloudT> cloud;

 private:
  std::shared_ptr<Refusals> refusals_;
  bool done_{};
};

// Counts the clouds it gets, and ends the run at the first.
class CountOp final : public holoscan::Operator<> {
 public:
  explicit CountOp(std::shared_ptr<Refusals> refusals) : refusals_(std::move(refusals)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(cloud, "cloud"); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{cloud});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = cloud.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    if (check_point_cloud(sample->data)) ++refusals_->received;
    context.request_completion();
    return {};
  }
  holoscan::Input<schema::PointCloudT> cloud;

 private:
  std::shared_ptr<Refusals> refusals_;
};

bool says(const std::optional<holoscan::Error>& error, std::string_view text) {
  if (!error) {
    std::cerr << "  no error, expected \"" << text << "\"\n";
    return false;
  }
  if (error->code == holoscan::ErrorCode::kInvalidArgument &&
      error->message.find(text) != std::string::npos) {
    return true;
  }
  std::cerr << "  error \"" << error->message << "\", expected \"" << text << "\"\n";
  return false;
}

void test_point_cloud_refusals() {
  auto refusals = std::make_shared<Refusals>();
  holoscan::Graph graph{"refusals"};
  auto probe = graph.op<RefusalProbeOp>("probe", refusals);
  auto count = graph.op<CountOp>("count", refusals);
  graph.add_flow(probe->cloud, count->cloud, test::lossless());
  const test::RunOutcome outcome = test::run_graph(graph);
  VERIFY_MSG(outcome.ok(), outcome.diagnostics);
  VERIFY(says(refusals->empty_organized, "an organized cloud needs H, W > 0"));
  VERIFY(says(refusals->long_frame_id, "the frame id has more than 256 bytes"));
  VERIFY(says(refusals->other_shape,
              "the points were allocated as [4, 3], not as the organized 2x2 cloud given"));
  VERIFY(says(refusals->too_many_valid, "valid_count must be -1 or 0..4, not 5"));
  VERIFY(refusals->sent && refusals->received == 1);
  // At setup: a pool size past 64 bits, and memory a pool cannot be in.
  holoscan::OperatorSpec spec;
  holoscan::Output<schema::PointCloudT> port;
  VERIFY(test::refuses(
      [&] {
        declare_point_cloud_output(spec, port, "cloud",
                                   {.max_points = std::uint64_t{1} << 62, .colors = true});
      },
      "max_points is too large"));
  VERIFY(test::refuses(
      [&] {
        declare_point_cloud_output(
            spec, port, "cloud",
            {.max_points = 4, .memory_kind = holoscan::MemoryKind::kCudaManaged});
      },
      "the memory kind must be host, pinned host or CUDA device memory"));
}

// --- check_point_cloud() on clouds made by hand ---------------------------------

std::shared_ptr<holoscan::Tensor> make(const std::vector<std::int64_t>& shape, DLDataType dtype,
                                       holoscan::MemoryStorageType storage,
                                       std::optional<std::vector<std::int64_t>> byte_strides = {}) {
  holoscan::TensorByteStridesView strides = std::nullopt;
  if (byte_strides) strides = std::span<const std::int64_t>(*byte_strides);
  auto tensor = holoscan::make_tensor(shape, dtype, strides, storage,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) throw std::runtime_error(tensor.error().message);
  return std::make_shared<holoscan::Tensor>(std::move(*tensor));
}

constexpr DLDataType kF32{kDLFloat, 32U, 1U};
constexpr DLDataType kU8{kDLUInt, 8U, 1U};
constexpr auto kPageable = holoscan::MemoryStorageType::kSystem;

// Four points with colors, all in pageable host memory.
schema::PointCloudT hand_made() {
  schema::PointCloudT cloud;
  cloud.header = std::make_shared<holoscan::schema::HeaderT>();
  cloud.points = make({4, 3}, kF32, kPageable);
  cloud.colors = make({4, 3}, kU8, kPageable);
  cloud.normals = make({0}, kF32, kPageable);
  cloud.intensity = make({0}, kF32, kPageable);
  cloud.valid_count = -1;
  return cloud;
}

bool refused(const schema::PointCloudT& cloud, std::string_view text) {
  auto checked = check_point_cloud(cloud);
  return says(checked ? std::nullopt : std::optional<holoscan::Error>(checked.error()), text);
}

void test_check_point_cloud() {
  VERIFY(check_point_cloud(hand_made()).has_value());
  schema::PointCloudT missing = hand_made();
  missing.colors.reset();
  VERIFY(refused(missing, "every message carries points, colors, normals and intensity"));
  // A missing tensor reads as an absent field.
  VERIFY(!is_present(layout_of(missing).colors));
  schema::PointCloudT long_id = hand_made();
  long_id.header->frame_id = std::string(kMaxFrameIdBytes + 1, 'x');
  VERIFY(refused(long_id, "point_cloud.header.frame_id is too long"));
  // Rows 16 bytes apart: a gap after each point's 12.
  schema::PointCloudT strided = hand_made();
  strided.points = make({4, 3}, kF32, kPageable, std::vector<std::int64_t>{16, 4});
  VERIFY(refused(strided, "point_cloud.points must be packed"));
  // One point: no element steps along its first axis, whatever its stride.
  schema::PointCloudT one = hand_made();
  one.points = make({1, 3}, kF32, kPageable, std::vector<std::int64_t>{64, 4});
  one.colors = make({1, 3}, kU8, kPageable);
  VERIFY(check_point_cloud(one).has_value());
  // Colors on the GPU, points in host memory; an absent field anywhere.
  schema::PointCloudT split = hand_made();
  split.colors = make({4, 3}, kU8, holoscan::MemoryStorageType::kDevice);
  VERIFY(refused(split, "point_cloud.colors must be in the same memory as the points"));
  schema::PointCloudT absent_elsewhere = hand_made();
  absent_elsewhere.normals = make({0}, kF32, holoscan::MemoryStorageType::kDevice);
  VERIFY(check_point_cloud(absent_elsewhere).has_value());
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
      {"the point cloud helpers refuse what consumers would", m::test_point_cloud_refusals},
      {"check_point_cloud() on clouds made by hand", m::test_check_point_cloud},
  });
}
