// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Graphs for the operator tests: a source of synthetic streams, each a known
// byte pattern in a given pixel format or tensor layout, in host or device
// memory; a sink that copies everything it receives to the host; and a runner.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
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
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/recorder/format.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace holoscan::recorder::test {

namespace schema = holoscan::schema;

constexpr DLDataType kU8{kDLUInt, 8U, 1U};
constexpr DLDataType kU16{kDLUInt, 16U, 1U};
constexpr DLDataType kU32{kDLUInt, 32U, 1U};
constexpr DLDataType kF16{kDLFloat, 16U, 1U};
constexpr DLDataType kF32{kDLFloat, 32U, 1U};
constexpr DLDataType kF64{kDLFloat, 64U, 1U};
constexpr DLDataType kI16x4{kDLInt, 16U, 4U};

// One synthetic stream.
struct Stream {
  std::string name;
  StreamKind kind{StreamKind::kImage};
  schema::ImageEncoding encoding{schema::ImageEncoding_CUSTOM};
  DLDataType dtype{kU8};
  std::vector<std::int64_t> shape;
  std::vector<PlaneLayout> planes;
  bool device{};
  // An image's size; zero for its shape's first two extents.
  std::int32_t width{};
  std::int32_t height{};

  [[nodiscard]] std::size_t bytes() const {
    std::size_t total = (static_cast<std::size_t>(dtype.bits) * dtype.lanes + 7U) / 8U;
    for (const std::int64_t extent : shape) total *= static_cast<std::size_t>(extent);
    return total;
  }
};

// Byte `i` of frame `frame` of stream `stream`: every byte value occurs, and
// no two streams or frames are alike.
inline std::uint8_t pattern(std::size_t stream, std::uint64_t frame, std::size_t i) {
  return static_cast<std::uint8_t>(stream * 131U + frame * 31U + i * 7U + (i >> 8U) * 13U);
}

inline std::vector<std::uint8_t> pattern_bytes(std::size_t stream, std::uint64_t frame,
                                               std::size_t bytes) {
  std::vector<std::uint8_t> out(bytes);
  for (std::size_t i = 0; i < bytes; ++i) out[i] = pattern(stream, frame, i);
  return out;
}

// What the source stamps on frame `frame`: its capture time, frame id, and
// image header capture time.
inline std::int64_t capture_ns(std::uint64_t frame) {
  return 1'000'000'000LL + static_cast<std::int64_t>(frame) * 10'000'000LL;
}
inline std::uint64_t frame_id_of(std::uint64_t frame) { return frame + 100U; }

// The image descriptor the source sends with frame `frame` of `stream`.
inline schema::ImageT descriptor_of(const Stream& stream, std::size_t index, std::uint64_t frame) {
  schema::ImageT image;
  const std::size_t rank = stream.shape.size();
  image.height = stream.height != 0 ? stream.height
                 : rank >= 2        ? static_cast<std::int32_t>(stream.shape[0])
                                    : 1;
  image.width = stream.width != 0 ? stream.width
                : rank >= 2       ? static_cast<std::int32_t>(stream.shape[1])
                                  : 1;
  image.encoding = stream.encoding;
  image.color_space = static_cast<schema::ColorSpace>(index % 7U);
  // The SDK checks this against the encoding's component width; MONO16 says 12.
  image.significant_bits = stream.encoding == schema::ImageEncoding_MONO16 ? 12U : 0U;
  image.roi_offset_x = static_cast<std::int32_t>(index);
  image.roi_offset_y = static_cast<std::int32_t>(frame);
  for (const PlaneLayout& plane : stream.planes) {
    image.plane_layouts.emplace_back(plane.offset_bytes, plane.row_stride_bytes, plane.size_bytes);
  }
  image.exposure_time_ns = 1000 + static_cast<std::int64_t>(frame);
  image.gain = 0.1F * static_cast<float>(index) + 1.0F / 3.0F;
  // Every other stream has no header, and one has an empty frame id.
  if (index % 2U == 0U) {
    image.header = std::make_shared<schema::HeaderT>();
    image.header->frame_id = index == 2U ? std::string() : stream.name + ",\"quoted\"";
    image.header->capture_timestamp_ns = capture_ns(frame) + 5;
    image.header->device_sequence = 7000U + frame;
  }
  return image;
}

// Every pixel format of schema::ImageEncoding, as a WxH image, and three tensors.
inline std::vector<Stream> every_format(int width, int height, bool device) {
  const std::int64_t w = width;
  const std::int64_t h = height;
  const std::uint64_t plane = static_cast<std::uint64_t>(w * h);
  const std::vector<PlaneLayout> nv12{{0, static_cast<std::uint32_t>(w), plane},
                                      {plane, static_cast<std::uint32_t>(w), plane / 2U}};
  const std::vector<PlaneLayout> i420{
      {0, static_cast<std::uint32_t>(w), plane},
      {plane, static_cast<std::uint32_t>(w / 2), plane / 4U},
      {plane + plane / 4U, static_cast<std::uint32_t>(w / 2), plane / 4U}};
  const std::vector<PlaneLayout> planar{{0, static_cast<std::uint32_t>(w), plane},
                                        {plane, static_cast<std::uint32_t>(w), plane},
                                        {2U * plane, static_cast<std::uint32_t>(w), plane}};
  const std::vector<PlaneLayout> p010{{0, static_cast<std::uint32_t>(2 * w), 2U * plane},
                                      {2U * plane, static_cast<std::uint32_t>(2 * w), plane}};
  using E = schema::ImageEncoding;
  const auto image = [&](const char* name, E encoding, DLDataType dtype,
                         std::vector<std::int64_t> shape, std::vector<PlaneLayout> planes = {}) {
    return Stream{.name = name,
                  .kind = StreamKind::kImage,
                  .encoding = encoding,
                  .dtype = dtype,
                  .shape = std::move(shape),
                  .planes = std::move(planes),
                  .device = device,
                  .width = width,
                  .height = height};
  };
  const auto tensor = [&](const char* name, DLDataType dtype, std::vector<std::int64_t> shape) {
    return Stream{.name = name,
                  .kind = StreamKind::kTensor,
                  .dtype = dtype,
                  .shape = std::move(shape),
                  .device = device};
  };
  return {
      image("custom", schema::ImageEncoding_CUSTOM, kF16, {h, w}),
      image("rgb8", schema::ImageEncoding_RGB8, kU8, {h, w, 3}),
      image("rgba8", schema::ImageEncoding_RGBA8, kU8, {h, w, 4}),
      image("bgr8", schema::ImageEncoding_BGR8, kU8, {h, w, 3}),
      image("bgra8", schema::ImageEncoding_BGRA8, kU8, {h, w, 4}),
      image("r8_g8_b8", schema::ImageEncoding_R8_G8_B8, kU8, {3, h, w}, planar),
      image("b8_g8_r8", schema::ImageEncoding_B8_G8_R8, kU8, {3, h, w}, planar),
      image("mono8", schema::ImageEncoding_MONO8, kU8, {h, w}),
      image("mono16", schema::ImageEncoding_MONO16, kU16, {h, w}),
      image("mono32", schema::ImageEncoding_MONO32, kU32, {h, w}),
      image("mono32f", schema::ImageEncoding_MONO32F, kF32, {h, w}),
      image("yuyv", schema::ImageEncoding_YUYV, kU8, {h, w, 2}),
      image("uyvy", schema::ImageEncoding_UYVY, kU8, {h, w, 2}),
      image("nv12", schema::ImageEncoding_NV12, kU8, {h * 3 / 2, w}, nv12),
      image("i420", schema::ImageEncoding_I420, kU8, {h * 3 / 2, w}, i420),
      image("rgb32f", schema::ImageEncoding_RGB32F, kF32, {h, w, 3}),
      image("rgba32f", schema::ImageEncoding_RGBA32F, kF32, {h, w, 4}),
      image("yuv32f", schema::ImageEncoding_YUV32F, kF32, {h, w, 3}),
      image("p010", schema::ImageEncoding_P010, kU16, {h * 3 / 2, w}, p010),
      image("rgba1010102", schema::ImageEncoding_RGBA1010102, kU32, {h, w}),
      image("bayer_rggb", schema::ImageEncoding_BAYER_RGGB, kU8, {h, w}),
      image("bayer_bggr", schema::ImageEncoding_BAYER_BGGR, kU8, {h, w}),
      image("bayer_grbg", schema::ImageEncoding_BAYER_GRBG, kU8, {h, w}),
      image("bayer_gbrg", schema::ImageEncoding_BAYER_GBRG, kU8, {h, w}),
      image("rgb16", schema::ImageEncoding_RGB16, kU16, {h, w, 3}),
      image("rgba16", schema::ImageEncoding_RGBA16, kU16, {h, w, 4}),
      tensor("imu", kF64, {6}),
      tensor("lanes", kI16x4, {5, 3}),
      tensor("scalar", kF32, {}),
  };
}

inline holoscan::TensorOutputSpec output_spec(const Stream& stream, std::size_t max_bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind =
                  stream.device ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost,
              .dtype = stream.dtype,
              .rank = static_cast<std::uint32_t>(stream.shape.size()),
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(std::max<std::size_t>(max_bytes, 1U)),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

struct SourceConfig {
  std::vector<Stream> streams;
  std::uint64_t frames{};
  std::chrono::nanoseconds period{std::chrono::milliseconds{5}};
  // Compute calls after which the source closes its outputs; enough to send
  // every frame, with room for retries.
  std::uint64_t calls{};
  // Live: a frame a consumer refuses is dropped, not sent again.
  bool live{};
  // Frames sent in full, and frames a consumer refused.
  std::shared_ptr<std::atomic<std::uint64_t>> sent;
  std::shared_ptr<std::atomic<std::uint64_t>> refused;
};

// Sends `frames` frames, each on every stream, one frame per period.
class PatternSourceOp final : public holoscan::Operator<> {
 public:
  explicit PatternSourceOp(SourceConfig config) : config_(std::move(config)) {
    for (const Stream& stream : config_.streams) {
      if (stream.kind == StreamKind::kImage) {
        images_.push_back(std::make_unique<holoscan::Output<schema::ImageT>>());
        tensors_.emplace_back();
      } else {
        images_.emplace_back();
        tensors_.push_back(std::make_unique<holoscan::Output<holoscan::Tensor>>());
      }
    }
  }

  void setup(holoscan::OperatorSpec& spec) override {
    for (std::size_t i = 0; i < config_.streams.size(); ++i) {
      const Stream& stream = config_.streams[i];
      if (images_[i]) {
        spec.output(*images_[i], stream.name)
            .produces_tensor(output_spec(stream, stream.bytes()))
            .payload_options(
                holoscan::PayloadOptions{.schema = holoscan::schema_identity<schema::ImageT>()});
      } else {
        spec.output(*tensors_[i], stream.name).produces_tensor(output_spec(stream, stream.bytes()));
      }
    }
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = config_.period})
        .complete_after(holoscan::ComputeCalls{.limit = config_.calls});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (pending_.empty()) {
      if (frame_ >= config_.frames) return {};
      for (std::size_t i = 0; i < config_.streams.size(); ++i) pending_.push_back(i);
    }
    while (!pending_.empty()) {
      auto sent = send(pending_.front(), context);
      if (!sent && sent.error().code != holoscan::ErrorCode::kBackpressured) {
        return holoscan::make_unexpected(holoscan::Error{
            sent.error().code, "PatternSourceOp: stream " + config_.streams[pending_.front()].name +
                                   ": " + sent.error().message});
      }
      if (!sent) {
        if (sent.error().code != holoscan::ErrorCode::kBackpressured) return sent;
        if (config_.refused) ++*config_.refused;
        if (!config_.live) return {};  // Send the rest of the frame on the next tick.
        pending_.clear();              // A live frame is not worth waiting for.
        ++frame_;
        return {};
      }
      pending_.erase(pending_.begin());
    }
    if (config_.sent) ++*config_.sent;
    ++frame_;
    return {};
  }

  [[nodiscard]] const holoscan::Output<schema::ImageT>& image(const std::string& name) const {
    return *images_[index(name)];
  }
  [[nodiscard]] const holoscan::Output<holoscan::Tensor>& tensor(const std::string& name) const {
    return *tensors_[index(name)];
  }

 private:
  [[nodiscard]] std::size_t index(const std::string& name) const {
    for (std::size_t i = 0; i < config_.streams.size(); ++i) {
      if (config_.streams[i].name == name) return i;
    }
    throw std::out_of_range("no stream " + name);
  }

  holoscan::expected<void, holoscan::Error> send(std::size_t i,
                                                 holoscan::ExecutionContext& context) {
    const Stream& stream = config_.streams[i];
    const std::vector<std::uint8_t> bytes = pattern_bytes(i, frame_, stream.bytes());
    const holoscan::TensorLoanRequest request{.shape = stream.shape, .dtype = stream.dtype};
    auto loan =
        images_[i] ? images_[i]->allocate_tensor(request) : tensors_[i]->allocate_tensor(request);
    if (!loan) return holoscan::make_unexpected(std::move(loan).error());
    holoscan::expected<void, holoscan::Error> written;
    if (stream.device) {
      written = loan->write_and_commit(
          context.cuda_stream(),
          [&](const holoscan::TensorOutputWriteGuard& writer)
              -> holoscan::expected<void, holoscan::Error> {
            if (cudaMemcpyAsync(writer.data(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice,
                                context.cuda_stream().get()) != cudaSuccess) {
              return holoscan::make_unexpected(
                  holoscan::Error{holoscan::ErrorCode::kFailure, "cudaMemcpyAsync"});
            }
            return {};
          });
    } else {
      written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer) {
        std::memcpy(writer.data(), bytes.data(), bytes.size());
      });
    }
    if (!written) return written;
    const holoscan::EmitOptions options{.capture_time = context.clock().at_ns(capture_ns(frame_)),
                                        .frame_id = frame_id_of(frame_)};
    if (images_[i]) {
      return images_[i]->emit_tensor(std::move(*loan), descriptor_of(stream, i, frame_), options);
    }
    return tensors_[i]->emit(std::move(*loan), options);
  }

  SourceConfig config_;
  std::vector<std::unique_ptr<holoscan::Output<schema::ImageT>>> images_;
  std::vector<std::unique_ptr<holoscan::Output<holoscan::Tensor>>> tensors_;
  std::vector<std::size_t> pending_;
  std::uint64_t frame_{};
};

// One sample as the sink saw it.
struct Received {
  std::int64_t capture_ns{};
  bool capture_valid{};
  std::uint64_t frame_id{};
  DLDataType dtype{};
  std::vector<std::int64_t> shape;
  DLDeviceType device{};
  std::optional<schema::ImageT> image;  // Without its tensor.
  std::vector<std::uint8_t> bytes;
  // When it arrived, on the steady clock.
  std::chrono::steady_clock::time_point arrival;
};

struct Collected {
  std::mutex mutex;
  std::map<std::string, std::vector<Received>> streams;
  std::atomic<std::uint64_t> count{0};
};

struct SinkConfig {
  std::vector<Stream> streams;  // Only the names and kinds matter.
  // Samples after which the run completes; 0 never completes.
  std::uint64_t expected{};
  std::shared_ptr<Collected> collected;
};

// Takes every sample of every input, copies it to the host, and ends the run
// after `expected` samples.
class CollectSinkOp final : public holoscan::Operator<> {
 public:
  explicit CollectSinkOp(SinkConfig config) : config_(std::move(config)) {
    for (const Stream& stream : config_.streams) {
      if (stream.kind == StreamKind::kImage) {
        images_.push_back(std::make_unique<holoscan::Input<schema::ImageT>>());
        tensors_.emplace_back();
      } else {
        images_.emplace_back();
        tensors_.push_back(std::make_unique<holoscan::Input<holoscan::Tensor>>());
      }
    }
  }

  void setup(holoscan::OperatorSpec& spec) override {
    for (std::size_t i = 0; i < config_.streams.size(); ++i) {
      if (images_[i]) {
        spec.input(*images_[i], config_.streams[i].name).queue_depth(4U);
      } else {
        spec.input(*tensors_[i], config_.streams[i].name).queue_depth(4U);
      }
    }
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAnyInput{});
    for (std::size_t i = 0; i < config_.streams.size(); ++i) {
      if (images_[i]) {
        result.input(*images_[i], holoscan::DrainAll{});
      } else {
        result.input(*tensors_[i], holoscan::DrainAll{});
      }
    }
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    using Result = holoscan::expected<void, holoscan::Error>;
    const auto arrival = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < config_.streams.size(); ++i) {
      const std::string& name = config_.streams[i].name;
      Result result;
      if (images_[i]) {
        result = images_[i]->for_each_selected(
            [&](const holoscan::Sample<schema::ImageT>& sample) -> Result {
              schema::ImageT image = sample.data;
              image.data.reset();
              return take(name, *sample.data.data, sample.metadata, std::move(image), arrival,
                          context);
            });
      } else {
        result = tensors_[i]->for_each_selected(
            [&](const holoscan::Sample<holoscan::Tensor>& sample) -> Result {
              return take(name, sample.data, sample.metadata, std::nullopt, arrival, context);
            });
      }
      if (!result && result.error().code != holoscan::ErrorCode::kInputEmpty &&
          result.error().code != holoscan::ErrorCode::kInputNotSelected) {
        return result;
      }
    }
    return {};
  }

  [[nodiscard]] const holoscan::Input<schema::ImageT>& image(const std::string& name) const {
    return *images_[index(name)];
  }
  [[nodiscard]] const holoscan::Input<holoscan::Tensor>& tensor(const std::string& name) const {
    return *tensors_[index(name)];
  }

 private:
  [[nodiscard]] std::size_t index(const std::string& name) const {
    for (std::size_t i = 0; i < config_.streams.size(); ++i) {
      if (config_.streams[i].name == name) return i;
    }
    throw std::out_of_range("no stream " + name);
  }

  holoscan::expected<void, holoscan::Error> take(const std::string& name,
                                                 const holoscan::Tensor& tensor,
                                                 const holoscan::SampleMetadata& metadata,
                                                 std::optional<schema::ImageT> image,
                                                 std::chrono::steady_clock::time_point arrival,
                                                 holoscan::ExecutionContext& context) {
    Received received;
    received.capture_ns = metadata.capture_timestamp_ns;
    received.capture_valid = metadata.capture_time().valid();
    received.frame_id = metadata.frame_id;
    received.dtype = tensor.dtype();
    const std::span<const std::int64_t> shape = tensor.shape_span();
    received.shape.assign(shape.begin(), shape.end());
    received.device = tensor.device().device_type;
    received.image = std::move(image);
    received.arrival = arrival;
    received.bytes.resize(static_cast<std::size_t>(tensor.nbytes()));
    if (!received.bytes.empty()) {
      if (auto copied = tensor.copy_to_host(received.bytes.data(), received.bytes.size(),
                                            context.cuda_stream());
          !copied) {
        return copied;
      }
    }
    {
      const std::lock_guard lock(config_.collected->mutex);
      config_.collected->streams[name].push_back(std::move(received));
    }
    if (++config_.collected->count == config_.expected) {
      context.request_completion();
    }
    return {};
  }

  SinkConfig config_;
  std::vector<std::unique_ptr<holoscan::Input<schema::ImageT>>> images_;
  std::vector<std::unique_ptr<holoscan::Input<holoscan::Tensor>>> tensors_;
};

struct RunOutcome {
  bool compiled{};
  // Ended by itself: naturally, or by a sink's request.
  bool completed{};
  holoscan::RunTermination termination{};
  std::string diagnostics;
  std::chrono::steady_clock::duration took{};
};

// Compiles and runs `graph`, stopping it after `timeout`.
inline RunOutcome run_graph(holoscan::Graph& graph, holoscan::CompileOptions options = {},
                            std::chrono::seconds timeout = std::chrono::seconds{120}) {
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  RunOutcome outcome;
  if (!plan.ok()) {
    outcome.diagnostics = holoscan::format_diagnostics(plan.diagnostics(), plan.graph());
    return outcome;
  }
  outcome.compiled = true;
  const auto started = std::chrono::steady_clock::now();
  holoscan::RunSession session = holoscan::run_async(plan);
  if (!session.wait_for(timeout)) {
    session.request_stop();
  }
  session.wait();
  outcome.took = std::chrono::steady_clock::now() - started;
  outcome.termination = session.termination();
  outcome.completed = (outcome.termination == holoscan::RunTermination::kRequestedCompleted ||
                       outcome.termination == holoscan::RunTermination::kCompleted) &&
                      session.diagnostics().empty();
  if (!session.diagnostics().empty()) {
    outcome.diagnostics = holoscan::format_diagnostics(session.diagnostics(), plan.graph());
  }
  return outcome;
}

// A fresh folder for a test's recording, under the test's working folder.
inline std::filesystem::path fresh_folder(const std::string& name) {
  const std::filesystem::path folder = std::filesystem::current_path() / "recordings" / name;
  std::filesystem::remove_all(folder);
  std::filesystem::create_directories(folder.parent_path());
  return folder;
}

}  // namespace holoscan::recorder::test
