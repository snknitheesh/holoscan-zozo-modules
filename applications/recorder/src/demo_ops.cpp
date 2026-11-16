// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "demo_ops.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>

namespace recorder_app {
namespace {

using Result = holoscan::expected<void, holoscan::Error>;
using holoscan::recorder::PlaneLayout;
using holoscan::recorder::StreamKind;

constexpr DLDataType kU8{kDLUInt, 8U, 1U};
constexpr DLDataType kU16{kDLUInt, 16U, 1U};
constexpr DLDataType kF32{kDLFloat, 32U, 1U};

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, message};
}

bool nothing(const holoscan::Error& error) {
  return error.code == holoscan::ErrorCode::kInputEmpty ||
         error.code == holoscan::ErrorCode::kInputNotSelected;
}

// The image descriptor frame `frame` of `stream` goes out with.
schema::ImageT descriptor_of(const Stream& stream, std::uint64_t frame) {
  schema::ImageT image;
  image.width = stream.width;
  image.height = stream.height;
  image.encoding = stream.encoding;
  for (const PlaneLayout& plane : stream.planes) {
    image.plane_layouts.emplace_back(plane.offset_bytes, plane.row_stride_bytes, plane.size_bytes);
  }
  image.header = std::make_shared<schema::HeaderT>();
  image.header->frame_id = stream.name;
  image.header->capture_timestamp_ns = capture_ns(frame);
  image.header->device_sequence = frame;
  return image;
}

std::int64_t steady_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

std::size_t Stream::bytes() const {
  std::size_t total = (static_cast<std::size_t>(dtype.bits) * dtype.lanes + 7U) / 8U;
  for (const std::int64_t extent : shape) total *= static_cast<std::size_t>(extent);
  return total;
}

std::vector<Stream> ci_streams(int width, int height) {
  const std::int64_t w = width;
  const std::int64_t h = height;
  const auto luma = static_cast<std::uint64_t>(w * h);
  const auto image = [&](const char* name, schema::ImageEncoding encoding, DLDataType dtype,
                         std::vector<std::int64_t> shape, bool device,
                         std::vector<PlaneLayout> planes = {}) {
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
  return {
      image("rgb", schema::ImageEncoding_RGB8, kU8, {h, w, 3}, true),
      image("mono", schema::ImageEncoding_MONO8, kU8, {h, w}, false),
      image("depth", schema::ImageEncoding_MONO16, kU16, {h, w}, true),
      image("nv12", schema::ImageEncoding_NV12, kU8, {h * 3 / 2, w}, false,
            {{0, static_cast<std::uint32_t>(w), luma},
             {luma, static_cast<std::uint32_t>(w), luma / 2U}}),
      Stream{.name = "imu", .kind = StreamKind::kTensor, .dtype = kF32, .shape = {6}},
  };
}

Stream bench_stream(const std::string& encoding, int width, int height, bool device) {
  const bool rgb = encoding == "rgb8";
  return Stream{.name = "camera",
                .kind = StreamKind::kImage,
                .encoding = rgb ? schema::ImageEncoding_RGB8 : schema::ImageEncoding_MONO8,
                .dtype = kU8,
                .shape = rgb ? std::vector<std::int64_t>{height, width, 3}
                             : std::vector<std::int64_t>{height, width},
                .device = device,
                .width = width,
                .height = height};
}

std::uint8_t pattern(std::size_t stream, std::uint64_t frame, std::size_t i) {
  return static_cast<std::uint8_t>(stream * 131U + frame * 31U + i * 7U + (i >> 8U) * 13U);
}

std::int64_t capture_ns(std::uint64_t frame) {
  return 1'700'000'000'000'000'000LL + static_cast<std::int64_t>(frame) * 33'333'333LL;
}

std::uint64_t frame_id_of(std::uint64_t frame) { return frame + 1U; }

// ---------------------------------------------------------------------------

SyntheticSourceOp::SyntheticSourceOp(SourceConfig config) : config_(std::move(config)) {
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

void SyntheticSourceOp::setup(holoscan::OperatorSpec& spec) {
  for (std::size_t i = 0; i < config_.streams.size(); ++i) {
    const Stream& stream = config_.streams[i];
    const holoscan::TensorOutputSpec tensor_spec{
        .representation =
            holoscan::TensorRepresentation{
                .memory_kind =
                    stream.device ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost,
                .dtype = stream.dtype,
                .rank = static_cast<std::uint32_t>(stream.shape.size()),
                .strides = holoscan::TensorStridePolicy::kContiguousOnly,
            },
        .bounds = holoscan::tensor_bounds(stream.bytes()),
        .storage = holoscan::TensorOutputStorage::kRuntimePool,
    };
    if (images_[i]) {
      spec.output(*images_[i], stream.name)
          .produces_tensor(tensor_spec)
          .payload_options(
              holoscan::PayloadOptions{.schema = holoscan::schema_identity<schema::ImageT>()});
    } else {
      spec.output(*tensors_[i], stream.name).produces_tensor(tensor_spec);
    }
  }
}

holoscan::Contract SyntheticSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period})
      .complete_after(
          holoscan::ComputeCalls{.limit = config_.live ? config_.frames : config_.frames * 20U});
  return result;
}

std::size_t SyntheticSourceOp::index(const std::string& name) const {
  for (std::size_t i = 0; i < config_.streams.size(); ++i) {
    if (config_.streams[i].name == name) return i;
  }
  throw std::out_of_range("no stream " + name);
}

const holoscan::Output<schema::ImageT>& SyntheticSourceOp::image(const std::string& name) const {
  return *images_.at(index(name));
}

const holoscan::Output<holoscan::Tensor>& SyntheticSourceOp::tensor(const std::string& name) const {
  return *tensors_.at(index(name));
}

holoscan::expected<void, holoscan::Error> SyntheticSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (pending_.empty()) {
    if (frame_ >= config_.frames) return {};
    for (std::size_t i = 0; i < config_.streams.size(); ++i) pending_.push_back(i);
    if (config_.stats && frame_ == 0) config_.stats->first_ns = steady_ns();
  }
  while (!pending_.empty()) {
    auto sent = send(pending_.front(), context);
    if (!sent) {
      if (sent.error().code != holoscan::ErrorCode::kBackpressured) return sent;
      if (!config_.live) return {};  // Send the rest of the frame on the next tick.
      // A live frame is not worth waiting for.
      if (config_.stats) ++config_.stats->dropped;
      pending_.clear();
      ++frame_;
      return {};
    }
    pending_.erase(pending_.begin());
  }
  if (config_.stats) {
    ++config_.stats->sent;
    config_.stats->last_ns = steady_ns();
  }
  ++frame_;
  return {};
}

Result SyntheticSourceOp::send(std::size_t i, holoscan::ExecutionContext& context) {
  const Stream& stream = config_.streams[i];
  const std::size_t bytes = stream.bytes();
  scratch_.resize(config_.streams.size());
  std::vector<std::uint8_t>& data = scratch_[i];
  if (!config_.reuse_pattern || data.size() != bytes) {
    data.resize(bytes);
    for (std::size_t b = 0; b < bytes; ++b) data[b] = pattern(i, frame_, b);
  }
  if (config_.reuse_pattern) {
    std::memcpy(data.data(), &frame_, std::min(sizeof(frame_), bytes));
  }
  const holoscan::TensorLoanRequest request{.shape = stream.shape, .dtype = stream.dtype};
  auto loan =
      images_[i] ? images_[i]->allocate_tensor(request) : tensors_[i]->allocate_tensor(request);
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  Result written;
  if (stream.device) {
    written = loan->write_and_commit(
        context.cuda_stream(), [&](const holoscan::TensorOutputWriteGuard& writer) -> Result {
          // From pageable memory the copy is staged before it returns.
          if (const cudaError_t status =
                  cudaMemcpyAsync(writer.data(), data.data(), bytes, cudaMemcpyHostToDevice,
                                  context.cuda_stream().get());
              status != cudaSuccess) {
            return holoscan::make_unexpected(
                failure(std::string("cudaMemcpyAsync: ") + cudaGetErrorString(status)));
          }
          return {};
        });
  } else {
    written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer) {
      std::memcpy(writer.data(), data.data(), bytes);
    });
  }
  if (!written) return written;
  const holoscan::EmitOptions options{.capture_time = context.clock().at_ns(capture_ns(frame_)),
                                      .frame_id = frame_id_of(frame_)};
  if (images_[i]) {
    return images_[i]->emit_tensor(std::move(*loan), descriptor_of(stream, frame_), options);
  }
  return tensors_[i]->emit(std::move(*loan), options);
}

// ---------------------------------------------------------------------------

VerifySinkOp::VerifySinkOp(VerifyConfig config) : config_(std::move(config)) {
  for (const Stream& stream : config_.streams) {
    if (stream.kind == StreamKind::kImage) {
      images_.push_back(std::make_unique<holoscan::Input<schema::ImageT>>());
      tensors_.emplace_back();
    } else {
      images_.emplace_back();
      tensors_.push_back(std::make_unique<holoscan::Input<holoscan::Tensor>>());
    }
  }
  next_.assign(config_.streams.size(), 0);
}

void VerifySinkOp::setup(holoscan::OperatorSpec& spec) {
  for (std::size_t i = 0; i < config_.streams.size(); ++i) {
    if (images_[i]) {
      spec.input(*images_[i], config_.streams[i].name).queue_depth(4U);
    } else {
      spec.input(*tensors_[i], config_.streams[i].name).queue_depth(4U);
    }
  }
}

holoscan::Contract VerifySinkOp::contract() const {
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

std::size_t VerifySinkOp::index(const std::string& name) const {
  for (std::size_t i = 0; i < config_.streams.size(); ++i) {
    if (config_.streams[i].name == name) return i;
  }
  throw std::out_of_range("no stream " + name);
}

const holoscan::Input<schema::ImageT>& VerifySinkOp::image(const std::string& name) const {
  return *images_.at(index(name));
}

const holoscan::Input<holoscan::Tensor>& VerifySinkOp::tensor(const std::string& name) const {
  return *tensors_.at(index(name));
}

Result VerifySinkOp::check(std::size_t i, const holoscan::Tensor& tensor,
                           const schema::ImageT* image, const holoscan::SampleMetadata& metadata,
                           holoscan::ExecutionContext& context) {
  const Stream& stream = config_.streams[i];
  const std::uint64_t frame = next_[i]++;
  ++config_.stats->samples;
  bool ok = metadata.capture_time().valid() && metadata.capture_timestamp_ns == capture_ns(frame) &&
            metadata.frame_id == frame_id_of(frame);
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const DLDataType dtype = tensor.dtype();
  ok = ok && std::equal(shape.begin(), shape.end(), stream.shape.begin(), stream.shape.end()) &&
       dtype.code == stream.dtype.code && dtype.bits == stream.dtype.bits &&
       dtype.lanes == stream.dtype.lanes;
  if (image != nullptr) {
    const schema::ImageT want = descriptor_of(stream, frame);
    ok = ok && image->width == want.width && image->height == want.height &&
         image->encoding == want.encoding &&
         image->plane_layouts.size() == want.plane_layouts.size() && image->header &&
         image->header->frame_id == want.header->frame_id &&
         image->header->capture_timestamp_ns == want.header->capture_timestamp_ns &&
         image->header->device_sequence == want.header->device_sequence;
  }
  if (ok && static_cast<std::size_t>(tensor.nbytes()) == stream.bytes()) {
    host_.resize(stream.bytes());
    if (auto copied = tensor.copy_to_host(host_.data(), host_.size(), context.cuda_stream());
        !copied) {
      return copied;
    }
    for (std::size_t b = 0; b < host_.size() && ok; ++b) ok = host_[b] == pattern(i, frame, b);
  } else {
    ok = false;
  }
  if (!ok) {
    ++config_.stats->mismatches;
    HOLOSCAN_LOG_ERROR("recorder: {} frame {} differs from what was recorded", stream.name, frame);
  }
  return {};
}

holoscan::expected<void, holoscan::Error> VerifySinkOp::compute(
    holoscan::ExecutionContext& context) {
  for (std::size_t i = 0; i < config_.streams.size(); ++i) {
    Result result;
    if (images_[i]) {
      result = images_[i]->for_each_selected(
          [&](const holoscan::Sample<schema::ImageT>& sample) -> Result {
            if (!sample.data.data) return holoscan::make_unexpected(failure("no tensor"));
            return check(i, *sample.data.data, &sample.data, sample.metadata, context);
          });
    } else {
      result = tensors_[i]->for_each_selected(
          [&](const holoscan::Sample<holoscan::Tensor>& sample) -> Result {
            return check(i, sample.data, nullptr, sample.metadata, context);
          });
    }
    if (!result && !nothing(result.error())) return result;
  }
  if (std::all_of(next_.begin(), next_.end(),
                  [this](std::uint64_t n) { return n >= config_.frames; })) {
    context.request_completion();
  }
  return {};
}

// ---------------------------------------------------------------------------

CountSinkOp::CountSinkOp(std::uint64_t count, std::shared_ptr<CountStats> stats)
    : count_(count), stats_(std::move(stats)) {}

void CountSinkOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(image, "image").queue_depth(1U);
}

holoscan::Contract CountSinkOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

holoscan::expected<void, holoscan::Error> CountSinkOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) return holoscan::make_unexpected(std::move(sample).error());
  if (stats_->frames == 0) stats_->first_capture_ns = sample->metadata.capture_timestamp_ns;
  stats_->last_capture_ns = sample->metadata.capture_timestamp_ns;
  if (++stats_->frames == count_) context.request_completion();
  return {};
}

}  // namespace recorder_app
