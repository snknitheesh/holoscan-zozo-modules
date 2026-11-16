// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/recorder/replayer_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
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

namespace holoscan::recorder {
namespace {

using Result = holoscan::expected<void, holoscan::Error>;

// Pinned buffers a device replay reads into before copying to the GPU: enough
// that the copy from one has finished by the time the next read reuses it.
constexpr std::size_t kStagingBuffers = 4;
// A pass's length is its first to last capture time plus one interval: the
// median between samples, or this when there is no interval.
constexpr std::int64_t kDefaultIntervalNs = 33'333'333;

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "ReplayerOp: " + message};
}

std::string cuda_text(const char* what, cudaError_t status) {
  return std::string(what) + ": " + cudaGetErrorString(status);
}

std::optional<holoscan::schema::ImageEncoding> encoding_named(std::string_view name) {
  for (const holoscan::schema::ImageEncoding value : holoscan::schema::EnumValuesImageEncoding()) {
    if (name == holoscan::schema::EnumNameImageEncoding(value)) return value;
  }
  return std::nullopt;
}

std::optional<holoscan::schema::ColorSpace> color_space_named(std::string_view name) {
  for (const holoscan::schema::ColorSpace value : holoscan::schema::EnumValuesColorSpace()) {
    if (name == holoscan::schema::EnumNameColorSpace(value)) return value;
  }
  return std::nullopt;
}

DLDataType dl_type(ElementType type) {
  return DLDataType{.code = type.code, .bits = type.bits, .lanes = type.lanes};
}

// Whether the consumer is busy: the sample goes out again on the next tick.
bool refused(const holoscan::Error& error) {
  return error.code == holoscan::ErrorCode::kBackpressured ||
         error.code == holoscan::ErrorCode::kResourceExhausted;
}

}  // namespace

struct ReplayerOp::Port {
  std::string name;
  std::uint32_t stream{};
  bool image{};
  std::unique_ptr<holoscan::Output<holoscan::schema::ImageT>> image_port;
  std::unique_ptr<holoscan::Output<holoscan::Tensor>> tensor_port;
  // Whether it has sent a sample on this tick.
  bool sent{};
};

struct ReplayerOp::Staging {
  struct Buffer {
    void* data{};
    cudaEvent_t event{};
    bool in_use{};
  };
  std::array<Buffer, kStagingBuffers> buffers{};
  std::size_t next{};
  std::size_t capacity{};

  void release() noexcept {
    for (Buffer& buffer : buffers) {
      if (buffer.in_use) cudaEventSynchronize(buffer.event);
      if (buffer.event != nullptr) cudaEventDestroy(buffer.event);
      if (buffer.data != nullptr) cudaFreeHost(buffer.data);
      buffer = {};
    }
  }
};

void validate_config(const ReplayerConfig& config) {
  if (config.directory.empty()) {
    throw std::invalid_argument("directory must not be empty");
  }
  if (config.first_frame >= config.end_frame) {
    throw std::invalid_argument("first_frame must be below end_frame");
  }
  if (config.period < std::chrono::microseconds{10} || config.period > std::chrono::seconds{1}) {
    throw std::invalid_argument("period must be 10 us..1 s");
  }
  for (std::size_t i = 0; i < config.streams.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      if (config.streams[i] == config.streams[j]) {
        throw std::invalid_argument("streams names \"" + config.streams[i] + "\" twice");
      }
    }
  }
}

ReplayerOp::ReplayerOp(ReplayerConfig config) : config_(std::move(config)) {
  validate_config(config_);
  reader_ = std::make_unique<RecordingReader>(config_.directory);
  const std::vector<StreamInfo>& streams = reader_->streams();

  // The streams played, in the recording's order.
  std::vector<bool> played(streams.size(), config_.streams.empty());
  for (const std::string& name : config_.streams) {
    const std::optional<std::uint32_t> index = reader_->find_stream(name);
    if (!index) {
      std::string names;
      for (const StreamInfo& stream : streams) {
        names += (names.empty() ? "" : ", ") + stream.name;
      }
      throw std::invalid_argument("the recording has no stream \"" + name + "\"; it has " + names);
    }
    played[*index] = true;
  }
  output_of_stream_.assign(streams.size(), std::nullopt);
  for (std::uint32_t s = 0; s < streams.size(); ++s) {
    if (!played[s]) continue;
    auto port = std::make_unique<Port>();
    port->name = streams[s].name;
    port->stream = s;
    port->image = streams[s].kind == StreamKind::kImage;
    if (port->image) {
      port->image_port = std::make_unique<holoscan::Output<holoscan::schema::ImageT>>();
    } else {
      port->tensor_port = std::make_unique<holoscan::Output<holoscan::Tensor>>();
    }
    output_of_stream_[s] = outputs_.size();
    outputs_.push_back(std::move(port));
  }

  // The samples played, in capture-time order; the index's order breaks ties.
  const std::vector<Record>& records = reader_->records();
  for (std::size_t i = 0; i < records.size(); ++i) {
    const Record& record = records[i];
    if (!output_of_stream_[record.stream] || record.sequence < config_.first_frame ||
        record.sequence >= config_.end_frame) {
      continue;
    }
    if (record.image && (!encoding_named(record.image->encoding) ||
                         !color_space_named(record.image->color_space))) {
      throw RecordingError(config_.directory + ": stream \"" + streams[record.stream].name +
                           "\" has an image with encoding " + record.image->encoding +
                           " and color space " + record.image->color_space +
                           ", which this SDK does not have");
    }
    order_.push_back(i);
  }
  if (order_.empty()) {
    throw std::invalid_argument(config_.directory +
                                ": no samples of the streams played in frames " +
                                std::to_string(config_.first_frame) + " to " +
                                (config_.end_frame == std::numeric_limits<std::uint64_t>::max()
                                     ? std::string("the end")
                                     : std::to_string(config_.end_frame)));
  }
  std::stable_sort(order_.begin(), order_.end(), [&records](std::size_t a, std::size_t b) {
    return records[a].timestamp_ns < records[b].timestamp_ns;
  });

  // A pass's length and frame ids, by which a loop moves each later pass on.
  std::vector<std::int64_t> intervals;
  for (std::size_t i = 1; i < order_.size(); ++i) {
    const std::int64_t gap = records[order_[i]].timestamp_ns - records[order_[i - 1]].timestamp_ns;
    if (gap > 0) intervals.push_back(gap);
  }
  std::int64_t interval = kDefaultIntervalNs;
  if (!intervals.empty()) {
    std::nth_element(intervals.begin(),
                     intervals.begin() + static_cast<std::ptrdiff_t>(intervals.size() / 2),
                     intervals.end());
    interval = intervals[intervals.size() / 2];
  }
  pass_span_ns_ =
      records[order_.back()].timestamp_ns - records[order_.front()].timestamp_ns + interval;
  for (const std::size_t i : order_) {
    pass_frames_ = std::max(pass_frames_, records[i].frame_id);
  }
}

ReplayerOp::~ReplayerOp() {
  if (staging_) staging_->release();
}

void ReplayerOp::setup(holoscan::OperatorSpec& spec) {
  const std::vector<StreamInfo>& streams = reader_->streams();
  for (const std::unique_ptr<Port>& port : outputs_) {
    const StreamInfo& info = streams[port->stream];
    holoscan::TensorRepresentation representation{
        .memory_kind = config_.memory == ReplayMemory::kDevice ? holoscan::MemoryKind::kCudaDevice
                                                               : holoscan::MemoryKind::kHost,
        .strides = holoscan::TensorStridePolicy::kContiguousOnly,
    };
    // A stream whose samples all have one element type and rank promises them.
    if (info.dtype) representation.dtype = dl_type(*info.dtype);
    if (info.rank) representation.rank = *info.rank;
    const holoscan::TensorOutputSpec tensor_spec{
        .representation = representation,
        .bounds = holoscan::tensor_bounds(std::max<std::uint64_t>(info.max_bytes, 1U)),
        .storage = holoscan::TensorOutputStorage::kRuntimePool,
    };
    if (port->image) {
      spec.output(*port->image_port, port->name)
          .produces_tensor(tensor_spec)
          .payload_options(holoscan::PayloadOptions{
              .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
          });
    } else {
      spec.output(*port->tensor_port, port->name).produces_tensor(tensor_spec);
    }
  }
}

holoscan::Contract ReplayerOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void ReplayerOp::start() {
  position_ = 0;
  pass_ = 0;
  start_ns_.reset();
  finished_ = false;
  if (config_.stats) config_.stats->finished = false;
  if (config_.memory != ReplayMemory::kDevice) return;
  staging_ = std::make_unique<Staging>();
  for (const std::unique_ptr<Port>& port : outputs_) {
    staging_->capacity =
        std::max<std::size_t>(staging_->capacity, reader_->streams()[port->stream].max_bytes);
  }
  staging_->capacity = std::max<std::size_t>(staging_->capacity, 1U);
  for (Staging::Buffer& buffer : staging_->buffers) {
    cudaError_t status = cudaMallocHost(&buffer.data, staging_->capacity);
    if (status == cudaSuccess) {
      status = cudaEventCreateWithFlags(&buffer.event, cudaEventDisableTiming);
    }
    if (status != cudaSuccess) {
      staging_->release();
      staging_.reset();
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                   "ReplayerOp: " + cuda_text("staging memory", status)};
    }
  }
}

void ReplayerOp::stop() {
  if (staging_) {
    staging_->release();
    staging_.reset();
  }
}

const holoscan::Output<holoscan::schema::ImageT>& ReplayerOp::image(std::string_view name) const {
  for (const std::unique_ptr<Port>& port : outputs_) {
    if (port->name == name && port->image) return *port->image_port;
  }
  throw std::out_of_range("ReplayerOp: no image stream \"" + std::string(name) + "\"");
}

const holoscan::Output<holoscan::Tensor>& ReplayerOp::tensor(std::string_view name) const {
  for (const std::unique_ptr<Port>& port : outputs_) {
    if (port->name == name && !port->image) return *port->tensor_port;
  }
  throw std::out_of_range("ReplayerOp: no tensor stream \"" + std::string(name) + "\"");
}

std::vector<std::string> ReplayerOp::stream_names() const {
  std::vector<std::string> names;
  for (const std::unique_ptr<Port>& port : outputs_) names.push_back(port->name);
  return names;
}

void ReplayerOp::bind_device(holoscan::CompileOptions& options, const std::string& operator_path,
                             holoscan::DeviceId device) const {
  if (config_.memory != ReplayMemory::kDevice) return;
  for (const std::unique_ptr<Port>& port : outputs_) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = operator_path, .output_port = port->name, .device = device});
  }
}

holoscan::expected<void, holoscan::Error> ReplayerOp::send(const Record& record,
                                                           holoscan::ExecutionContext& context) {
  Port& port = *outputs_[*output_of_stream_[record.stream]];
  const std::uint64_t bytes = *record.bytes();
  const holoscan::TensorLoanRequest request{.shape = record.shape, .dtype = dl_type(record.dtype)};
  auto loan = port.image ? port.image_port->allocate_tensor(request)
                         : port.tensor_port->allocate_tensor(request);
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  const auto read = [&](void* out) -> Result {
    try {
      reader_->read(record, std::span(static_cast<std::byte*>(out), bytes));
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(failure(error.what()));
    }
    return {};
  };
  if (config_.memory == ReplayMemory::kHost) {
    if (auto written = loan->write_host_and_commit(
            [&](const holoscan::TensorOutputWriteGuard& writer) { return read(writer.data()); });
        !written) {
      return written;
    }
  } else {
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(
          failure("the runtime gave no CUDA stream for the device outputs; bind their device"));
    }
    // Read into the next pinned buffer, once its last copy is done, then copy it
    // to the GPU on the operator's stream.
    Staging::Buffer& buffer = staging_->buffers[staging_->next];
    staging_->next = (staging_->next + 1) % staging_->buffers.size();
    if (buffer.in_use) {
      if (const cudaError_t status = cudaEventSynchronize(buffer.event); status != cudaSuccess) {
        return holoscan::make_unexpected(failure(cuda_text("cudaEventSynchronize", status)));
      }
      buffer.in_use = false;
    }
    if (auto filled = read(buffer.data); !filled) {
      return filled;
    }
    auto written = loan->write_and_commit(
        stream, [&](const holoscan::TensorOutputWriteGuard& writer) -> Result {
          if (bytes == 0) return {};
          cudaError_t status =
              cudaMemcpyAsync(writer.data(), buffer.data, bytes, cudaMemcpyHostToDevice, stream);
          if (status == cudaSuccess) status = cudaEventRecord(buffer.event, stream);
          if (status != cudaSuccess) {
            return holoscan::make_unexpected(failure(cuda_text("cudaMemcpyAsync", status)));
          }
          buffer.in_use = true;
          return {};
        });
    if (!written) {
      return written;
    }
  }

  // Each pass after the first moves the capture times and frame ids on.
  const std::int64_t timestamp =
      record.timestamp_ns + static_cast<std::int64_t>(pass_) * pass_span_ns_;
  holoscan::EmitOptions options{.capture_time = context.clock().at_ns(timestamp)};
  if (record.frame_id != 0) {
    options.frame_id = record.frame_id + pass_ * pass_frames_;
  }
  if (!port.image) {
    return port.tensor_port->emit(std::move(*loan), options);
  }
  const ImageDescriptor& d = *record.image;
  holoscan::schema::ImageT descriptor;
  descriptor.width = d.width;
  descriptor.height = d.height;
  descriptor.encoding = *encoding_named(d.encoding);
  descriptor.color_space = *color_space_named(d.color_space);
  descriptor.significant_bits = d.significant_bits;
  descriptor.roi_offset_x = d.roi_offset_x;
  descriptor.roi_offset_y = d.roi_offset_y;
  for (const PlaneLayout& plane : d.plane_layouts) {
    descriptor.plane_layouts.emplace_back(plane.offset_bytes, plane.row_stride_bytes,
                                          plane.size_bytes);
  }
  descriptor.exposure_time_ns = d.exposure_time_ns;
  descriptor.gain = d.gain;
  if (d.has_header) {
    descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
    descriptor.header->frame_id = d.header_frame_id;
    descriptor.header->capture_timestamp_ns =
        d.header_capture_timestamp_ns + static_cast<std::int64_t>(pass_) * pass_span_ns_;
    descriptor.header->device_sequence = d.header_device_sequence;
  }
  return port.image_port->emit_tensor(std::move(*loan), descriptor, options);
}

holoscan::expected<void, holoscan::Error> ReplayerOp::compute(holoscan::ExecutionContext& context) {
  if (finished_) {
    return {};  // The recording has ended: idle.
  }
  const std::vector<Record>& records = reader_->records();
  const std::int64_t now = context.activation_time().timestamp_ns;
  if (!start_ns_) {
    start_ns_ = now;
  }
  const std::int64_t first_ns = records[order_.front()].timestamp_ns;
  for (const std::unique_ptr<Port>& port : outputs_) port->sent = false;
  while (true) {
    if (position_ == order_.size()) {
      if (config_.stats) ++config_.stats->passes;
      if (!config_.loop) {
        finished_ = true;
        if (config_.stats) config_.stats->finished = true;
        return {};
      }
      position_ = 0;
      ++pass_;
    }
    const Record& record = records[order_[position_]];
    // Its time in the replay, from the first sample of the first pass.
    const std::int64_t due =
        record.timestamp_ns - first_ns + static_cast<std::int64_t>(pass_) * pass_span_ns_;
    if (config_.pacing == ReplayPacing::kRealtime && due > now - *start_ns_) {
      return {};  // Not yet.
    }
    Port& port = *outputs_[*output_of_stream_[record.stream]];
    if (port.sent) {
      return {};  // One sample per output each tick; the next one waits for the next tick.
    }
    if (auto sent = send(record, context); !sent) {
      if (refused(sent.error())) {
        return {};  // A consumer is busy: send it again on the next tick.
      }
      return sent;
    }
    port.sent = true;
    ++position_;
    if (config_.stats) {
      ++config_.stats->emitted;
      if (config_.pacing == ReplayPacing::kRealtime) {
        const std::int64_t late = now - *start_ns_ - due;
        std::int64_t worst = config_.stats->max_lateness_ns.load();
        while (late > worst && !config_.stats->max_lateness_ns.compare_exchange_weak(worst, late)) {
        }
      }
    }
  }
}

}  // namespace holoscan::recorder
