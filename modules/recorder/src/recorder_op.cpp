// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/recorder/recorder_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

#include "json.hpp"

namespace holoscan::recorder {
namespace {

using Result = holoscan::expected<void, holoscan::Error>;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kMaxQueueSamples = 4096;
constexpr std::chrono::seconds kWarningInterval{1};

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "RecorderOp: " + message};
}

std::string cuda_text(const char* what, cudaError_t status) {
  return std::string(what) + ": " + cudaGetErrorString(status);
}

bool on_gpu(const holoscan::Tensor& tensor) {
  const DLDeviceType type = tensor.device().device_type;
  return type == kDLCUDA || type == kDLCUDAManaged;
}

// The image's descriptor, as the format keeps it.
holoscan::expected<ImageDescriptor, holoscan::Error> describe(
    const holoscan::schema::ImageT& image) {
  ImageDescriptor d;
  d.width = image.width;
  d.height = image.height;
  d.encoding = holoscan::schema::EnumNameImageEncoding(image.encoding);
  d.color_space = holoscan::schema::EnumNameColorSpace(image.color_space);
  if (d.encoding.empty() || d.color_space.empty()) {
    return holoscan::make_unexpected(
        failure("an image with encoding " + std::to_string(static_cast<int>(image.encoding)) +
                " and color space " + std::to_string(static_cast<int>(image.color_space)) +
                ", which this SDK does not name"));
  }
  d.significant_bits = image.significant_bits;
  d.roi_offset_x = image.roi_offset_x;
  d.roi_offset_y = image.roi_offset_y;
  for (const holoscan::schema::ImagePlaneLayout& plane : image.plane_layouts) {
    d.plane_layouts.push_back(PlaneLayout{.offset_bytes = plane.offset_bytes(),
                                          .row_stride_bytes = plane.row_stride_bytes(),
                                          .size_bytes = plane.size_bytes()});
  }
  d.exposure_time_ns = image.exposure_time_ns;
  d.gain = image.gain;
  if (image.header) {
    d.has_header = true;
    d.header_frame_id = image.header->frame_id;
    d.header_capture_timestamp_ns = image.header->capture_timestamp_ns;
    d.header_device_sequence = image.header->device_sequence;
  }
  return d;
}

// A layout the recorder copies: packed, or packed rows `pitch` bytes apart.
struct Layout {
  bool packed{};
  std::size_t rows{};
  std::size_t row_bytes{};
  std::size_t pitch{};
};

std::optional<Layout> layout_of(const holoscan::Tensor& tensor, std::uint64_t bytes) {
  if (tensor.is_contiguous()) {
    return Layout{.packed = true, .rows = 1, .row_bytes = bytes, .pitch = bytes};
  }
  // Rows of packed elements: every axis but the first is packed, and the first
  // steps over at least a row.
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const std::vector<std::int64_t> strides = tensor.strides();
  if (shape.size() < 2 || strides.size() != shape.size()) {
    return std::nullopt;
  }
  std::int64_t expected = static_cast<std::int64_t>(tensor.itemsize());
  for (std::size_t axis = shape.size(); axis-- > 1;) {
    if (shape[axis] > 1 && strides[axis] != expected) {
      return std::nullopt;
    }
    expected *= shape[axis];
  }
  if (strides[0] < expected) {
    return std::nullopt;
  }
  return Layout{.packed = false,
                .rows = static_cast<std::size_t>(shape[0]),
                .row_bytes = static_cast<std::size_t>(expected),
                .pitch = static_cast<std::size_t>(strides[0])};
}

}  // namespace

// The writer: a bounded queue of samples in pinned host memory, and the thread
// that writes them to the recording.
class RecorderOp::Writer {
 public:
  struct Slot {
    void* data{};
    std::size_t capacity{};
    cudaEvent_t event{};
    // An asynchronous device copy fills it; the writer waits for `event`.
    bool copying{};
  };
  struct Item {
    std::uint32_t slot{};
    Record record;
    std::size_t bytes{};
  };

  explicit Writer(const RecorderConfig& config) : config_(config) {
    std::vector<StreamDeclaration> streams;
    for (const RecorderStream& stream : config.streams) {
      streams.push_back(StreamDeclaration{.name = stream.name, .kind = stream.kind});
    }
    recording_.emplace(config.directory, std::move(streams),
                       WriterOptions{.overwrite = config.overwrite,
                                     .calibration = config.calibration,
                                     .metadata = config.metadata});
    // Slots never move: the writer thread reads them while compute() adds more.
    slots_.reserve(config.queue_samples);
    thread_ = std::thread([this] { run(); });
  }

  ~Writer() { close({}); }
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;

  // A free slot of at least `bytes`, or none when the queue is full.
  holoscan::expected<std::optional<std::uint32_t>, holoscan::Error> acquire(std::size_t bytes) {
    const std::lock_guard lock(mutex_);
    if (bytes > config_.queue_bytes) {
      return std::optional<std::uint32_t>{};
    }
    // The smallest free slot that is large enough.
    std::optional<std::size_t> best;
    for (std::size_t i = 0; i < free_.size(); ++i) {
      const std::size_t capacity = slots_[free_[i]].capacity;
      if (capacity >= bytes && (!best || capacity < slots_[free_[*best]].capacity)) {
        best = i;
      }
    }
    if (best) {
      const std::uint32_t slot = free_[*best];
      free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(*best));
      return std::optional<std::uint32_t>{slot};
    }
    // A new slot, while the queue has room for one.
    if (slots_.size() < config_.queue_samples && pinned_ + bytes <= config_.queue_bytes) {
      Slot slot;
      if (const cudaError_t status = cudaMallocHost(&slot.data, bytes); status != cudaSuccess) {
        return holoscan::make_unexpected(failure(cuda_text("cudaMallocHost", status)));
      }
      if (const cudaError_t status = cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming);
          status != cudaSuccess) {
        cudaFreeHost(slot.data);
        return holoscan::make_unexpected(failure(cuda_text("cudaEventCreate", status)));
      }
      slot.capacity = bytes;
      pinned_ += bytes;
      slots_.push_back(slot);
      return std::optional<std::uint32_t>{static_cast<std::uint32_t>(slots_.size() - 1)};
    }
    // A free slot that is too small, grown. Its old memory is freed at close():
    // freeing pinned memory synchronizes the device.
    if (!free_.empty()) {
      const std::uint32_t index = free_.back();
      Slot& slot = slots_[index];
      if (pinned_ - slot.capacity + bytes <= config_.queue_bytes) {
        void* data = nullptr;
        if (const cudaError_t status = cudaMallocHost(&data, bytes); status != cudaSuccess) {
          return holoscan::make_unexpected(failure(cuda_text("cudaMallocHost", status)));
        }
        retired_.push_back(slot.data);
        pinned_ = pinned_ - slot.capacity + bytes;
        slot.data = data;
        slot.capacity = bytes;
        free_.pop_back();
        return std::optional<std::uint32_t>{index};
      }
    }
    return std::optional<std::uint32_t>{};
  }

  // Slots stay put once made, so the caller fills one without the lock.
  Slot& slot(std::uint32_t index) {
    const std::lock_guard lock(mutex_);
    return slots_[index];
  }

  // Returns a slot that was not submitted.
  void give_back(std::uint32_t index) {
    const std::lock_guard lock(mutex_);
    free_.push_back(index);
  }

  void submit(Item item, RecorderStats* stats) {
    {
      const std::lock_guard lock(mutex_);
      queue_.push_back(std::move(item));
      if (stats != nullptr) {
        const auto queued = static_cast<std::uint32_t>(queue_.size());
        std::uint32_t seen = stats->max_queued.load();
        while (queued > seen && !stats->max_queued.compare_exchange_weak(seen, queued)) {
        }
      }
    }
    wake_.notify_one();
  }

  // The first write error, once there has been one.
  [[nodiscard]] std::optional<std::string> error() const {
    const std::lock_guard lock(mutex_);
    return error_;
  }

  [[nodiscard]] std::uint64_t pinned_bytes() const {
    const std::lock_guard lock(mutex_);
    return pinned_;
  }

  // Writes what is queued, finishes the recording with `dropped`, and frees
  // everything. Returns what failed, if anything did.
  std::optional<std::string> close(std::span<const std::uint64_t> dropped) {
    {
      const std::lock_guard lock(mutex_);
      if (closed_) return error_;
      stopping_ = true;
    }
    wake_.notify_one();
    if (thread_.joinable()) thread_.join();
    if (recording_ && !error_) {
      try {
        recording_->finish(dropped);
      } catch (const std::exception& error) {
        error_ = error.what();
      }
    }
    recording_.reset();
    for (Slot& slot : slots_) {
      if (slot.copying) cudaEventSynchronize(slot.event);
      cudaEventDestroy(slot.event);
      cudaFreeHost(slot.data);
    }
    for (void* data : retired_) cudaFreeHost(data);
    slots_.clear();
    retired_.clear();
    free_.clear();
    closed_ = true;
    return error_;
  }

 private:
  void run() {
    // With a rate limit: when the writer may start on the next sample.
    Clock::time_point next_write = Clock::now();
    while (true) {
      Item item;
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [this] { return !queue_.empty() || stopping_; });
        if (queue_.empty()) return;
        item = std::move(queue_.front());
        queue_.pop_front();
      }
      Slot* slot = nullptr;
      {
        const std::lock_guard lock(mutex_);
        slot = &slots_[item.slot];
      }
      std::optional<std::string> failed;
      if (slot->copying) {
        if (const cudaError_t status = cudaEventSynchronize(slot->event); status != cudaSuccess) {
          failed = cuda_text("copying a sample to the host", status);
        }
        slot->copying = false;
      }
      bool ok = false;
      {
        const std::lock_guard lock(mutex_);
        ok = !error_.has_value();
      }
      const Clock::time_point write_started = Clock::now();
      if (ok && !failed) {
        try {
          recording_->append(std::move(item.record),
                             std::span(static_cast<const std::byte*>(slot->data), item.bytes));
          if (config_.stats) {
            ++config_.stats->recorded;
            config_.stats->bytes += item.bytes;
          }
        } catch (const std::exception& error) {
          failed = error.what();
        }
      }
      if (ok && !failed && config_.max_write_bytes_per_second != 0) {
        // Hold to the rate over any stretch of time: this sample's bytes take
        // their share of a second, from when it could start.
        next_write = std::max(next_write, write_started) +
                     std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(
                         static_cast<double>(item.bytes) /
                         static_cast<double>(config_.max_write_bytes_per_second)));
        std::this_thread::sleep_until(next_write);
      }
      const std::lock_guard lock(mutex_);
      if (failed && !error_) error_ = std::move(failed);
      free_.push_back(item.slot);
    }
  }

  const RecorderConfig& config_;
  std::optional<RecordingWriter> recording_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::vector<Slot> slots_;
  std::vector<std::uint32_t> free_;
  std::vector<void*> retired_;
  std::uint64_t pinned_{};
  std::deque<Item> queue_;
  bool stopping_{};
  bool closed_{};
  std::optional<std::string> error_;
  std::thread thread_;
};

void validate_config(const RecorderConfig& config) {
  if (config.directory.empty()) {
    throw std::invalid_argument("directory must not be empty");
  }
  if (config.streams.empty()) {
    throw std::invalid_argument("streams must name at least one stream");
  }
  for (std::size_t i = 0; i < config.streams.size(); ++i) {
    const std::string& name = config.streams[i].name;
    if (!valid_stream_name(name)) {
      throw std::invalid_argument("stream name \"" + name +
                                  "\" must be 1 to 64 letters, digits, '_', '-' and '.', not "
                                  "starting with '.'");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (config.streams[j].name == name) {
        throw std::invalid_argument("two streams are named \"" + name + "\"");
      }
    }
  }
  for (const auto& [name, blob] : config.calibration) {
    if (name.empty()) {
      throw std::invalid_argument("a calibration blob needs a name");
    }
    try {
      (void)json::parse(blob);
    } catch (const std::invalid_argument& error) {
      throw std::invalid_argument("calibration \"" + name + "\" is not JSON: " + error.what());
    }
  }
  if (config.queue_samples < 1 || config.queue_samples > kMaxQueueSamples) {
    throw std::invalid_argument("queue_samples must be 1.." + std::to_string(kMaxQueueSamples));
  }
  if (config.queue_bytes < 1) {
    throw std::invalid_argument("queue_bytes must be positive");
  }
  if (config.input_queue_depth < 1) {
    throw std::invalid_argument("input_queue_depth must be positive");
  }
}

RecorderOp::RecorderOp(RecorderConfig config) : config_(std::move(config)) {
  validate_config(config_);
  for (const RecorderStream& stream : config_.streams) {
    if (stream.kind == StreamKind::kImage) {
      images_.push_back(std::make_unique<holoscan::Input<holoscan::schema::ImageT>>());
      tensors_.emplace_back();
    } else {
      images_.emplace_back();
      tensors_.push_back(std::make_unique<holoscan::Input<holoscan::Tensor>>());
    }
  }
}

RecorderOp::~RecorderOp() {
  if (writer_) {
    writer_->close(dropped_);
  }
}

void RecorderOp::setup(holoscan::OperatorSpec& spec) {
  for (std::size_t i = 0; i < config_.streams.size(); ++i) {
    const std::string& name = config_.streams[i].name;
    if (images_[i]) {
      spec.input(*images_[i], name).queue_depth(config_.input_queue_depth);
    } else {
      spec.input(*tensors_[i], name).queue_depth(config_.input_queue_depth);
    }
  }
}

holoscan::Contract RecorderOp::contract() const {
  // Whatever arrives, on any input: every queued sample of every input.
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

void RecorderOp::start() {
  sequence_.assign(config_.streams.size(), 0);
  dropped_.assign(config_.streams.size(), 0);
  unreported_drops_ = 0;
  warned_ = false;
  try {
    writer_ = std::make_unique<Writer>(config_);
  } catch (const std::exception& error) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                 std::string("RecorderOp: ") + error.what()};
  }
  HOLOSCAN_LOG_INFO("RecorderOp: recording {} stream(s) to {}", config_.streams.size(),
                    config_.directory);
}

void RecorderOp::stop() {
  if (!writer_) return;
  const std::optional<std::string> error = writer_->close(dropped_);
  writer_.reset();
  if (error) {
    HOLOSCAN_LOG_ERROR("RecorderOp: {}: the recording is incomplete: {}", config_.directory,
                       *error);
  }
  std::uint64_t dropped = 0;
  for (const std::uint64_t count : dropped_) dropped += count;
  HOLOSCAN_LOG_INFO("RecorderOp: finished {}: dropped {} sample(s)", config_.directory, dropped);
}

const holoscan::Input<holoscan::schema::ImageT>& RecorderOp::image(std::string_view name) const {
  for (std::size_t i = 0; i < config_.streams.size(); ++i) {
    if (config_.streams[i].name == name && images_[i]) return *images_[i];
  }
  throw std::out_of_range("RecorderOp: no image stream \"" + std::string(name) + "\"");
}

const holoscan::Input<holoscan::Tensor>& RecorderOp::tensor(std::string_view name) const {
  for (std::size_t i = 0; i < config_.streams.size(); ++i) {
    if (config_.streams[i].name == name && tensors_[i]) return *tensors_[i];
  }
  throw std::out_of_range("RecorderOp: no tensor stream \"" + std::string(name) + "\"");
}

void RecorderOp::count_drop(std::uint32_t stream) {
  ++dropped_[stream];
  ++unreported_drops_;
  if (config_.stats) ++config_.stats->dropped;
  const std::int64_t now =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
  if (warned_ && now - last_warning_ns_ < std::chrono::nanoseconds(kWarningInterval).count()) {
    return;
  }
  std::uint64_t total = 0;
  for (const std::uint64_t count : dropped_) total += count;
  HOLOSCAN_LOG_WARN(
      "RecorderOp: the disk is falling behind: dropped {} sample(s) since the last warning, {} in "
      "all; the queue holds at most {} samples in {} MiB",
      unreported_drops_, total, config_.queue_samples, config_.queue_bytes >> 20U);
  unreported_drops_ = 0;
  last_warning_ns_ = now;
  warned_ = true;
}

holoscan::expected<void, holoscan::Error> RecorderOp::enqueue(
    std::uint32_t stream, const holoscan::Tensor* tensor, const holoscan::schema::ImageT* image,
    const holoscan::SampleMetadata& metadata, holoscan::ExecutionContext& context) {
  const std::uint64_t sequence = sequence_[stream]++;
  if (config_.stats) ++config_.stats->received;
  const std::string& name = config_.streams[stream].name;
  if (tensor == nullptr) {
    return holoscan::make_unexpected(failure("stream \"" + name + "\": an image without a tensor"));
  }

  Record record;
  record.stream = stream;
  record.sequence = sequence;
  // The capture time; a sample without one gets the time it was recorded.
  record.timestamp_ns = metadata.capture_time().valid() ? metadata.capture_timestamp_ns
                                                        : context.activation_time().timestamp_ns;
  record.frame_id = metadata.frame_id;
  const DLDataType dtype = tensor->dtype();
  record.dtype = ElementType{.code = dtype.code, .bits = dtype.bits, .lanes = dtype.lanes};
  const std::span<const std::int64_t> shape = tensor->shape_span();
  record.shape.assign(shape.begin(), shape.end());
  if (image != nullptr) {
    auto descriptor = describe(*image);
    if (!descriptor) {
      return holoscan::make_unexpected(std::move(descriptor).error());
    }
    record.image = std::move(*descriptor);
  }
  const std::optional<std::uint64_t> bytes = record.bytes();
  if (!bytes || record.dtype.bits == 0 || record.dtype.lanes == 0) {
    return holoscan::make_unexpected(failure("stream \"" + name + "\": a tensor of " +
                                             to_string(record.dtype) +
                                             " elements whose size cannot be represented"));
  }
  if (*bytes != 0 && tensor->data() == nullptr) {
    return holoscan::make_unexpected(failure("stream \"" + name + "\": a tensor without data"));
  }
  const std::optional<Layout> layout = layout_of(*tensor, *bytes);
  if (!layout) {
    return holoscan::make_unexpected(failure(
        "stream \"" + name + "\": a tensor that is neither packed nor rows of packed elements"));
  }

  auto acquired = writer_->acquire(std::max<std::size_t>(*bytes, 1U));
  if (!acquired) {
    return holoscan::make_unexpected(std::move(acquired).error());
  }
  if (!*acquired) {
    count_drop(stream);
    return {};
  }
  const std::uint32_t index = **acquired;
  Writer::Slot& slot = writer_->slot(index);
  const bool device = on_gpu(*tensor);
  cudaStream_t cuda_stream = nullptr;
  if (device) {
    cuda_stream = context.cuda_stream().get();
    if (cuda_stream == nullptr) {
      writer_->give_back(index);
      return holoscan::make_unexpected(
          failure("stream \"" + name + "\": device data, but the runtime gave no CUDA stream"));
    }
  }
  Result copied;
  if (*bytes == 0) {
    copied = {};
  } else if (layout->packed) {
    // Device data asynchronously on the operator's stream, which the runtime
    // orders after the producer and before the input's release; host data now.
    copied = tensor->copy_to_host_async(slot.data, *bytes, cuda_stream);
  } else if (device) {
    if (const cudaError_t status =
            cudaMemcpy2DAsync(slot.data, layout->row_bytes, tensor->data(), layout->pitch,
                              layout->row_bytes, layout->rows, cudaMemcpyDeviceToHost, cuda_stream);
        status != cudaSuccess) {
      copied = holoscan::make_unexpected(failure(cuda_text("cudaMemcpy2DAsync", status)));
    }
  } else {
    copied = tensor->synchronize_for_host();
    if (copied) {
      const auto* source = static_cast<const std::byte*>(tensor->data());
      auto* target = static_cast<std::byte*>(slot.data);
      for (std::size_t row = 0; row < layout->rows; ++row) {
        std::memcpy(target + row * layout->row_bytes, source + row * layout->pitch,
                    layout->row_bytes);
      }
    }
  }
  if (copied && device && *bytes != 0) {
    if (const cudaError_t status = cudaEventRecord(slot.event, cuda_stream);
        status != cudaSuccess) {
      copied = holoscan::make_unexpected(failure(cuda_text("cudaEventRecord", status)));
    } else {
      slot.copying = true;
    }
  }
  if (!copied) {
    if (slot.copying) {
      cudaEventSynchronize(slot.event);
      slot.copying = false;
    }
    writer_->give_back(index);
    return holoscan::make_unexpected(std::move(copied).error());
  }
  writer_->submit(Writer::Item{.slot = index, .record = std::move(record), .bytes = *bytes},
                  config_.stats.get());
  return {};
}

holoscan::expected<void, holoscan::Error> RecorderOp::compute(holoscan::ExecutionContext& context) {
  if (!writer_) {
    return holoscan::make_unexpected(failure("compute() before start()"));
  }
  if (const std::optional<std::string> error = writer_->error()) {
    return holoscan::make_unexpected(failure("writing " + config_.directory + ": " + *error));
  }
  // Nothing selected on an input is not an error: another input woke the operator.
  const auto nothing = [](const holoscan::Error& error) {
    return error.code == holoscan::ErrorCode::kInputEmpty ||
           error.code == holoscan::ErrorCode::kInputNotSelected;
  };
  for (std::uint32_t i = 0; i < config_.streams.size(); ++i) {
    Result result;
    if (images_[i]) {
      result = images_[i]->for_each_selected(
          [&](const holoscan::Sample<holoscan::schema::ImageT>& sample) -> Result {
            return enqueue(i, sample.data.data.get(), &sample.data, sample.metadata, context);
          });
    } else {
      result = tensors_[i]->for_each_selected(
          [&](const holoscan::Sample<holoscan::Tensor>& sample) -> Result {
            return enqueue(i, &sample.data, nullptr, sample.metadata, context);
          });
    }
    if (!result && !nothing(result.error())) {
      return result;
    }
  }
  return {};
}

}  // namespace holoscan::recorder
