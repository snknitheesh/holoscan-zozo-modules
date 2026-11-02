// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stitching/stitch_op.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/stitching/stitcher.hpp>

namespace holoscan::stitching {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "StitchOp: " + message};
}

holoscan::Error cuda_failure(const std::string& what, cudaError_t status) {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         "StitchOp: " + what + ": " + cudaGetErrorString(status)};
}

bool on_gpu(const holoscan::Tensor& tensor) {
  const DLDevice device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

std::optional<PixelFormat> format_of(holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_MONO8:
      return PixelFormat::kMono8;
    case holoscan::schema::ImageEncoding_RGB8:
      return PixelFormat::kRgb8;
    case holoscan::schema::ImageEncoding_RGBA8:
      return PixelFormat::kRgba8;
    case holoscan::schema::ImageEncoding_BGR8:
      return PixelFormat::kBgr8;
    case holoscan::schema::ImageEncoding_BGRA8:
      return PixelFormat::kBgra8;
    default:
      return std::nullopt;
  }
}

// Keeps a pinned host tensor's memory from reuse until the work queued on
// `stream` is done: its allocator waits for the stream if it can, else this.
holoscan::expected<void, holoscan::Error> hold_until_done(const holoscan::Tensor& tensor,
                                                          cudaStream_t stream) {
  holoscan::Tensor handle = tensor;
  if (handle.set_deallocation_stream(stream)) {
    return {};
  }
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
  }
  return {};
}

// Says, once, that the images are copied from host to device memory, and how
// not to.
void log_host_copy(std::size_t bytes) {
  HOLOSCAN_LOG_INFO(
      "StitchOp: images arrive in host memory and the stitcher reads CUDA device memory, so "
      "each frame set is copied host to device, {} bytes ({:.1f} MB) per frame; to avoid the "
      "copy, send CUDA device images",
      bytes, static_cast<double>(bytes) / 1e6);
}

}  // namespace

void validate_config(const StitchConfig& config) {
  validate(config.rig);
  validate(config.output);
  validate(config.blend);
  if (config.sync_window.count() < 0) {
    throw std::invalid_argument("sync_window must not be negative");
  }
}

struct StitchOp::State {
  std::unique_ptr<Stitcher> stitcher;
  // A device copy of each camera's host images, grown as needed.
  struct Staging {
    void* data{};
    std::size_t bytes{};
  };
  std::array<Staging, kMaxCameras> staging{};
  bool host_copy_logged{};
  // Used when the runtime gives this operator no CUDA stream.
  cudaStream_t own_stream{};
  std::atomic<std::size_t> table_bytes{0};

  void release() noexcept {
    if (own_stream != nullptr) {
      cudaStreamSynchronize(own_stream);
      cudaStreamDestroy(own_stream);
      own_stream = nullptr;
    }
    for (Staging& s : staging) {
      if (s.data != nullptr) cudaFree(s.data);  // Waits for the work that reads it.
      s = {};
    }
    stitcher.reset();
    table_bytes = 0;
  }
};

StitchOp::StitchOp(StitchConfig config)
    : config_(std::move(config)), state_(std::make_unique<State>()) {
  validate_config(config_);
  for (std::size_t i = 0; i < kMaxCameras; ++i) {
    port_names_[i] = "image_" + std::to_string(i);
  }
}

StitchOp::~StitchOp() { state_->release(); }

void StitchOp::setup(holoscan::OperatorSpec& spec) {
  // Only the rig's cameras have an input: an unconnected one would never be
  // satisfied. Two slots each: one frame in compute() while the next arrives.
  for (std::size_t i = 0; i < config_.rig.cameras.size(); ++i) {
    spec.input(images[i], port_names_[i]).queue_depth(2U);
  }
  const int out_channels = channels(config_.blend.output_format);
  const auto bytes = static_cast<std::uint64_t>(config_.output.width) *
                     static_cast<std::uint64_t>(config_.output.height) *
                     static_cast<std::uint64_t>(out_channels);
  spec.output(stitched, "stitched")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kUInt8,
                  .rank = 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(bytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract StitchOp::contract() const {
  holoscan::Contract result;
  const std::size_t n = config_.rig.cameras.size();
  if (n == 1) {
    result.trigger(holoscan::OnEach{images[0]});
    return result;
  }
  std::vector<holoscan::PortRef> ports;
  for (std::size_t i = 0; i < n; ++i) {
    ports.emplace_back(images[i]);
  }
  result.trigger(
      holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = config_.sync_window});
  for (std::size_t i = 0; i < n; ++i) {
    result.input(images[i], holoscan::Exact{});
  }
  return result;
}

void StitchOp::start() {
  state_->release();
  state_->stitcher = std::make_unique<Stitcher>(config_.rig, config_.output, config_.blend);
  state_->table_bytes = state_->stitcher->table_bytes();
  const std::vector<std::uint64_t>& covered = state_->stitcher->covered_pixels();
  const auto total = static_cast<std::uint64_t>(config_.output.width) *
                     static_cast<std::uint64_t>(config_.output.height);
  for (std::size_t c = 0; c < covered.size(); ++c) {
    const Camera& camera = config_.rig.cameras[c];
    if (covered[c] == 0) {
      HOLOSCAN_LOG_WARN(
          "StitchOp: camera {} ({}) sees none of the {} output; its images are "
          "received and ignored",
          c, camera.name, projection_name(config_.output.projection));
    } else {
      HOLOSCAN_LOG_DEBUG("StitchOp: camera {} ({}) sees {:.1f}% of the output", c, camera.name,
                         100.0 * static_cast<double>(covered[c]) / static_cast<double>(total));
    }
  }
  HOLOSCAN_LOG_INFO("StitchOp: {} {}x{} from {} cameras; remap tables {:.1f} MB on the GPU",
                    projection_name(config_.output.projection), config_.output.width,
                    config_.output.height, config_.rig.cameras.size(),
                    static_cast<double>(state_->table_bytes.load()) / 1e6);
}

void StitchOp::stop() { state_->release(); }

std::size_t StitchOp::table_bytes() const noexcept { return state_->table_bytes.load(); }

holoscan::expected<void, holoscan::Error> StitchOp::compute(holoscan::ExecutionContext& context) {
  State& s = *state_;
  if (!s.stitcher) {
    return holoscan::make_unexpected(invalid("compute() before start()"));
  }
  const std::size_t n = config_.rig.cameras.size();
  std::vector<holoscan::Sample<holoscan::schema::ImageT>> samples;
  samples.reserve(n);
  holoscan::sensor_io::ProvenanceAccumulator provenance;
  for (std::size_t i = 0; i < n; ++i) {
    auto sample = images[i].receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    provenance.observe(*sample);
    samples.push_back(std::move(*sample));
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    // Not reached in a compiled graph, whose device output gets a stream.
    // Kept so the operator never uses the legacy default stream.
    if (s.own_stream == nullptr) {
      if (const cudaError_t status =
              cudaStreamCreateWithFlags(&s.own_stream, cudaStreamNonBlocking);
          status != cudaSuccess) {
        s.own_stream = nullptr;
        return holoscan::make_unexpected(cuda_failure("cudaStreamCreate", status));
      }
    }
    stream = s.own_stream;
  }

  std::array<DeviceImage, kMaxCameras> views{};
  std::size_t copied_bytes = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Camera& camera = config_.rig.cameras[i];
    const holoscan::schema::ImageT& image = samples[i].data;
    const std::string which = "image_" + std::to_string(i) + " (" + camera.name + ")";
    if (!image.data) {
      return holoscan::make_unexpected(invalid(which + " has no tensor"));
    }
    const std::optional<PixelFormat> format = format_of(image.encoding);
    if (!format) {
      return holoscan::make_unexpected(
          invalid(which + ": unsupported encoding " +
                  holoscan::schema::EnumNameImageEncoding(image.encoding) +
                  "; use MONO8, RGB8, BGR8, RGBA8 or BGRA8"));
    }
    const int ch = channels(*format);
    const holoscan::Tensor& tensor = *image.data;
    const DLDataType dtype = tensor.dtype();
    const std::span<const std::int64_t> shape = tensor.shape_span();
    const bool shape_ok = (shape.size() == 3 && shape[2] == ch) || (ch == 1 && shape.size() == 2);
    if (dtype.code != kDLUInt || dtype.bits != 8 || dtype.lanes != 1 || !shape_ok) {
      return holoscan::make_unexpected(
          invalid(which + ": the tensor must be uint8 [H, W, " + std::to_string(ch) + "]" +
                  (ch == 1 ? " or [H, W]" : "") + " for " + std::string(format_name(*format))));
    }
    if (shape[0] != camera.height || shape[1] != camera.width) {
      return holoscan::make_unexpected(
          invalid(which + " is " + std::to_string(shape[1]) + "x" + std::to_string(shape[0]) +
                  "; the camera's calibration is " + std::to_string(camera.width) + "x" +
                  std::to_string(camera.height)));
    }
    const std::vector<std::int64_t> strides = tensor.strides();
    const auto row_bytes = static_cast<std::int64_t>(camera.width) * ch;
    if ((shape.size() == 3 && strides[2] != 1) || strides[1] != ch || strides[0] < row_bytes) {
      return holoscan::make_unexpected(
          invalid(which + ": pixels must be packed within a row (rows may be padded)"));
    }
    const auto pitch = static_cast<std::size_t>(strides[0]);
    if (on_gpu(tensor)) {
      views[i] = DeviceImage{tensor.data(), camera.width, camera.height, pitch, *format};
      continue;
    }
    // Host memory: the runtime does not order plain host memory after its
    // producer, so wait for it, then copy it to the GPU.
    if (tensor.device().device_type == kDLCPU) {
      if (auto ready = tensor.synchronize_for_host(); !ready) {
        return holoscan::make_unexpected(std::move(ready).error());
      }
    }
    const std::size_t packed = static_cast<std::size_t>(row_bytes) * camera.height;
    State::Staging& staging = s.staging[i];
    if (staging.bytes < packed) {
      // Free only after the stream is done with the old buffer.
      cudaStreamSynchronize(stream);
      cudaFree(staging.data);
      staging = {};
      if (const cudaError_t status = cudaMalloc(&staging.data, packed); status != cudaSuccess) {
        staging.data = nullptr;
        return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
      }
      staging.bytes = packed;
    }
    if (const cudaError_t status = cudaMemcpy2DAsync(
            staging.data, static_cast<std::size_t>(row_bytes), tensor.data(), pitch,
            static_cast<std::size_t>(row_bytes), static_cast<std::size_t>(camera.height),
            cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMemcpy2DAsync", status));
    }
    // A pinned buffer could be reused by its producer while the copy still
    // reads it; pageable memory is staged by the time the call returns.
    if (tensor.device().device_type == kDLCUDAHost) {
      if (stream == s.own_stream) {
        if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
          return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
        }
      } else if (auto held = hold_until_done(tensor, stream); !held) {
        return holoscan::make_unexpected(std::move(held).error());
      }
    }
    copied_bytes += packed;
    views[i] = DeviceImage{staging.data, camera.width, camera.height,
                           static_cast<std::size_t>(row_bytes), *format};
  }
  if (copied_bytes != 0 && !s.host_copy_logged) {
    s.host_copy_logged = true;
    log_host_copy(copied_bytes);
  }

  const int out_channels = channels(config_.blend.output_format);
  const std::array<std::int64_t, 3> out_shape{config_.output.height, config_.output.width,
                                              out_channels};
  auto loan =
      stitched.allocate_tensor(holoscan::TensorLoanRequest{.shape = out_shape, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  cudaError_t status = cudaSuccess;
  try {
    status = s.stitcher->stitch(
        std::span<const DeviceImage>(views.data(), n), writer->data(),
        static_cast<std::size_t>(config_.output.width) * static_cast<std::size_t>(out_channels),
        stream);
  } catch (const std::invalid_argument& error) {
    return holoscan::make_unexpected(invalid(error.what()));
  }
  if (status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("stitch kernels", status));
  }
  if (stream == s.own_stream) {
    // The runtime knows nothing of this stream; finish before the output goes.
    if (const cudaError_t synced = cudaStreamSynchronize(stream); synced != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", synced));
    }
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }

  auto emit = provenance.emit_options(holoscan::sensor_io::CaptureTimePolicy::kEarliest);
  if (!emit) {
    return holoscan::make_unexpected(
        invalid("the frame set's capture times are in different clock domains"));
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = config_.output.width;
  descriptor.height = config_.output.height;
  descriptor.encoding = out_channels == 4 ? holoscan::schema::ImageEncoding_RGBA8
                                          : holoscan::schema::ImageEncoding_RGB8;
  descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
  if (emit->capture_time) {
    descriptor.header->capture_timestamp_ns = emit->capture_time->timestamp_ns;
  }
  descriptor.header->frame_id = config_.frame_id;
  return stitched.emit_tensor(std::move(*loan), descriptor, *emit);
}

}  // namespace holoscan::stitching
