// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/tsdf/volume_op.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/logger/logger.hpp>

#include "device_buffer.hpp"
#include "host_input.hpp"

namespace holoscan::tsdf::detail {
namespace {

bool on_gpu(const holoscan::Tensor& tensor) {
  const DLDevice device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

// "GPU 1", "host memory", ... for errors.
std::string where(const DLDevice& device) {
  switch (device.device_type) {
    case kDLCUDA:
      return "GPU " + std::to_string(device.device_id);
    case kDLCUDAManaged:
      return "managed memory of GPU " + std::to_string(device.device_id);
    case kDLCPU:
      return "host memory";
    case kDLCUDAHost:
      return "pinned host memory";
    default:
      return "memory of DLPack device type " + std::to_string(static_cast<int>(device.device_type));
  }
}

}  // namespace

bool is_dtype(const DLDataType& dtype, const DLDataType& expected) {
  return dtype.code == expected.code && dtype.bits == expected.bits &&
         dtype.lanes == expected.lanes;
}

bool packed(const holoscan::Tensor& tensor) {
  const std::span<const std::int64_t> strides = tensor.dlpack_strides_span();
  if (strides.empty()) {
    return true;  // DLPack's compact row-major layout.
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  std::int64_t expected = 1;  // In elements, as DLPack counts strides.
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    if (shape[axis] > 1 && strides[axis] != expected) {
      return false;
    }
    expected *= shape[axis];
  }
  return true;
}

struct VolumeOp::Impl {
  std::array<DeviceBuffer, 2> staging;
  // Used when the runtime gives this operator no CUDA stream, which it does
  // when every input is in host memory and every output too.
  cudaStream_t own_stream{};
  // Whether this activation's work on own_stream reads an input in place.
  bool settle{};
  bool logged_host_copy{};
  bool logged_pose_copy{};

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  ~Impl() {
    if (own_stream != nullptr) {
      cudaStreamSynchronize(own_stream);
      cudaStreamDestroy(own_stream);
    }
  }
};

VolumeOp::VolumeOp(std::string kind, std::shared_ptr<TsdfVolume> volume)
    : kind_(std::move(kind)),
      name_(kind_),
      volume_(std::move(volume)),
      impl_(std::make_unique<Impl>()) {
  if (!volume_) {
    throw std::invalid_argument(kind_ + " needs a volume");
  }
}

VolumeOp::~VolumeOp() = default;

void VolumeOp::stop() {
  // Frees the staging buffers; cudaFree waits for the work that may still use
  // them.
  impl_ = std::make_unique<Impl>();
}

void VolumeOp::bind(const holoscan::ExecutionContext& context) {
  const std::string_view path = context.operator_path();
  if (path != path_) {
    path_ = std::string(path);
    name_ = kind_ + " '" + path_ + "'";
  }
}

holoscan::Error VolumeOp::invalid(const std::string& message) const {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, name_ + ": " + message};
}

holoscan::Error VolumeOp::cuda_failure(const char* what, cudaError_t status) const {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         name_ + ": " + what + ": " + cudaGetErrorString(status)};
}

holoscan::unexpected<holoscan::Error> VolumeOp::emit_failure(const char* port,
                                                             holoscan::Error error) const {
  // Keeps the code, so that a caller can still tell backpressure from failure.
  const char* hint = error.code == holoscan::ErrorCode::kBackpressured
                         ? " (its queue is full: the consumer is behind)"
                         : " (is the port connected?)";
  error.message = name_ + ": emitting " + port + " failed" +
                  (error.message.empty() ? std::string(hint) : ": " + error.message);
  return holoscan::make_unexpected(std::move(error));
}

holoscan::TensorOutputSpec VolumeOp::device_output(DLDataType dtype, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = dtype,
          },
      .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = bytes},
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

holoscan::TensorOutputSpec VolumeOp::host_output(DLDataType dtype, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = dtype,
          },
      .bounds = holoscan::TensorBounds{.max_rank = 2U, .max_byte_span = bytes},
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

holoscan::expected<cudaStream_t, holoscan::Error> VolumeOp::stream_for(
    holoscan::ExecutionContext& context) {
  impl_->settle = false;
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    // Not the legacy default stream, which would serialize with every other
    // stream in the process.
    if (impl_->own_stream == nullptr) {
      if (const cudaError_t status =
              cudaStreamCreateWithFlags(&impl_->own_stream, cudaStreamNonBlocking);
          status != cudaSuccess) {
        impl_->own_stream = nullptr;
        return holoscan::make_unexpected(cuda_failure("cudaStreamCreate", status));
      }
    }
    stream = impl_->own_stream;
  }
  int device = 0;
  if (const cudaError_t status = cudaStreamGetDevice(stream, &device); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamGetDevice", status));
  }
  if (device != volume_->device()) {
    return holoscan::make_unexpected(invalid("the runtime's stream is on GPU " +
                                             std::to_string(device) + ", the volume on GPU " +
                                             std::to_string(volume_->device())));
  }
  return stream;
}

holoscan::expected<Matrix4, holoscan::Error> VolumeOp::read_pose(const holoscan::Tensor& tensor,
                                                                 const char* name,
                                                                 cudaStream_t stream) {
  const DLDataType dtype = tensor.dtype();
  const bool f32 = is_dtype(dtype, kFloat32);
  const bool f64 = is_dtype(dtype, kFloat64);
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if ((!f32 && !f64) || shape.size() != 2 || shape[0] != 4 || shape[1] != 4 || !packed(tensor)) {
    return holoscan::make_unexpected(
        invalid(std::string(name) + " must be a packed float32 or float64 [4, 4]"));
  }
  std::array<double, 16> f64_values{};
  std::array<float, 16> f32_values{};
  void* destination = f32 ? static_cast<void*>(f32_values.data()) : f64_values.data();
  const std::size_t bytes = f32 ? sizeof(f32_values) : sizeof(f64_values);
  if (on_gpu(tensor)) {
    if (!impl_->logged_pose_copy) {
      impl_->logged_pose_copy = true;
      HOLOSCAN_LOG_INFO(
          "{}: {} is in device memory; copying it to the host, which waits for "
          "the GPU each frame (send it in host memory to avoid that)",
          name_, name);
    }
    if (auto copied = tensor.copy_to_host(destination, bytes, stream); !copied) {
      return holoscan::make_unexpected(std::move(copied).error());
    }
  } else {
    if (auto ready = tensor.synchronize_for_host(); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    std::memcpy(destination, tensor.data(), bytes);
  }
  Matrix4 pose{};
  for (std::size_t i = 0; i < pose.size(); ++i) {
    pose[i] = f32 ? f32_values[i] : static_cast<float>(f64_values[i]);
  }
  if (!valid_pose(pose)) {
    return holoscan::make_unexpected(
        invalid(std::string(name) + " is not a rigid transform (rotation, translation, 0 0 0 1)"));
  }
  return pose;
}

holoscan::expected<const void*, holoscan::Error> VolumeOp::on_device(const holoscan::Tensor& tensor,
                                                                     std::size_t bytes,
                                                                     std::size_t capacity, int slot,
                                                                     cudaStream_t stream) {
  const bool own = stream == impl_->own_stream;
  const DLDevice device = tensor.device();
  if (on_gpu(tensor)) {
    if (device.device_id != volume_->device()) {
      return holoscan::make_unexpected(invalid("an input is in " + where(device) +
                                               ", the volume on GPU " +
                                               std::to_string(volume_->device())));
    }
    if (own && bytes != 0U) {
      // The runtime orders device input only on its own streams, and its
      // release only after them: wait for the producer here, and for this
      // activation's work in settle().
      if (auto ready = tensor.synchronize_for_consumer_stream(stream); !ready) {
        return holoscan::make_unexpected(std::move(ready).error());
      }
      impl_->settle = true;
    }
    return tensor.data();
  }
  if (device.device_type != kDLCPU && device.device_type != kDLCUDAHost) {
    return holoscan::make_unexpected(invalid("an input is in " + where(device) +
                                             "; use host, pinned host, device or managed memory"));
  }
  if (bytes == 0U) {
    return tensor.data();
  }
  // The runtime orders device, managed and stream-ready pinned memory on the
  // operator's stream before compute(); plain host memory it does not.
  if (device.device_type == kDLCPU) {
    if (auto ready = tensor.synchronize_for_host(); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
  }
  if (!impl_->logged_host_copy) {
    impl_->logged_host_copy = true;
    HOLOSCAN_LOG_INFO(
        "{}: input in host memory; copying it to the GPU every frame (from pageable memory "
        "the copy waits until it is staged; send pinned or device memory to avoid that)",
        name_);
  }
  // Sized once to the largest input, so that it never grows (which would
  // wait for the stream to free the old buffer).
  DeviceBuffer& staging = impl_->staging[slot == 0 ? 0 : 1];
  if (const cudaError_t status = staging.reserve(std::max(bytes, capacity), stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
  }
  if (const cudaError_t status =
          cudaMemcpyAsync(staging.as<void>(), tensor.data(), bytes, cudaMemcpyHostToDevice, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
  }
  // The runtime orders the release of device, managed and stream-ready pinned
  // inputs after this operator's stream, but not of host-written pinned
  // memory, which is held here until the copy is done. Pageable memory needs
  // nothing: cudaMemcpyAsync has staged it by the time it returns.
  if (device.device_type == kDLCUDAHost) {
    if (own) {
      // Destroyed at stop(), so no allocator may wait for this stream later.
      if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
      }
    } else if (auto held = hold_until_done(tensor, stream, name_); !held) {
      return holoscan::make_unexpected(std::move(held).error());
    }
  }
  return static_cast<const void*>(staging.as<void>());
}

holoscan::expected<void, holoscan::Error> VolumeOp::settle(cudaStream_t stream) {
  if (!impl_->settle) {
    return {};
  }
  impl_->settle = false;
  // The operator's own stream is destroyed at stop(), so no allocator may be
  // left to wait for it: the input is released when compute() returns.
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
  }
  return {};
}

holoscan::expected<void, holoscan::Error> VolumeOp::allocate(
    OutputTensor& out, holoscan::Output<holoscan::Tensor>& port,
    std::span<const std::int64_t> shape, DLDataType dtype, cudaStream_t stream) {
  auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  out.loan.emplace(std::move(*loan));
  auto writer = out.loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  out.writer.emplace(std::move(*writer));
  return {};
}

holoscan::expected<void, holoscan::Error> VolumeOp::send(OutputTensor& out,
                                                         holoscan::Output<holoscan::Tensor>& port,
                                                         const char* name,
                                                         const holoscan::EmitOptions& options) {
  if (!out.loan || !out.writer) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, name_ + ": " + name + " was not written"});
  }
  auto committed = std::move(*out.writer).commit();
  out.writer.reset();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  auto sent = port.emit(std::move(*out.loan), options);
  out.loan.reset();
  if (!sent) {
    return emit_failure(name, std::move(sent).error());
  }
  return {};
}

}  // namespace holoscan::tsdf::detail
