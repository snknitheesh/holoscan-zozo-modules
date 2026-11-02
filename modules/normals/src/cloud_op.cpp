// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/normals/cloud_op.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/normals/normals.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "device_buffer.hpp"
#include "host_input.hpp"

namespace holoscan::normals {
namespace {

bool is_dtype(const DLDataType& dtype, const DLDataType& expected) {
  return dtype.code == expected.code && dtype.bits == expected.bits &&
         dtype.lanes == expected.lanes;
}

bool on_gpu(const holoscan::Tensor& tensor) {
  const DLDevice device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

// Whether a tensor in device memory is on `device`; managed memory is
// reachable from every device.
bool on_device_of(const holoscan::Tensor& tensor, int device) {
  const DLDevice where = tensor.device();
  return where.device_type != kDLCUDA || where.device_id == device;
}

// Whether the tensor is packed. Any stride is accepted on an axis of extent 1,
// which no element steps along.
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

bool aligned(const holoscan::Tensor& tensor, std::size_t alignment) {
  return !on_gpu(tensor) || reinterpret_cast<std::uintptr_t>(tensor.data()) % alignment == 0U;
}

// The runtime orders device, managed and stream-ready pinned memory on the
// operator's stream before compute(); plain host memory it does not, so the
// host waits for its producer here.
holoscan::expected<void, holoscan::Error> wait_for_host(const holoscan::Tensor& tensor) {
  if (tensor.device().device_type != kDLCPU) {
    return {};
  }
  return tensor.synchronize_for_host();
}

}  // namespace

struct CloudOp::Impl {
  std::array<detail::DeviceBuffer, 3> staging;
  std::uint32_t* host_count{};  // Pinned.

  ~Impl() {
    if (host_count != nullptr) {
      cudaFreeHost(host_count);
    }
  }
};

CloudOp::CloudOp(std::string name, std::uint32_t max_points)
    : name_(std::move(name)), max_points_(max_points), impl_(std::make_unique<Impl>()) {
  if (max_points_ < 1U || max_points_ > kMaxPointsLimit) {
    throw std::invalid_argument("max_points must be 1.." + std::to_string(kMaxPointsLimit));
  }
}

CloudOp::~CloudOp() = default;

void CloudOp::stop() {
  // Frees the staging scratch; cudaFree waits for the work on the runtime's
  // streams that may still use it.
  impl_ = std::make_unique<Impl>();
}

holoscan::Error CloudOp::invalid(const std::string& message) const {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, name_ + ": " + message};
}

holoscan::Error CloudOp::cuda_failure(const char* what, cudaError_t status) const {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         name_ + ": " + what + ": " + cudaGetErrorString(status)};
}

holoscan::unexpected<holoscan::Error> CloudOp::emit_failure(const char* port,
                                                            holoscan::Error error) const {
  // Keeps the code, so that a caller can still tell backpressure from failure.
  const char* hint = error.code == holoscan::ErrorCode::kBackpressured
                         ? " (its queue is full: the consumer is behind)"
                         : " (is the port connected?)";
  error.message = name_ + ": emitting " + port + " failed" +
                  (error.message.empty() ? std::string(hint) : ": " + error.message);
  return holoscan::make_unexpected(std::move(error));
}

holoscan::TensorOutputSpec CloudOp::device_output(DLDataType dtype, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = dtype,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = bytes},
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

holoscan::expected<CloudOp::Cloud, holoscan::Error> CloudOp::receive_cloud(
    holoscan::Input<holoscan::Tensor>& port, holoscan::ExecutionContext& context) {
  auto sample = port.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  Cloud cloud{std::move(*sample)};
  // The runtime assigns a stream to an operator with device outputs, and
  // writes their loans on that stream only.
  cloud.stream = context.cuda_stream().get();
  if (cloud.stream == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        name_ + ": the runtime gave no CUDA stream for the device outputs"});
  }
  int device = 0;
  if (const cudaError_t status = cudaStreamGetDevice(cloud.stream, &device);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamGetDevice", status));
  }

  const holoscan::Tensor& tensor = cloud.points.data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (!is_dtype(tensor.dtype(), kFloat32) || (shape.size() != 2 && shape.size() != 3) ||
      shape.back() != 3) {
    return holoscan::make_unexpected(invalid("points must be float32 [N, 3] or [H, W, 3]"));
  }
  std::uint64_t count = 1;
  for (std::size_t axis = 0; axis + 1 < shape.size(); ++axis) {
    if (shape[axis] < 0) {
      return holoscan::make_unexpected(invalid("points have a negative extent"));
    }
    // Saturates instead of wrapping, so that a huge shape cannot pass for a small one.
    const auto extent = static_cast<std::uint64_t>(shape[axis]);
    count = extent != 0 && count > std::numeric_limits<std::uint64_t>::max() / extent
                ? std::numeric_limits<std::uint64_t>::max()
                : count * extent;
  }
  if (count > max_points_) {
    return holoscan::make_unexpected(invalid(std::to_string(count) +
                                             " points exceed the configured maximum of " +
                                             std::to_string(max_points_)));
  }
  cloud.count = static_cast<std::uint32_t>(count);
  cloud.shape.assign(shape.begin(), shape.end());
  cloud.organized = shape.size() == 3;
  cloud.width = static_cast<int>(cloud.organized ? shape[1] : shape[0]);
  cloud.height = static_cast<int>(cloud.organized ? shape[0] : 1);
  if (!packed(tensor)) {
    return holoscan::make_unexpected(invalid("points must be packed"));
  }
  if (!aligned(tensor, sizeof(float))) {
    return holoscan::make_unexpected(invalid("point data must be aligned to 4 bytes"));
  }
  if (!on_device_of(tensor, device)) {
    return holoscan::make_unexpected(invalid("points are on another GPU than the operator"));
  }
  if (auto ready = wait_for_host(tensor); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  // The cloud stays in cloud.points; staging only copies it.
  const std::size_t bytes = std::size_t{cloud.count} * 3U * sizeof(float);
  if (on_gpu(tensor) || bytes == 0U) {
    cloud.xyz = static_cast<const float*>(tensor.data());
    return cloud;
  }
  detail::DeviceBuffer& staging = impl_->staging[2];
  if (const cudaError_t status = staging.reserve(bytes, cloud.stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
  }
  if (const cudaError_t status = cudaMemcpyAsync(staging.as<void>(), tensor.data(), bytes,
                                                 cudaMemcpyHostToDevice, cloud.stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
  }
  // The runtime orders the release of device, managed and stream-ready pinned
  // inputs after this operator's stream, but not of host-written pinned
  // memory, which is held here until the copy is done. Pageable memory needs
  // nothing: cudaMemcpyAsync has staged it by the time it returns.
  if (tensor.device().device_type == kDLCUDAHost) {
    if (auto held = detail::hold_until_done(tensor, cloud.stream, name_); !held) {
      return holoscan::make_unexpected(std::move(held).error());
    }
  }
  cloud.xyz = staging.as<float>();
  return cloud;
}

holoscan::expected<const void*, holoscan::Error> CloudOp::receive_matching(
    holoscan::Input<holoscan::Tensor>& port, const char* name, DLDataType dtype, const Cloud& cloud,
    std::optional<holoscan::Sample<holoscan::Tensor>>& sample, int slot) {
  auto received = port.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  sample.emplace(std::move(*received));
  const holoscan::Tensor& tensor = sample->data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const std::string kind = dtype.code == kDLFloat ? "float32" : "uint8";
  if (!is_dtype(tensor.dtype(), dtype) ||
      !std::equal(shape.begin(), shape.end(), cloud.shape.begin(), cloud.shape.end())) {
    return holoscan::make_unexpected(
        invalid(std::string(name) + " must be " + kind + " of the points' shape"));
  }
  if (!packed(tensor)) {
    return holoscan::make_unexpected(invalid(std::string(name) + " must be packed"));
  }
  if (!aligned(tensor, dtype.bits / 8U)) {
    return holoscan::make_unexpected(invalid(std::string(name) + " data must be aligned"));
  }
  int device = 0;
  if (const cudaError_t status = cudaStreamGetDevice(cloud.stream, &device);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamGetDevice", status));
  }
  if (!on_device_of(tensor, device)) {
    return holoscan::make_unexpected(
        invalid(std::string(name) + " are on another GPU than the operator"));
  }
  if (auto ready = wait_for_host(tensor); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  const std::size_t bytes = std::size_t{cloud.count} * 3U * (dtype.bits / 8U);
  if (on_gpu(tensor) || bytes == 0U) {
    return tensor.data();
  }
  detail::DeviceBuffer& staging = impl_->staging[slot == 0 ? 0 : 1];
  if (const cudaError_t status = staging.reserve(bytes, cloud.stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
  }
  if (const cudaError_t status = cudaMemcpyAsync(staging.as<void>(), tensor.data(), bytes,
                                                 cudaMemcpyHostToDevice, cloud.stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
  }
  if (tensor.device().device_type == kDLCUDAHost) {
    if (auto held = detail::hold_until_done(tensor, cloud.stream, name_); !held) {
      return holoscan::make_unexpected(std::move(held).error());
    }
  }
  return static_cast<const void*>(staging.as<void>());
}

holoscan::expected<void, holoscan::Error> CloudOp::allocate(
    OutputTensor& out, holoscan::Output<holoscan::Tensor>& port,
    std::span<const std::int64_t> shape, DLDataType dtype, const Cloud& cloud) {
  auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  out.loan.emplace(std::move(*loan));
  auto writer = out.loan->write(cloud.stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  out.writer.emplace(std::move(*writer));
  return {};
}

holoscan::expected<void, holoscan::Error> CloudOp::send(OutputTensor& out,
                                                        holoscan::Output<holoscan::Tensor>& port,
                                                        const char* name, const Cloud& cloud) {
  if (!out.loan || !out.writer) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, name_ + ": " + name + " was not written"});
  }
  auto committed = std::move(*out.writer).commit();
  out.writer.reset();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  auto sent = port.emit(std::move(*out.loan), holoscan::sensor_io::derive_from(cloud.points));
  out.loan.reset();
  if (!sent) {
    return emit_failure(name, std::move(sent).error());
  }
  return {};
}

holoscan::expected<std::uint32_t, holoscan::Error> CloudOp::wait_for_count(
    const std::uint32_t* device_count, const Cloud& cloud) {
  if (impl_->host_count == nullptr) {
    if (const cudaError_t status =
            cudaMallocHost(reinterpret_cast<void**>(&impl_->host_count), sizeof(std::uint32_t));
        status != cudaSuccess) {
      impl_->host_count = nullptr;
      return holoscan::make_unexpected(cuda_failure("cudaMallocHost", status));
    }
  }
  if (const cudaError_t status =
          cudaMemcpyAsync(impl_->host_count, device_count, sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, cloud.stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
  }
  if (const cudaError_t status = cudaStreamSynchronize(cloud.stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
  }
  const std::uint32_t count = *impl_->host_count;
  if (count > cloud.count) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, name_ + ": counted more points than the cloud has"});
  }
  return count;
}

}  // namespace holoscan::normals
