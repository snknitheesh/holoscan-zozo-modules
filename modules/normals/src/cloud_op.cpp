// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/normals/cloud_op.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/normals/normals.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "device_buffer.hpp"
#include "host_input.hpp"

namespace holoscan::normals {
namespace {

namespace hm = holoscan::messages;

constexpr std::array<const char*, 4> kFieldNames{"points", "colors", "normals", "intensity"};

bool on_gpu_memory(const holoscan::Tensor& tensor) {
  const DLDevice device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

// Whether a tensor in device memory is on `device`; managed memory is
// reachable from every device.
bool on_device_of(const holoscan::Tensor& tensor, int device) {
  const DLDevice where = tensor.device();
  return where.device_type != kDLCUDA || where.device_id == device;
}

bool aligned(const holoscan::Tensor& tensor, std::size_t alignment) {
  return !on_gpu_memory(tensor) ||
         reinterpret_cast<std::uintptr_t>(tensor.data()) % alignment == 0U;
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

const char* memory_name(const holoscan::Tensor& tensor) {
  return tensor.device().device_type == kDLCUDAHost ? "pinned host" : "host";
}

}  // namespace

struct CloudOp::Impl {
  // One staging buffer per field, for host input.
  std::array<detail::DeviceBuffer, 4> staging;
  std::uint32_t* host_count{};  // Pinned.
  cudaEvent_t counted{};
  // Whether the copy of host input to the GPU has been logged.
  bool copy_logged{};

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  ~Impl() {
    if (host_count != nullptr) {
      cudaFreeHost(host_count);
    }
    if (counted != nullptr) {
      cudaEventDestroy(counted);
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
  // streams that may still use it, on the device the runtime made current
  // for the operator.
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

holoscan::expected<const void*, holoscan::Error> CloudOp::on_gpu(const holoscan::Tensor& tensor,
                                                                 std::size_t bytes, int field,
                                                                 const Cloud& cloud,
                                                                 const char* port_name) {
  if (auto ready = wait_for_host(tensor); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  if (on_gpu_memory(tensor) || bytes == 0U) {
    return bytes == 0U ? nullptr : tensor.data();
  }
  if (!impl_->copy_logged) {
    impl_->copy_logged = true;
    HOLOSCAN_LOG_INFO(
        "{}: the clouds on {} are in {} memory, so each frame's fields are copied to the GPU "
        "({} bytes of {} this frame); send clouds in device memory to avoid the copy",
        name_, port_name, memory_name(tensor), bytes, kFieldNames[static_cast<std::size_t>(field)]);
  }
  detail::DeviceBuffer& staging = impl_->staging[static_cast<std::size_t>(field)];
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
  return static_cast<const void*>(staging.as<void>());
}

holoscan::expected<CloudOp::Cloud, holoscan::Error> CloudOp::receive_cloud(
    holoscan::Input<hm::schema::PointCloudT>& port, const char* port_name,
    holoscan::ExecutionContext& context) {
  auto sample = port.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  Cloud cloud{.sample = std::move(*sample)};
  const hm::schema::PointCloudT& message = cloud.sample.data;
  if (auto checked = hm::check_point_cloud(message); !checked) {
    return holoscan::make_unexpected(
        invalid(std::string(port_name) + ": " + checked.error().message));
  }
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
  if (message.header) {
    cloud.header = hm::from_message(*message.header);
  }

  // check_point_cloud() has checked the layout: float32 [N, 3] or [H, W, 3],
  // every present field of the points' shape, packed, in the points' memory.
  const holoscan::Tensor& points = *message.points;
  const std::span<const std::int64_t> shape = points.shape_span();
  cloud.organized = message.organized;
  std::uint64_t count = 1;
  for (std::size_t axis = 0; axis + 1 < shape.size(); ++axis) {
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
  cloud.width = static_cast<int>(cloud.organized ? shape[1] : shape[0]);
  cloud.height = static_cast<int>(cloud.organized ? shape[0] : 1);
  cloud.has_colors = hm::is_present(message.colors);
  cloud.has_normals = hm::is_present(message.normals);
  cloud.has_intensity = hm::is_present(message.intensity);
  if (!aligned(points, sizeof(float)) ||
      (cloud.has_normals && !aligned(*message.normals, sizeof(float))) ||
      (cloud.has_intensity && !aligned(*message.intensity, sizeof(float)))) {
    return holoscan::make_unexpected(invalid("float32 fields must be aligned to 4 bytes"));
  }
  if (!on_device_of(points, device)) {
    return holoscan::make_unexpected(invalid("the cloud is on another GPU than the operator"));
  }

  const std::size_t n = cloud.count;
  auto xyz = on_gpu(points, n * 3U * sizeof(float), 0, cloud, port_name);
  if (!xyz) {
    return holoscan::make_unexpected(std::move(xyz).error());
  }
  cloud.xyz = static_cast<const float*>(*xyz);
  if (cloud.has_colors) {
    auto rgb = on_gpu(*message.colors, n * 3U, 1, cloud, port_name);
    if (!rgb) {
      return holoscan::make_unexpected(std::move(rgb).error());
    }
    cloud.rgb = static_cast<const std::uint8_t*>(*rgb);
  }
  if (cloud.has_normals) {
    auto normals = on_gpu(*message.normals, n * 3U * sizeof(float), 2, cloud, port_name);
    if (!normals) {
      return holoscan::make_unexpected(std::move(normals).error());
    }
    cloud.normals = static_cast<const float*>(*normals);
  }
  if (cloud.has_intensity) {
    auto intensity = on_gpu(*message.intensity, n * sizeof(float), 3, cloud, port_name);
    if (!intensity) {
      return holoscan::make_unexpected(std::move(intensity).error());
    }
    cloud.intensity = static_cast<const float*>(*intensity);
  }
  return cloud;
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
  auto sent = port.emit(std::move(*out.loan), holoscan::sensor_io::derive_from(cloud.sample));
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
  if (impl_->counted == nullptr) {
    if (const cudaError_t status =
            cudaEventCreateWithFlags(&impl_->counted, cudaEventDisableTiming);
        status != cudaSuccess) {
      impl_->counted = nullptr;
      return holoscan::make_unexpected(cuda_failure("cudaEventCreate", status));
    }
  }
  if (const cudaError_t status =
          cudaMemcpyAsync(impl_->host_count, device_count, sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, cloud.stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
  }
  if (const cudaError_t status = cudaEventRecord(impl_->counted, cloud.stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaEventRecord", status));
  }
  if (const cudaError_t status = cudaEventSynchronize(impl_->counted); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaEventSynchronize", status));
  }
  const std::uint32_t count = *impl_->host_count;
  if (count > cloud.count) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, name_ + ": counted more points than the cloud has"});
  }
  return count;
}

}  // namespace holoscan::normals
