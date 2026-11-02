// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/clustering/cloud_op.hpp"

#include <algorithm>
#include <chrono>
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

#include <holoscan/clustering/clustering.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "device_buffer.hpp"
#include "host_input.hpp"

namespace holoscan::clustering {
namespace {

using std::chrono_literals::operator""ns;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

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

holoscan::TensorOutputSpec device_tensor_spec(DLDataType dtype, std::uint32_t max_rank,
                                              std::uint64_t max_bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = dtype,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::TensorBounds{.max_rank = max_rank, .max_byte_span = max_bytes},
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

void validate_settings(const CloudSettings& settings) {
  if (settings.max_points < 1U || settings.max_points > kMaxPointsLimit) {
    throw std::invalid_argument("max_points must be 1.." + std::to_string(kMaxPointsLimit));
  }
  if (settings.layout != OutputLayout::kKeep && settings.layout != OutputLayout::kDense) {
    throw std::invalid_argument("layout must be kKeep or kDense");
  }
}

struct CloudOp::Impl {
  detail::DeviceBuffer points_staging;
  detail::DeviceBuffer colors_staging;
};

CloudOp::CloudOp(std::string name, CloudSettings settings)
    : name_(std::move(name)), settings_(settings), impl_(std::make_unique<Impl>()) {
  validate_settings(settings_);
}

CloudOp::~CloudOp() = default;

holoscan::Error CloudOp::invalid(const std::string& message) const {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, name_ + ": " + message};
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

holoscan::Error CloudOp::cuda_failure(const char* what, cudaError_t status) const {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         name_ + ": " + what + ": " + cudaGetErrorString(status)};
}

void CloudOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: one cloud in compute() while the next one arrives.
  spec.input(points, "points").queue_depth(2U);
  // An optional input exists only when configured: an unconnected input would
  // never be satisfied.
  if (settings_.color) {
    spec.input(colors, "colors").queue_depth(2U);
  }
}

holoscan::Contract CloudOp::contract() const {
  holoscan::Contract result;
  if (!settings_.color) {
    result.trigger(holoscan::OnEach{points});
    return result;
  }
  // A cloud and its colors carry the same capture time.
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{points, colors}}, .window = 0ns});
  result.input(points, holoscan::Exact{});
  result.input(colors, holoscan::Exact{});
  return result;
}

void CloudOp::stop() {
  // Frees the staging buffers; cudaFree waits for the work on the runtime's
  // streams that may still use them.
  impl_ = std::make_unique<Impl>();
}

holoscan::expected<CloudOp::Cloud, holoscan::Error> CloudOp::receive(
    holoscan::ExecutionContext& context) {
  auto points_sample = points.receive();
  if (!points_sample) {
    return holoscan::make_unexpected(std::move(points_sample).error());
  }
  Cloud cloud{std::move(*points_sample)};
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
  if (count > settings_.max_points) {
    return holoscan::make_unexpected(invalid(std::to_string(count) +
                                             " points exceed the configured maximum of " +
                                             std::to_string(settings_.max_points)));
  }
  cloud.count = static_cast<std::uint32_t>(count);
  cloud.width = shape.size() == 3 ? static_cast<std::uint32_t>(shape[1]) : 0U;
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

  if (settings_.color) {
    auto colors_sample = colors.receive();
    if (!colors_sample) {
      return holoscan::make_unexpected(std::move(colors_sample).error());
    }
    cloud.colors.emplace(std::move(*colors_sample));
    const holoscan::Tensor& rgb = cloud.colors->data;
    const std::span<const std::int64_t> rgb_shape = rgb.shape_span();
    if (!is_dtype(rgb.dtype(), kUInt8) ||
        !std::equal(rgb_shape.begin(), rgb_shape.end(), shape.begin(), shape.end())) {
      return holoscan::make_unexpected(invalid("colors must be uint8 RGB of the points' shape"));
    }
    if (!packed(rgb)) {
      return holoscan::make_unexpected(invalid("colors must be packed"));
    }
    if (!on_device_of(rgb, device)) {
      return holoscan::make_unexpected(invalid("colors are on another GPU than the operator"));
    }
    if (auto ready = wait_for_host(rgb); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
  }

  // Host input goes to the GPU through staging buffers. The runtime orders
  // the release of device, managed and stream-ready pinned inputs after this
  // operator's stream, but not of host-written pinned memory, which is held
  // here until the copy is done. Pageable memory needs nothing:
  // cudaMemcpyAsync has staged it by the time it returns.
  const auto on_device =
      [&](const holoscan::Tensor& input, std::size_t bytes,
          detail::DeviceBuffer& staging) -> holoscan::expected<const void*, holoscan::Error> {
    if (on_gpu(input) || bytes == 0U) {
      return input.data();
    }
    if (const cudaError_t status = staging.reserve(bytes, cloud.stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    if (const cudaError_t status = cudaMemcpyAsync(staging.as<void>(), input.data(), bytes,
                                                   cudaMemcpyHostToDevice, cloud.stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
    }
    if (input.device().device_type == kDLCUDAHost) {
      if (auto held = detail::hold_until_done(input, cloud.stream, name_); !held) {
        return holoscan::make_unexpected(std::move(held).error());
      }
    }
    return static_cast<const void*>(staging.as<void>());
  };
  auto xyz =
      on_device(tensor, std::size_t{cloud.count} * 3U * sizeof(float), impl_->points_staging);
  if (!xyz) {
    return holoscan::make_unexpected(std::move(xyz).error());
  }
  cloud.xyz = static_cast<const float*>(*xyz);
  if (cloud.colors) {
    auto rgb = on_device(cloud.colors->data, std::size_t{cloud.count} * 3U, impl_->colors_staging);
    if (!rgb) {
      return holoscan::make_unexpected(std::move(rgb).error());
    }
    cloud.rgb = static_cast<const std::uint8_t*>(*rgb);
  }
  return cloud;
}

holoscan::expected<void, holoscan::Error> CloudOp::emit_tensor(
    holoscan::Output<holoscan::Tensor>& port, const char* port_name, const Cloud& cloud,
    std::span<const std::int64_t> shape, DLDataType dtype, const Writer& write) {
  auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(cloud.stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  std::int64_t elements = 1;
  for (const std::int64_t extent : shape) {
    elements *= extent;
  }
  if (elements != 0) {
    if (const cudaError_t status = write(writer->data()); status != cudaSuccess) {
      return holoscan::make_unexpected(
          cuda_failure((std::string("writing ") + port_name).c_str(), status));
    }
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  if (auto sent = port.emit(std::move(*loan), holoscan::sensor_io::derive_from(cloud.points));
      !sent) {
    return emit_failure(port_name, std::move(sent).error());
  }
  return {};
}

}  // namespace holoscan::clustering
