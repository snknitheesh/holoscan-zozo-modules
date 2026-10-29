// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/pcl_filters/cloud_filter_op.hpp"

#include <algorithm>
#include <array>
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

#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "device_buffer.hpp"
#include "host_input.hpp"

namespace holoscan::pcl_filters {
namespace {

using std::chrono_literals::operator""ns;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// Device memory of any shape up to rank 3: organized and dense clouds alike.
holoscan::TensorOutputSpec device_cloud(DLDataType dtype, std::uint64_t bytes) {
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

void validate_settings(const CloudSettings& settings) {
  if (settings.max_points < 1U || settings.max_points > kMaxPointsLimit) {
    throw std::invalid_argument("max_points must be 1.." + std::to_string(kMaxPointsLimit));
  }
  if (settings.layout != OutputLayout::kKeep && settings.layout != OutputLayout::kDense) {
    throw std::invalid_argument("layout must be kKeep or kDense");
  }
}

struct CloudFilterOp::Impl {
  detail::DeviceBuffer points_staging;
  detail::DeviceBuffer colors_staging;
  detail::DeviceBuffer keep;
  Compactor compactor;
  std::uint32_t* host_count{};  // Pinned.

  ~Impl() { release(); }
  void release() noexcept {
    if (host_count != nullptr) {
      cudaFreeHost(host_count);
      host_count = nullptr;
    }
  }
};

CloudFilterOp::CloudFilterOp(std::string name, CloudSettings settings)
    : name_(std::move(name)), settings_(settings), impl_(std::make_unique<Impl>()) {
  validate_settings(settings_);
}

CloudFilterOp::~CloudFilterOp() = default;

holoscan::Error CloudFilterOp::invalid(const std::string& message) const {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, name_ + ": " + message};
}

holoscan::unexpected<holoscan::Error> CloudFilterOp::emit_failure(const char* port,
                                                                  holoscan::Error error) const {
  // Keeps the code, so that a caller can still tell backpressure from failure.
  const char* hint = error.code == holoscan::ErrorCode::kBackpressured
                         ? " (its queue is full: the consumer is behind)"
                         : " (is the port connected?)";
  error.message = name_ + ": emitting " + port + " failed" +
                  (error.message.empty() ? std::string(hint) : ": " + error.message);
  return holoscan::make_unexpected(std::move(error));
}

holoscan::Error CloudFilterOp::cuda_failure(const char* what, cudaError_t status) const {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         name_ + ": " + what + ": " + cudaGetErrorString(status)};
}

void CloudFilterOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t points_max = settings_.max_points;
  // Two slots per input: one cloud in compute() while the next one arrives.
  spec.input(points, "points").queue_depth(2U);
  // Optional ports exist only when configured: an unconnected input would
  // never be satisfied, and an emit into an unconnected output fails.
  if (settings_.color) {
    spec.input(colors, "colors").queue_depth(2U);
  }
  spec.output(filtered_points, "filtered_points")
      .produces_tensor(device_cloud(kFloat32, points_max * 3U * sizeof(float)));
  if (settings_.color) {
    spec.output(filtered_colors, "filtered_colors")
        .produces_tensor(device_cloud(kUInt8, points_max * 3U));
  }
}

holoscan::Contract CloudFilterOp::contract() const {
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

void CloudFilterOp::stop() {
  // Frees the staging and output scratch; cudaFree waits for the work on the
  // runtime's streams that may still use it.
  impl_ = std::make_unique<Impl>();
}

holoscan::expected<CloudFilterOp::Cloud, holoscan::Error> CloudFilterOp::receive(
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

holoscan::expected<std::uint8_t*, holoscan::Error> CloudFilterOp::mask(const Cloud& cloud) {
  if (const cudaError_t status =
          impl_->keep.reserve(std::max<std::size_t>(cloud.count, 1U), cloud.stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
  }
  return impl_->keep.as<std::uint8_t>();
}

holoscan::expected<void, holoscan::Error> CloudFilterOp::emit_kept(const Cloud& cloud,
                                                                   const std::uint8_t* keep) {
  if (settings_.layout == OutputLayout::kDense) {
    if (const cudaError_t status = impl_->compactor.count(keep, cloud.count, cloud.stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("compaction", status));
    }
    return emit_dense(cloud, impl_->compactor.device_count(),
                      [this, &cloud](float* xyz, std::uint8_t* rgb) {
                        return impl_->compactor.write(cloud.xyz, cloud.rgb, xyz, rgb, cloud.stream);
                      });
  }
  const std::span<const std::int64_t> shape = cloud.points.data.shape_span();
  const std::vector<std::int64_t> out_shape(shape.begin(), shape.end());
  auto points_loan = filtered_points.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = out_shape, .dtype = kFloat32});
  if (!points_loan) {
    return holoscan::make_unexpected(std::move(points_loan).error());
  }
  auto points_writer = points_loan->write(cloud.stream);
  if (!points_writer) {
    return holoscan::make_unexpected(std::move(points_writer).error());
  }
  std::optional<holoscan::TensorOutputLoan> colors_loan;
  std::optional<holoscan::TensorOutputWriteGuard> colors_writer;
  if (settings_.color) {
    auto loan = filtered_colors.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = out_shape, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    colors_loan.emplace(std::move(*loan));
    auto writer = colors_loan->write(cloud.stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    colors_writer.emplace(std::move(*writer));
  }
  if (const cudaError_t status = apply_mask(
          cloud.xyz, cloud.rgb, keep, cloud.count, points_writer->data_as<float>(),
          colors_writer ? colors_writer->data_as<std::uint8_t>() : nullptr, cloud.stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("apply_mask", status));
  }
  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(cloud.points);
  if (auto committed = std::move(*points_writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  if (auto sent = filtered_points.emit(std::move(*points_loan), emit); !sent) {
    return emit_failure("filtered_points", std::move(sent).error());
  }
  if (settings_.color) {
    if (auto committed = std::move(*colors_writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    if (auto sent = filtered_colors.emit(std::move(*colors_loan), emit); !sent) {
      return emit_failure("filtered_colors", std::move(sent).error());
    }
  }
  return {};
}

holoscan::expected<void, holoscan::Error> CloudFilterOp::emit_dense(
    const Cloud& cloud, const std::uint32_t* device_count, const DenseWriter& write) {
  if (impl_->host_count == nullptr) {
    if (const cudaError_t status =
            cudaMallocHost(reinterpret_cast<void**>(&impl_->host_count), sizeof(std::uint32_t));
        status != cudaSuccess) {
      impl_->host_count = nullptr;
      return holoscan::make_unexpected(cuda_failure("cudaMallocHost", status));
    }
  }
  // The output's size is the filter's result: wait for it.
  if (const cudaError_t status =
          cudaMemcpyAsync(impl_->host_count, device_count, sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, cloud.stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
  }
  if (const cudaError_t status = cudaStreamSynchronize(cloud.stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
  }
  const std::uint32_t kept = *impl_->host_count;
  if (kept > cloud.count) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, name_ + ": the filter counted more points than it got"});
  }
  const std::array<std::int64_t, 2> out_shape{kept, 3};
  auto points_loan = filtered_points.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = out_shape, .dtype = kFloat32});
  if (!points_loan) {
    return holoscan::make_unexpected(std::move(points_loan).error());
  }
  auto points_writer = points_loan->write(cloud.stream);
  if (!points_writer) {
    return holoscan::make_unexpected(std::move(points_writer).error());
  }
  std::optional<holoscan::TensorOutputLoan> colors_loan;
  std::optional<holoscan::TensorOutputWriteGuard> colors_writer;
  if (settings_.color) {
    auto loan = filtered_colors.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = out_shape, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    colors_loan.emplace(std::move(*loan));
    auto writer = colors_loan->write(cloud.stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    colors_writer.emplace(std::move(*writer));
  }
  if (kept != 0U) {
    if (const cudaError_t status =
            write(points_writer->data_as<float>(),
                  colors_writer ? colors_writer->data_as<std::uint8_t>() : nullptr);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("filter kernels", status));
    }
  }
  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(cloud.points);
  if (auto committed = std::move(*points_writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  if (auto sent = filtered_points.emit(std::move(*points_loan), emit); !sent) {
    return emit_failure("filtered_points", std::move(sent).error());
  }
  if (settings_.color) {
    if (auto committed = std::move(*colors_writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    if (auto sent = filtered_colors.emit(std::move(*colors_loan), emit); !sent) {
      return emit_failure("filtered_colors", std::move(sent).error());
    }
  }
  return {};
}

}  // namespace holoscan::pcl_filters
