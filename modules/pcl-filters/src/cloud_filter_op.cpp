// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/pcl_filters/cloud_filter_op.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "device_buffer.hpp"
#include "host_input.hpp"

namespace holoscan::pcl_filters {
namespace {

namespace hm = holoscan::messages;
using FieldSelector = holoscan::TensorFieldSelector<hm::schema::PointCloudT>;

// The message's four tensors, in the order of CloudFields.
constexpr std::size_t kPoints = 0;
constexpr std::size_t kColors = 1;
constexpr std::size_t kNormals = 2;
constexpr std::size_t kIntensity = 3;
constexpr std::size_t kFields = 4;
struct FieldInfo {
  FieldSelector selector;
  const char* name;
  std::size_t bytes_per_point;
  std::size_t alignment;
};

const FieldInfo& field_info(std::size_t field) {
  static constexpr std::array<FieldInfo, kFields> kInfo{{
      {hm::PointCloudFields::points, "points", 3U * sizeof(float), sizeof(float)},
      {hm::PointCloudFields::colors, "colors", 3U, 1U},
      {hm::PointCloudFields::normals, "normals", 3U * sizeof(float), sizeof(float)},
      {hm::PointCloudFields::intensity, "intensity", sizeof(float), sizeof(float)},
  }};
  return kInfo[field];
}

std::array<const std::shared_ptr<holoscan::Tensor>*, kFields> tensors_of(
    const hm::schema::PointCloudT& message) {
  return {&message.points, &message.colors, &message.normals, &message.intensity};
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

// N, or H * W, saturating rather than wrapping, so that a huge shape cannot
// pass for a small one. The shape is a checked cloud's points: [N, 3] or
// [H, W, 3].
std::uint64_t point_count(std::span<const std::int64_t> shape) {
  std::uint64_t count = 1;
  for (std::size_t axis = 0; axis + 1 < shape.size(); ++axis) {
    const auto extent = static_cast<std::uint64_t>(shape[axis]);
    count = extent != 0 && count > std::numeric_limits<std::uint64_t>::max() / extent
                ? std::numeric_limits<std::uint64_t>::max()
                : count * extent;
  }
  return count;
}

// Whether a cloud with `contents` has `field`; every cloud has points.
bool has_field(const hm::PointCloudContents& contents, std::size_t field) {
  switch (field) {
    case kColors:
      return contents.colors;
    case kNormals:
      return contents.normals;
    case kIntensity:
      return contents.intensity;
    default:
      return true;
  }
}

// Opens the device writes of `fields` one inside the other, keeping each
// one's address in `data`, and runs `fill` with all of them open: one kernel
// writes every field before any is committed.
template <typename Fill>
holoscan::expected<void, holoscan::Error> write_fields(hm::PointCloudLoan& loan,
                                                       std::span<const std::size_t> fields,
                                                       cudaStream_t stream,
                                                       std::array<void*, kFields>& data,
                                                       const Fill& fill) {
  if (fields.empty()) {
    return fill();
  }
  const std::size_t field = fields.front();
  return loan.write_and_commit(field_info(field).selector, ::cuda::stream_ref{stream},
                               [&](const holoscan::TensorOutputWriteGuard& guard)
                                   -> holoscan::expected<void, holoscan::Error> {
                                 data[field] = guard.data();
                                 return write_fields(loan, fields.subspan(1), stream, data, fill);
                               });
}

}  // namespace

void validate_settings(const CloudSettings& settings) {
  if (settings.max_points < 1U || settings.max_points > kMaxPointsLimit) {
    throw std::invalid_argument("max_points must be 1.." + std::to_string(kMaxPointsLimit));
  }
  if (settings.layout != OutputLayout::kKeep && settings.layout != OutputLayout::kDense &&
      settings.layout != OutputLayout::kPadded) {
    throw std::invalid_argument("layout must be kKeep, kDense or kPadded");
  }
}

struct CloudFilterOp::Impl {
  std::array<detail::DeviceBuffer, kFields> staging;
  detail::DeviceBuffer keep;
  Compactor compactor;
  std::uint32_t* host_count{};  // Pinned.

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  ~Impl() {
    if (host_count != nullptr) {
      cudaFreeHost(host_count);
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
  // Two slots: one cloud in compute() while the next one arrives.
  spec.input(point_cloud, "point_cloud").queue_depth(2U);
  hm::declare_point_cloud_output(spec, filtered_point_cloud, "filtered_point_cloud",
                                 hm::PointCloudOutputSpec{
                                     .max_points = settings_.max_points,
                                     .colors = settings_.fields.colors,
                                     .normals = settings_.fields.normals,
                                     .intensity = settings_.fields.intensity,
                                 });
}

holoscan::Contract CloudFilterOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{point_cloud});
  return result;
}

void CloudFilterOp::stop() {
  // Frees the staging and output scratch; cudaFree waits for the work on the
  // runtime's streams that may still use it.
  impl_ = std::make_unique<Impl>();
}

holoscan::expected<CloudFilterOp::Cloud, holoscan::Error> CloudFilterOp::receive(
    holoscan::ExecutionContext& context) {
  auto sample = point_cloud.receive();
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

  const hm::schema::PointCloudT& message = cloud.sample.data;
  if (auto checked = hm::check_point_cloud(message); !checked) {
    return holoscan::make_unexpected(invalid(checked.error().message));
  }
  const std::span<const std::int64_t> shape = message.points->shape_span();
  const std::uint64_t count = point_count(shape);
  if (count > settings_.max_points) {
    return holoscan::make_unexpected(invalid(std::to_string(count) +
                                             " points exceed the configured maximum of " +
                                             std::to_string(settings_.max_points)));
  }
  cloud.count = static_cast<std::uint32_t>(count);
  cloud.shape =
      message.organized ? hm::organized_cloud(shape[0], shape[1]) : hm::unorganized_cloud(shape[0]);

  // Checks every field the cloud carries, then copies the host ones to the
  // GPU through staging buffers. The runtime orders the release of device,
  // managed and stream-ready pinned inputs after this operator's stream, but
  // may reuse pinned memory of its own pools once compute() returns, which
  // is held here until the copies are done. Pageable memory needs nothing:
  // cudaMemcpyAsync has staged it by the time it returns.
  const auto tensors = tensors_of(message);
  std::array<const void*, kFields> data{};
  std::array<const holoscan::Tensor*, kFields> pinned{};
  std::size_t pinned_count = 0;
  cloud.contents = hm::PointCloudContents{
      .colors = hm::is_present(message.colors),
      .normals = hm::is_present(message.normals),
      .intensity = hm::is_present(message.intensity),
  };
  for (std::size_t field = 0; field < kFields; ++field) {
    if (!has_field(cloud.contents, field)) {
      continue;
    }
    const holoscan::Tensor& tensor = **tensors[field];
    const FieldInfo& info = field_info(field);
    const std::string name = info.name;
    if (!has_field(settings_.fields, field)) {
      return holoscan::make_unexpected(invalid("the cloud has " + name +
                                               ", which the output has no room for (fields." +
                                               name + " is off)"));
    }
    if (!packed(tensor)) {
      return holoscan::make_unexpected(invalid(name + " must be packed"));
    }
    if (!aligned(tensor, info.alignment)) {
      return holoscan::make_unexpected(
          invalid(name + " must be aligned to " + std::to_string(info.alignment) + " bytes"));
    }
    if (!on_device_of(tensor, device)) {
      return holoscan::make_unexpected(
          invalid("the " + name + " tensor is on another GPU than the operator"));
    }
    if (auto ready = wait_for_host(tensor); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    const std::size_t bytes = std::size_t{cloud.count} * info.bytes_per_point;
    if (on_gpu(tensor) || bytes == 0U) {
      data[field] = tensor.data();
      continue;
    }
    detail::DeviceBuffer& staging = impl_->staging[field];
    if (const cudaError_t status = staging.reserve(bytes, cloud.stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    if (const cudaError_t status = cudaMemcpyAsync(staging.as<void>(), tensor.data(), bytes,
                                                   cudaMemcpyHostToDevice, cloud.stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
    }
    if (tensor.device().device_type == kDLCUDAHost) {
      pinned[pinned_count++] = &tensor;
    }
    data[field] = staging.as<void>();
  }
  if (pinned_count != 0U) {
    if (auto held =
            detail::hold_until_done(std::span(pinned.data(), pinned_count), cloud.stream, name_);
        !held) {
      return holoscan::make_unexpected(std::move(held).error());
    }
  }
  cloud.fields = CloudFields{
      .xyz = static_cast<const float*>(data[kPoints]),
      .rgb = static_cast<const std::uint8_t*>(data[kColors]),
      .normals = static_cast<const float*>(data[kNormals]),
      .intensity = static_cast<const float*>(data[kIntensity]),
  };
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
  if (settings_.layout == OutputLayout::kKeep) {
    return emit_cloud(cloud, cloud.shape, -1, [&](const CloudOutput& out) {
      return apply_mask(cloud.fields, keep, cloud.count, out, cloud.stream);
    });
  }
  if (const cudaError_t status = impl_->compactor.count(keep, cloud.count, cloud.stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("compaction", status));
  }
  return emit_packed(cloud, impl_->compactor.device_count(), [&](const CloudOutput& out) {
    return impl_->compactor.write(cloud.fields, out, cloud.stream);
  });
}

holoscan::expected<void, holoscan::Error> CloudFilterOp::emit_packed(
    const Cloud& cloud, const std::uint32_t* device_count, const PackedWriter& write) {
  if (settings_.layout == OutputLayout::kPadded) {
    return emit_cloud(cloud, hm::unorganized_cloud(cloud.count), -1, [&](const CloudOutput& out) {
      if (cloud.count == 0U) {
        return cudaSuccess;
      }
      const cudaError_t status = write(out);
      return status != cudaSuccess ? status
                                   : pad_cloud(out, device_count, cloud.count, cloud.stream);
    });
  }
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
  // Every point a filter keeps, and every voxel, is valid.
  return emit_cloud(cloud, hm::unorganized_cloud(kept), kept,
                    [&](const CloudOutput& out) { return kept == 0U ? cudaSuccess : write(out); });
}

holoscan::expected<void, holoscan::Error> CloudFilterOp::emit_cloud(
    const Cloud& cloud, const hm::PointCloudShape& shape, std::int64_t valid_count,
    const PackedWriter& write) {
  auto loan = hm::allocate_point_cloud(filtered_point_cloud, shape, cloud.contents,
                                       ::cuda::stream_ref{cloud.stream});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  std::array<std::size_t, kFields> fields{};
  std::size_t field_count = 0;
  for (std::size_t field = 0; field < kFields; ++field) {
    if (has_field(cloud.contents, field)) {
      fields[field_count++] = field;
    }
  }
  std::array<void*, kFields> data{};
  auto written =
      write_fields(*loan, std::span(fields.data(), field_count), cloud.stream, data,
                   [&]() -> holoscan::expected<void, holoscan::Error> {
                     const CloudOutput out{
                         .xyz = static_cast<float*>(data[kPoints]),
                         .rgb = static_cast<std::uint8_t*>(data[kColors]),
                         .normals = static_cast<float*>(data[kNormals]),
                         .intensity = static_cast<float*>(data[kIntensity]),
                     };
                     if (const cudaError_t status = write(out); status != cudaSuccess) {
                       return holoscan::make_unexpected(cuda_failure("filter kernels", status));
                     }
                     return {};
                   });
  if (!written) {
    return holoscan::make_unexpected(std::move(written).error());
  }
  const hm::schema::PointCloudT& message = cloud.sample.data;
  const hm::Header header = message.header ? hm::from_message(*message.header) : hm::Header{};
  if (auto sent = hm::emit_point_cloud(filtered_point_cloud, std::move(*loan), header, shape,
                                       valid_count, holoscan::sensor_io::derive_from(cloud.sample));
      !sent) {
    return emit_failure("filtered_point_cloud", std::move(sent).error());
  }
  return {};
}

}  // namespace holoscan::pcl_filters
