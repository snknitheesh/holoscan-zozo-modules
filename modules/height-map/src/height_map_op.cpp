// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/height_map/height_map_op.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/height_map/views.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "device_buffer.hpp"
#include "host_input.hpp"
#include "tensor_checks.hpp"

namespace holoscan::height_map {
namespace {

using std::chrono_literals::operator""ns;
using detail::is_dtype;
using detail::on_gpu;
using detail::packed;

}  // namespace

void validate_config(const HeightMapConfig& config) {
  const char* what = nullptr;
  if (!valid_params(config.map, &what)) {
    throw std::invalid_argument(what);
  }
  if (!config.pose_input && !is_rigid(config.sensor_pose)) {
    throw std::invalid_argument("sensor_pose must be a rigid transform");
  }
  if (!std::isfinite(config.image_low) || !std::isfinite(config.image_high)) {
    throw std::invalid_argument("image_low and image_high must be finite");
  }
  if (config.max_points < 1U || config.max_points > kMaxPointsLimit) {
    throw std::invalid_argument("max_points must be 1.." + std::to_string(kMaxPointsLimit));
  }
}

struct HeightMapOp::Impl {
  explicit Impl(const HeightMapParams& params) : mapper(params) {}

  HeightMapper mapper;
  detail::DeviceBuffer staging;  // Host clouds, copied to the GPU.
  detail::DeviceBuffer range;    // The image's automatic color range.
  bool logged_host_copy{};
  bool logged_pose_copy{};
};

HeightMapOp::HeightMapOp(HeightMapConfig config)
    : GridOp("HeightMapOp"), config_(std::move(config)) {
  validate_config(config_);
  impl_ = std::make_unique<Impl>(config_.map);
}

HeightMapOp::~HeightMapOp() = default;

void HeightMapOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t cells = config_.map.geometry.cells();
  // Two slots per input: one cloud in compute() while the next one arrives.
  spec.input(points, "points").queue_depth(2U);
  // Optional ports exist only when configured: an unconnected input would
  // never be satisfied, and an emit into an unconnected output fails.
  if (config_.pose_input) {
    spec.input(pose, "pose").queue_depth(2U);
  }
  const holoscan::TensorOutputSpec layer = layer_output(kFloat32, cells * sizeof(float));
  spec.output(elevation, "elevation").produces_tensor(layer);
  if (config_.outputs.variance) {
    spec.output(variance, "variance").produces_tensor(layer);
  }
  if (config_.outputs.count) {
    spec.output(count, "count").produces_tensor(layer);
  }
  if (config_.outputs.min_max) {
    spec.output(min_elevation, "min_elevation").produces_tensor(layer);
    spec.output(max_elevation, "max_elevation").produces_tensor(layer);
  }
  if (config_.outputs.time_since_update) {
    spec.output(time_since_update, "time_since_update").produces_tensor(layer);
  }
  spec.output(map_info, "map_info").produces_tensor(info_output());
  if (config_.outputs.image) {
    declare_image(spec, elevation_image, "elevation_image", cells * 3U);
  }
}

holoscan::Contract HeightMapOp::contract() const {
  holoscan::Contract result;
  if (!config_.pose_input) {
    result.trigger(holoscan::OnEach{points});
    return result;
  }
  // A cloud and its pose carry the same capture time.
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{points, pose}}, .window = 0ns});
  result.input(points, holoscan::Exact{});
  result.input(pose, holoscan::Exact{});
  return result;
}

void HeightMapOp::start() { impl_->mapper.reset(); }

void HeightMapOp::stop() {
  // Frees the map and the scratch; cudaFree waits for the work on the
  // runtime's streams that may still use them.
  impl_ = std::make_unique<Impl>(config_.map);
}

holoscan::expected<void, holoscan::Error> HeightMapOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = points.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  auto stream_or = stream_of(context);
  if (!stream_or) {
    return holoscan::make_unexpected(std::move(stream_or).error());
  }
  cudaStream_t stream = *stream_or;
  int device = 0;
  if (const cudaError_t status = cudaStreamGetDevice(stream, &device); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamGetDevice", status));
  }

  // The cloud.
  const holoscan::Tensor& tensor = sample->data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (!is_dtype(tensor.dtype(), kFloat32) || (shape.size() != 2 && shape.size() != 3) ||
      shape.back() != 3) {
    return holoscan::make_unexpected(invalid("points must be float32 [N, 3] or [H, W, 3]"));
  }
  std::uint64_t n = 1;
  for (std::size_t axis = 0; axis + 1 < shape.size(); ++axis) {
    if (shape[axis] < 0) {
      return holoscan::make_unexpected(invalid("points have a negative extent"));
    }
    // Saturates instead of wrapping, so that a huge shape cannot pass for a small one.
    const auto extent = static_cast<std::uint64_t>(shape[axis]);
    n = extent != 0 && n > std::numeric_limits<std::uint64_t>::max() / extent
            ? std::numeric_limits<std::uint64_t>::max()
            : n * extent;
  }
  if (n > config_.max_points) {
    return holoscan::make_unexpected(invalid(std::to_string(n) +
                                             " points exceed the configured maximum of " +
                                             std::to_string(config_.max_points)));
  }
  if (!packed(tensor)) {
    return holoscan::make_unexpected(invalid("points must be packed"));
  }
  if (on_gpu(tensor) && reinterpret_cast<std::uintptr_t>(tensor.data()) % sizeof(float) != 0U) {
    return holoscan::make_unexpected(invalid("point data must be aligned to 4 bytes"));
  }
  if (tensor.device().device_type == kDLCUDA && tensor.device().device_id != device) {
    return holoscan::make_unexpected(invalid("points are on another GPU than the operator"));
  }
  const auto point_count = static_cast<std::uint32_t>(n);
  const std::size_t bytes = std::size_t{point_count} * 3U * sizeof(float);
  const float* xyz = static_cast<const float*>(tensor.data());
  if (!on_gpu(tensor) && bytes != 0U) {
    // The runtime orders device, managed and stream-ready pinned memory on
    // the operator's stream before compute(); plain host memory it does not,
    // so the host waits for its producer here.
    if (tensor.device().device_type == kDLCPU) {
      if (auto ready = tensor.synchronize_for_host(); !ready) {
        return holoscan::make_unexpected(std::move(ready).error());
      }
    }
    if (!impl_->logged_host_copy) {
      impl_->logged_host_copy = true;
      HOLOSCAN_LOG_INFO(
          "HeightMapOp: clouds arrive in host memory and are copied to the GPU; send them in "
          "device memory to avoid the copy");
    }
    if (const cudaError_t status = impl_->staging.reserve(bytes, stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    if (const cudaError_t status = cudaMemcpyAsync(impl_->staging.as<void>(), tensor.data(), bytes,
                                                   cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
    }
    // The runtime orders the release of stream-ready pinned inputs after this
    // operator's stream, but not of host-written pinned memory, which is held
    // here until the copy is done. Pageable memory needs nothing:
    // cudaMemcpyAsync has staged it by the time it returns.
    if (tensor.device().device_type == kDLCUDAHost) {
      if (auto held = detail::hold_until_done(tensor, stream, op_name()); !held) {
        return holoscan::make_unexpected(std::move(held).error());
      }
    }
    xyz = impl_->staging.as<float>();
  }

  // The pose.
  Transform sensor_to_map = config_.sensor_pose;
  if (config_.pose_input) {
    auto pose_sample = pose.receive();
    if (!pose_sample) {
      return holoscan::make_unexpected(std::move(pose_sample).error());
    }
    const holoscan::Tensor& matrix = pose_sample->data;
    const std::span<const std::int64_t> pose_shape = matrix.shape_span();
    const bool f32 = is_dtype(matrix.dtype(), kFloat32);
    if (!(f32 || is_dtype(matrix.dtype(), kFloat64)) || pose_shape.size() != 2 ||
        pose_shape[0] != 4 || pose_shape[1] != 4 || !packed(matrix)) {
      return holoscan::make_unexpected(invalid("pose must be float32 or float64 [4, 4]"));
    }
    if (on_gpu(matrix) && !impl_->logged_pose_copy) {
      impl_->logged_pose_copy = true;
      HOLOSCAN_LOG_INFO(
          "HeightMapOp: poses arrive in device memory and are copied to the host, waiting for "
          "the GPU; send them in host memory to avoid the wait");
    }
    std::array<std::byte, 16U * sizeof(double)> raw{};
    const std::size_t size = 16U * (f32 ? sizeof(float) : sizeof(double));
    if (auto copied = matrix.copy_to_host(raw.data(), size, stream); !copied) {
      return holoscan::make_unexpected(std::move(copied).error());
    }
    std::array<double, 16> m{};
    for (std::size_t i = 0; i < 16U; ++i) {
      if (f32) {
        float value = 0.0F;
        std::memcpy(&value, raw.data() + i * sizeof(float), sizeof(float));
        m[i] = value;
      } else {
        std::memcpy(&m[i], raw.data() + i * sizeof(double), sizeof(double));
      }
    }
    for (std::size_t i = 0; i < 12U; ++i) {
      sensor_to_map.m[i] = m[i];
    }
    const bool affine_row = m[12] == 0.0 && m[13] == 0.0 && m[14] == 0.0 && m[15] == 1.0;
    if (!affine_row || !is_rigid(sensor_to_map)) {
      return holoscan::make_unexpected(invalid("pose is not a rigid transform"));
    }
  }

  // The map.
  const holoscan::SampleMetadata& metadata = sample->metadata;
  const std::int64_t stamp = metadata.source_clock_id.valid()
                                 ? metadata.capture_timestamp_ns
                                 : context.activation_time().timestamp_ns;
  if (const cudaError_t status =
          impl_->mapper.update(xyz, point_count, sensor_to_map, stamp, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("map update", status));
  }

  // The outputs.
  const MapGeometry geometry = impl_->mapper.geometry();
  const std::array<std::int64_t, 2> layer_shape{geometry.rows, geometry.cols};
  const HeightMapOutputs& wanted = config_.outputs;
  OutputTensor out_elevation;
  OutputTensor out_variance;
  OutputTensor out_count;
  OutputTensor out_min;
  OutputTensor out_max;
  OutputTensor out_time;
  OutputTensor out_image;
  const auto make = [&](OutputTensor& out, holoscan::Output<holoscan::Tensor>& port,
                        bool enabled) -> holoscan::expected<void, holoscan::Error> {
    return enabled ? allocate(out, port, layer_shape, kFloat32, stream)
                   : holoscan::expected<void, holoscan::Error>{};
  };
  for (auto made :
       {make(out_elevation, elevation, true), make(out_variance, variance, wanted.variance),
        make(out_count, count, wanted.count), make(out_min, min_elevation, wanted.min_max),
        make(out_max, max_elevation, wanted.min_max),
        make(out_time, time_since_update, wanted.time_since_update)}) {
    if (!made) {
      return made;
    }
  }
  const MapLayers layers{.elevation = out_elevation.data<float>(),
                         .variance = out_variance.data<float>(),
                         .count = out_count.data<float>(),
                         .min_elevation = out_min.data<float>(),
                         .max_elevation = out_max.data<float>(),
                         .time_since_update = out_time.data<float>()};
  if (const cudaError_t status = impl_->mapper.write_layers(layers, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("writing the layers", status));
  }
  if (wanted.image) {
    const std::array<std::int64_t, 3> image_shape{geometry.rows, geometry.cols, 3};
    if (auto made = allocate(out_image, elevation_image, image_shape, kUInt8, stream); !made) {
      return made;
    }
    if (const cudaError_t status = impl_->range.reserve(2U * sizeof(float), stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    if (const cudaError_t status = colorize_elevation(
            out_elevation.data<float>(), geometry.rows, geometry.cols, config_.image_low,
            config_.image_high, impl_->range.as<float>(), out_image.data<std::uint8_t>(), stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("colorizing the elevation", status));
    }
  }

  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*sample);
  if (auto sent = send(out_elevation, elevation, "elevation", emit); !sent) {
    return sent;
  }
  if (wanted.variance) {
    if (auto sent = send(out_variance, variance, "variance", emit); !sent) {
      return sent;
    }
  }
  if (wanted.count) {
    if (auto sent = send(out_count, count, "count", emit); !sent) {
      return sent;
    }
  }
  if (wanted.min_max) {
    if (auto sent = send(out_min, min_elevation, "min_elevation", emit); !sent) {
      return sent;
    }
    if (auto sent = send(out_max, max_elevation, "max_elevation", emit); !sent) {
      return sent;
    }
  }
  if (wanted.time_since_update) {
    if (auto sent = send(out_time, time_since_update, "time_since_update", emit); !sent) {
      return sent;
    }
  }
  if (auto sent = send_info(map_info, "map_info", geometry, emit); !sent) {
    return sent;
  }
  if (wanted.image) {
    return send_image(out_image, elevation_image, "elevation_image", geometry.rows, geometry.cols,
                      config_.frame_id, emit);
  }
  return {};
}

}  // namespace holoscan::height_map
