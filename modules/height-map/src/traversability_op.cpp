// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/height_map/traversability_op.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
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

void validate_config(const TraversabilityConfig& config) {
  const char* what = nullptr;
  if (!valid_params(config.traversability, &what)) {
    throw std::invalid_argument(what);
  }
  if (config.max_rows < 1 || config.max_cols < 1 || config.max_rows > kMaxGridSide ||
      config.max_cols > kMaxGridSide) {
    throw std::invalid_argument("max_rows and max_cols must be 1..8192");
  }
}

struct TraversabilityOp::Impl {
  detail::DeviceBuffer staging;  // Host elevation, copied to the GPU.
  bool logged_host_copy{};
};

TraversabilityOp::TraversabilityOp(TraversabilityConfig config)
    : GridOp("TraversabilityOp"), config_(std::move(config)), impl_(std::make_unique<Impl>()) {
  validate_config(config_);
}

TraversabilityOp::~TraversabilityOp() = default;

void TraversabilityOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t cells =
      static_cast<std::uint64_t>(config_.max_rows) * static_cast<std::uint64_t>(config_.max_cols);
  spec.input(elevation, "elevation").queue_depth(2U);
  spec.input(map_info, "map_info").queue_depth(2U);
  const holoscan::TensorOutputSpec layer = layer_output(kFloat32, cells * sizeof(float));
  spec.output(cost, "cost").produces_tensor(layer);
  spec.output(unknown, "unknown").produces_tensor(layer_output(kUInt8, cells));
  if (config_.components) {
    spec.output(slope, "slope").produces_tensor(layer);
    spec.output(step_height, "step_height").produces_tensor(layer);
    spec.output(roughness, "roughness").produces_tensor(layer);
  }
  if (config_.image) {
    declare_image(spec, cost_image, "cost_image", cells * 3U);
  }
}

holoscan::Contract TraversabilityOp::contract() const {
  holoscan::Contract result;
  // A layer and its map_info carry the same capture time.
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{elevation, map_info}},
      .window = 0ns});
  result.input(elevation, holoscan::Exact{});
  result.input(map_info, holoscan::Exact{});
  return result;
}

void TraversabilityOp::stop() { impl_ = std::make_unique<Impl>(); }

holoscan::expected<void, holoscan::Error> TraversabilityOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = elevation.receive();
  auto info_sample = map_info.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  if (!info_sample) {
    return holoscan::make_unexpected(std::move(info_sample).error());
  }
  auto stream_or = stream_of(context);
  if (!stream_or) {
    return holoscan::make_unexpected(std::move(stream_or).error());
  }
  cudaStream_t stream = *stream_or;
  auto geometry = read_map_info(info_sample->data, stream);
  if (!geometry) {
    return holoscan::make_unexpected(invalid(geometry.error().message));
  }
  const holoscan::Tensor& tensor = sample->data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (!is_dtype(tensor.dtype(), kFloat32) || shape.size() != 2 || shape[0] != geometry->rows ||
      shape[1] != geometry->cols) {
    return holoscan::make_unexpected(
        invalid("elevation must be float32 [rows, cols] of its map_info"));
  }
  if (geometry->rows > config_.max_rows || geometry->cols > config_.max_cols) {
    return holoscan::make_unexpected(
        invalid("a " + std::to_string(geometry->rows) + "x" + std::to_string(geometry->cols) +
                " map exceeds the configured maximum of " + std::to_string(config_.max_rows) + "x" +
                std::to_string(config_.max_cols)));
  }
  if (!packed(tensor)) {
    return holoscan::make_unexpected(invalid("elevation must be packed"));
  }
  int device = 0;
  if (const cudaError_t status = cudaStreamGetDevice(stream, &device); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamGetDevice", status));
  }
  if (tensor.device().device_type == kDLCUDA && tensor.device().device_id != device) {
    return holoscan::make_unexpected(invalid("elevation is on another GPU than the operator"));
  }
  const std::size_t cells = geometry->cells();
  const float* heights = static_cast<const float*>(tensor.data());
  if (!on_gpu(tensor)) {
    if (tensor.device().device_type == kDLCPU) {
      if (auto ready = tensor.synchronize_for_host(); !ready) {
        return holoscan::make_unexpected(std::move(ready).error());
      }
    }
    if (!impl_->logged_host_copy) {
      impl_->logged_host_copy = true;
      HOLOSCAN_LOG_INFO(
          "TraversabilityOp: elevation arrives in host memory and is copied to the GPU; send it "
          "in device memory to avoid the copy");
    }
    if (const cudaError_t status = impl_->staging.reserve(cells * sizeof(float), stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    if (const cudaError_t status =
            cudaMemcpyAsync(impl_->staging.as<void>(), tensor.data(), cells * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
    }
    if (tensor.device().device_type == kDLCUDAHost) {
      if (auto held = detail::hold_until_done(tensor, stream, op_name()); !held) {
        return holoscan::make_unexpected(std::move(held).error());
      }
    }
    heights = impl_->staging.as<float>();
  }

  const std::array<std::int64_t, 2> layer_shape{geometry->rows, geometry->cols};
  OutputTensor out_cost;
  OutputTensor out_unknown;
  OutputTensor out_slope;
  OutputTensor out_step;
  OutputTensor out_roughness;
  OutputTensor out_image;
  if (auto made = allocate(out_cost, cost, layer_shape, kFloat32, stream); !made) {
    return made;
  }
  if (auto made = allocate(out_unknown, unknown, layer_shape, kUInt8, stream); !made) {
    return made;
  }
  if (config_.components) {
    for (auto made : {allocate(out_slope, slope, layer_shape, kFloat32, stream),
                      allocate(out_step, step_height, layer_shape, kFloat32, stream),
                      allocate(out_roughness, roughness, layer_shape, kFloat32, stream)}) {
      if (!made) {
        return made;
      }
    }
  }
  const TraversabilityLayers layers{.cost = out_cost.data<float>(),
                                    .unknown = out_unknown.data<std::uint8_t>(),
                                    .slope = out_slope.data<float>(),
                                    .step_height = out_step.data<float>(),
                                    .roughness = out_roughness.data<float>()};
  if (const cudaError_t status =
          compute_traversability(heights, geometry->rows, geometry->cols, geometry->resolution,
                                 config_.traversability, layers, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("traversability", status));
  }
  if (config_.image) {
    const std::array<std::int64_t, 3> image_shape{geometry->rows, geometry->cols, 3};
    if (auto made = allocate(out_image, cost_image, image_shape, kUInt8, stream); !made) {
      return made;
    }
    if (const cudaError_t status =
            colorize_cost(out_cost.data<float>(), geometry->rows, geometry->cols,
                          out_image.data<std::uint8_t>(), stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("colorizing the cost", status));
    }
  }

  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*sample);
  if (auto sent = send(out_cost, cost, "cost", emit); !sent) {
    return sent;
  }
  if (auto sent = send(out_unknown, unknown, "unknown", emit); !sent) {
    return sent;
  }
  if (config_.components) {
    if (auto sent = send(out_slope, slope, "slope", emit); !sent) {
      return sent;
    }
    if (auto sent = send(out_step, step_height, "step_height", emit); !sent) {
      return sent;
    }
    if (auto sent = send(out_roughness, roughness, "roughness", emit); !sent) {
      return sent;
    }
  }
  if (config_.image) {
    return send_image(out_image, cost_image, "cost_image", geometry->rows, geometry->cols,
                      config_.frame_id, emit);
  }
  return {};
}

}  // namespace holoscan::height_map
