// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/tsdf/tsdf_integrate_op.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/sensor_io/provenance.hpp>

namespace holoscan::tsdf {
namespace {

using std::chrono_literals::operator""ns;

bool valid_intrinsics(const CameraIntrinsics& k) {
  return std::isfinite(k.fx) && k.fx > 0.0F && std::isfinite(k.fy) && k.fy > 0.0F &&
         std::isfinite(k.cx) && std::isfinite(k.cy);
}

}  // namespace

void validate_config(const TsdfIntegrateConfig& config) {
  if (!config.volume) {
    throw std::invalid_argument("volume is required");
  }
  if (config.intrinsics) {
    if (!valid_intrinsics(*config.intrinsics)) {
      throw std::invalid_argument("intrinsics need finite, positive fx and fy and finite cx, cy");
    }
    if (config.intrinsics_input) {
      throw std::invalid_argument("give intrinsics or enable the intrinsics input, not both");
    }
  } else if (!config.intrinsics_input) {
    throw std::invalid_argument("give intrinsics or enable the intrinsics input");
  }
  if (!(std::isfinite(config.depth_scale) && config.depth_scale > 0.0F)) {
    throw std::invalid_argument("depth_scale must be finite and positive");
  }
  if (!(std::isfinite(config.depth_min) && std::isfinite(config.depth_max) &&
        config.depth_min >= 0.0F && config.depth_max > config.depth_min)) {
    throw std::invalid_argument("the depth range needs 0 <= depth_min < depth_max, finite");
  }
  if (config.color && !config.volume->config().color) {
    throw std::invalid_argument("color needs a volume with color");
  }
  if (config.max_width < 1 || config.max_height < 1 || config.max_width > 8192 ||
      config.max_height > 8192) {
    throw std::invalid_argument("max_width and max_height must be 1..8192");
  }
}

TsdfIntegrateOp::TsdfIntegrateOp(TsdfIntegrateConfig config)
    : VolumeOp("TsdfIntegrateOp", config.volume), config_(std::move(config)) {
  validate_config(config_);
}

TsdfIntegrateOp::~TsdfIntegrateOp() = default;

void TsdfIntegrateOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: one frame in compute() while the next one arrives.
  spec.input(depth, "depth").queue_depth(2U);
  // Optional ports exist only when configured: an unconnected input would
  // never be satisfied, and an emit into an unconnected output fails.
  if (config_.color) {
    spec.input(color, "color").queue_depth(2U);
  }
  spec.input(pose, "pose").queue_depth(2U);
  if (config_.intrinsics_input) {
    spec.input(intrinsics, "intrinsics").queue_depth(2U);
  }
  if (config_.pose_output) {
    spec.output(integrated_pose, "integrated_pose")
        .produces_tensor(host_output(kFloat32, 16U * sizeof(float)));
  }
}

holoscan::Contract TsdfIntegrateOp::contract() const {
  // Every input of one frame carries the same capture time.
  std::vector<holoscan::PortRef> ports{depth, pose};
  if (config_.color) {
    ports.emplace_back(color);
  }
  if (config_.intrinsics_input) {
    ports.emplace_back(intrinsics);
  }
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
  result.input(depth, holoscan::Exact{});
  result.input(pose, holoscan::Exact{});
  if (config_.color) {
    result.input(color, holoscan::Exact{});
  }
  if (config_.intrinsics_input) {
    result.input(intrinsics, holoscan::Exact{});
  }
  return result;
}

holoscan::expected<void, holoscan::Error> TsdfIntegrateOp::compute(
    holoscan::ExecutionContext& context) {
  auto depth_sample = depth.receive();
  if (!depth_sample) {
    return holoscan::make_unexpected(std::move(depth_sample).error());
  }
  auto pose_sample = pose.receive();
  if (!pose_sample) {
    return holoscan::make_unexpected(std::move(pose_sample).error());
  }
  std::optional<holoscan::Sample<holoscan::schema::ImageT>> color_sample;
  if (config_.color) {
    auto sample = color.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    color_sample.emplace(std::move(*sample));
  }
  std::optional<holoscan::Sample<holoscan::Tensor>> intrinsics_sample;
  if (config_.intrinsics_input) {
    auto sample = intrinsics.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    intrinsics_sample.emplace(std::move(*sample));
  }
  auto stream = stream_for(context);
  if (!stream) {
    return holoscan::make_unexpected(std::move(stream).error());
  }

  // The depth image.
  const holoscan::schema::ImageT& depth_image = depth_sample->data;
  if (!depth_image.data) {
    return holoscan::make_unexpected(invalid("depth image has no tensor"));
  }
  const holoscan::Tensor& depth_tensor = *depth_image.data;
  DepthImage image;
  std::size_t element = 0;
  if (depth_image.encoding == holoscan::schema::ImageEncoding_MONO16) {
    image.type = DepthType::kUint16;
    element = sizeof(std::uint16_t);
  } else if (depth_image.encoding == holoscan::schema::ImageEncoding_MONO32F) {
    image.type = DepthType::kFloat32;
    element = sizeof(float);
  } else {
    return holoscan::make_unexpected(invalid(
        std::string("unsupported depth encoding ") +
        holoscan::schema::EnumNameImageEncoding(depth_image.encoding) + "; use MONO16 or MONO32F"));
  }
  const std::span<const std::int64_t> shape = depth_tensor.shape_span();
  if (!is_dtype(depth_tensor.dtype(), image.type == DepthType::kUint16 ? kUInt16 : kFloat32)) {
    return holoscan::make_unexpected(
        invalid("depth tensor element type disagrees with its encoding"));
  }
  if (!(shape.size() == 2 || (shape.size() == 3 && shape[2] == 1))) {
    return holoscan::make_unexpected(invalid("depth tensor must be [H, W] or [H, W, 1]"));
  }
  if (shape[0] < 0 || shape[1] < 0 || shape[0] > config_.max_height ||
      shape[1] > config_.max_width) {
    return holoscan::make_unexpected(invalid("depth image " + std::to_string(shape[1]) + "x" +
                                             std::to_string(shape[0]) + " exceeds the configured " +
                                             std::to_string(config_.max_width) + "x" +
                                             std::to_string(config_.max_height)));
  }
  image.height = static_cast<int>(shape[0]);
  image.width = static_cast<int>(shape[1]);
  image.scale = config_.depth_scale;
  if (!packed(depth_tensor)) {
    return holoscan::make_unexpected(invalid("depth rows must be packed"));
  }
  const std::size_t pixels =
      static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height);

  // The pose and the intrinsics.
  auto camera_pose = read_pose(pose_sample->data, "pose", *stream);
  if (!camera_pose) {
    return holoscan::make_unexpected(std::move(camera_pose).error());
  }
  IntegrateParams params;
  params.depth_min = config_.depth_min;
  params.depth_max = config_.depth_max;
  if (config_.intrinsics_input) {
    const holoscan::Tensor& tensor = intrinsics_sample->data;
    if (!is_dtype(tensor.dtype(), kFloat32) || tensor.size() != 4 || !packed(tensor)) {
      return holoscan::make_unexpected(invalid("intrinsics must be float32 [fx, fy, cx, cy]"));
    }
    std::array<float, 4> k{};
    if (tensor.device().device_type == kDLCUDA || tensor.device().device_type == kDLCUDAManaged) {
      if (auto copied = tensor.copy_to_host(k.data(), sizeof(k), *stream); !copied) {
        return holoscan::make_unexpected(std::move(copied).error());
      }
    } else {
      if (auto ready = tensor.synchronize_for_host(); !ready) {
        return holoscan::make_unexpected(std::move(ready).error());
      }
      std::memcpy(k.data(), tensor.data(), sizeof(k));
    }
    params.intrinsics = {k[0], k[1], k[2], k[3]};
    if (!valid_intrinsics(params.intrinsics)) {
      return holoscan::make_unexpected(
          invalid("intrinsics need finite, positive fx and fy and finite cx, cy"));
    }
  } else {
    params.intrinsics = *config_.intrinsics;
  }

  // The color image.
  ColorImage color_image;
  if (config_.color) {
    const holoscan::schema::ImageT& source = color_sample->data;
    if (!source.data) {
      return holoscan::make_unexpected(invalid("color image has no tensor"));
    }
    switch (source.encoding) {
      case holoscan::schema::ImageEncoding_RGB8:
        color_image = {nullptr, 3, false};
        break;
      case holoscan::schema::ImageEncoding_BGR8:
        color_image = {nullptr, 3, true};
        break;
      case holoscan::schema::ImageEncoding_RGBA8:
        color_image = {nullptr, 4, false};
        break;
      case holoscan::schema::ImageEncoding_BGRA8:
        color_image = {nullptr, 4, true};
        break;
      default:
        return holoscan::make_unexpected(
            invalid(std::string("unsupported color encoding ") +
                    holoscan::schema::EnumNameImageEncoding(source.encoding) +
                    "; use RGB8, BGR8, RGBA8 or BGRA8"));
    }
    const holoscan::Tensor& tensor = *source.data;
    const std::span<const std::int64_t> cshape = tensor.shape_span();
    if (!is_dtype(tensor.dtype(), kUInt8) || cshape.size() != 3 ||
        cshape[2] != color_image.channels) {
      return holoscan::make_unexpected(
          invalid("color tensor must be uint8 [H, W, channels] matching its encoding"));
    }
    if (cshape[0] != image.height || cshape[1] != image.width) {
      return holoscan::make_unexpected(invalid("color image size must match the depth image"));
    }
    if (!packed(tensor)) {
      return holoscan::make_unexpected(invalid("color rows must be packed"));
    }
    auto data =
        on_device(tensor, pixels * static_cast<std::size_t>(color_image.channels), 1, *stream);
    if (!data) {
      return holoscan::make_unexpected(std::move(data).error());
    }
    color_image.data = static_cast<const std::uint8_t*>(*data);
  }
  auto depth_data = on_device(depth_tensor, pixels * element, 0, *stream);
  if (!depth_data) {
    return holoscan::make_unexpected(std::move(depth_data).error());
  }
  image.data = *depth_data;
  if (reinterpret_cast<std::uintptr_t>(image.data) % element != 0U) {
    return holoscan::make_unexpected(invalid("depth data must be aligned to its element size"));
  }

  if (const cudaError_t status = volume()->integrate(image, config_.color ? &color_image : nullptr,
                                                     *camera_pose, params, *stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("integration", status));
  }

  if (!config_.pose_output) {
    return {};
  }
  auto loan = integrated_pose.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = std::array<std::int64_t, 2>{4, 4}, .dtype = kFloat32});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write_host();
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  std::memcpy(writer->data_as<float>(), camera_pose->data(), sizeof(Matrix4));
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  if (auto sent =
          integrated_pose.emit(std::move(*loan), holoscan::sensor_io::derive_from(*depth_sample));
      !sent) {
    return emit_failure("integrated_pose", std::move(sent).error());
  }
  return {};
}

}  // namespace holoscan::tsdf
