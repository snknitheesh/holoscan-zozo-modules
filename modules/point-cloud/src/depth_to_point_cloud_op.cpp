// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/point_cloud/depth_to_point_cloud_op.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/camera_info.hpp>
#include <holoscan/point_cloud/deproject.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "host_input.hpp"
#include "pinhole_camera.hpp"

namespace holoscan::point_cloud {
namespace {

namespace hm = holoscan::messages;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "DepthToPointCloudOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return holoscan::Error{
      holoscan::ErrorCode::kFailure,
      std::string("DepthToPointCloudOp: ") + what + ": " + cudaGetErrorString(status)};
}

bool on_gpu(const holoscan::Tensor& tensor) {
  const DLDevice device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

// Whether the tensor's rows are packed: each element directly after the last.
bool packed(const holoscan::Tensor& tensor, std::size_t element_bytes) {
  const std::vector<std::int64_t> strides = tensor.strides();
  const std::span<const std::int64_t> shape = tensor.shape_span();
  std::int64_t expected = static_cast<std::int64_t>(element_bytes);
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    if (shape[axis] > 1 && strides[axis] != expected) {
      return false;
    }
    expected *= shape[axis];
  }
  return true;
}

// Waits for the producer of an input that only this operator's stream reads.
// The runtime has already ordered device, managed and stream-ready pinned
// memory on that stream before compute(), which is all the GPU needs; plain
// host memory it does not order, so wait for that here. (An input the host
// reads itself needs synchronize_for_host() in every case.)
holoscan::expected<void, holoscan::Error> wait_for_host(const holoscan::Tensor& tensor) {
  if (tensor.device().device_type != kDLCPU) {
    return {};
  }
  return tensor.synchronize_for_host();
}

// Whether a GPU-resident tensor's data can be read in place as elements of
// `alignment` bytes.
bool aligned(const holoscan::Tensor& tensor, std::size_t alignment) {
  return !on_gpu(tensor) || reinterpret_cast<std::uintptr_t>(tensor.data()) % alignment == 0U;
}

}  // namespace

void validate_config(const DepthToPointCloudConfig& config) {
  if (!(std::isfinite(config.depth_scale) && config.depth_scale > 0)) {
    throw std::invalid_argument("depth_scale must be finite and positive");
  }
  if (!(config.depth_min >= 0 && config.depth_max > config.depth_min)) {
    throw std::invalid_argument("depth range needs 0 <= depth_min < depth_max");
  }
  if (config.max_width < 1 || config.max_height < 1 || config.max_width > 16384 ||
      config.max_height > 16384) {
    throw std::invalid_argument("max_width and max_height must be 1..16384");
  }
  if (config.camera_info.max_age <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("camera_info.max_age must be > 0");
  }
}

DepthToPointCloudOp::DepthToPointCloudOp(DepthToPointCloudConfig config)
    : config_(std::move(config)) {
  validate_config(config_);
}

DepthToPointCloudOp::~DepthToPointCloudOp() { release(); }

void DepthToPointCloudOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: one frame in compute() while the next one arrives.
  spec.input(depth, "depth").queue_depth(2U);
  spec.input(camera_info, "camera_info").queue_depth(2U);
  // The color input exists only when configured: an unconnected input would
  // never be satisfied.
  if (config_.color) {
    spec.input(color, "color").queue_depth(2U);
  }
  hm::declare_point_cloud_output(spec, point_cloud, "point_cloud",
                                 {.max_points = static_cast<std::uint64_t>(config_.max_width) *
                                                static_cast<std::uint64_t>(config_.max_height),
                                  .colors = config_.color});
}

holoscan::Contract DepthToPointCloudOp::contract() const {
  // The depth and color of one frame carry the same capture time.
  if (config_.color) {
    return hm::camera_info_contract(config_.camera_info, camera_info, depth, color);
  }
  return hm::camera_info_contract(config_.camera_info, camera_info, depth);
}

// Each run starts without a camera; the SDK may start an operator again.
void DepthToPointCloudOp::start() {
  info_.reset();
  intrinsics_.reset();
}

void DepthToPointCloudOp::stop() { release(); }

void DepthToPointCloudOp::release() noexcept {
  // cudaFree waits for the work on the runtime's streams that may still use
  // the buffers.
  for (Staging* staging : {&depth_staging_, &color_staging_}) {
    if (staging->data != nullptr) {
      cudaFree(staging->data);
    }
    *staging = {};
  }
}

holoscan::expected<const void*, holoscan::Error> DepthToPointCloudOp::on_device(
    const holoscan::Tensor& tensor, std::size_t bytes, Staging& staging, cudaStream_t stream) {
  if (on_gpu(tensor)) {
    return tensor.data();
  }
  if (staging.bytes < bytes) {
    // Free only after the stream is done with the old buffer.
    cudaStreamSynchronize(stream);
    cudaFree(staging.data);
    staging = {};
    if (const cudaError_t status = cudaMalloc(&staging.data, bytes); status != cudaSuccess) {
      staging.data = nullptr;
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    staging.bytes = bytes;
  }
  if (const cudaError_t status =
          cudaMemcpyAsync(staging.data, tensor.data(), bytes, cudaMemcpyHostToDevice, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
  }
  // The runtime orders the release of an input the GPU reads after this
  // operator's stream, but not of one in host memory, so a producer could
  // reuse a pinned buffer while the copy still reads it. Pageable memory needs
  // nothing: cudaMemcpyAsync has staged it by the time it returns.
  if (tensor.device().device_type != kDLCUDAHost) {
    return static_cast<const void*>(staging.data);
  }
  if (auto held = detail::hold_until_done(tensor, stream, "DepthToPointCloudOp"); !held) {
    return holoscan::make_unexpected(std::move(held).error());
  }
  return static_cast<const void*>(staging.data);
}

holoscan::expected<void, holoscan::Error> DepthToPointCloudOp::compute(
    holoscan::ExecutionContext& context) {
  auto depth_sample = depth.receive();
  if (!depth_sample) {
    return holoscan::make_unexpected(std::move(depth_sample).error());
  }
  auto info = info_.update(camera_info);
  if (!info) {
    return holoscan::make_unexpected(std::move(info).error());
  }
  if (*info == nullptr) {
    return {};  // No camera info yet: this frame gives no cloud.
  }
  const hm::CameraInfo& camera = **info;
  if (info_.changed()) {
    intrinsics_.reset();
    auto intrinsics = detail::pinhole_intrinsics(camera, "DepthToPointCloudOp");
    if (!intrinsics) {
      return holoscan::make_unexpected(std::move(intrinsics).error());
    }
    intrinsics_ = *intrinsics;
  }
  if (!depth_sample->data.data) {
    return holoscan::make_unexpected(invalid("depth image has no tensor"));
  }
  // The runtime assigns a stream to an operator with device outputs, and
  // writes their loans on that stream only, so there is no other to use.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        "DepthToPointCloudOp: the runtime gave no CUDA stream for the device output"});
  }

  // The depth image.
  const holoscan::schema::ImageT& depth_image = depth_sample->data;
  const holoscan::Tensor& depth_tensor = *depth_image.data;
  DeprojectParams params;
  std::size_t depth_bytes = 0;
  if (depth_image.encoding == holoscan::schema::ImageEncoding_MONO16) {
    params.depth_type = DepthType::kUint16;
    depth_bytes = sizeof(std::uint16_t);
  } else if (depth_image.encoding == holoscan::schema::ImageEncoding_MONO32F) {
    params.depth_type = DepthType::kFloat32;
    depth_bytes = sizeof(float);
  } else {
    return holoscan::make_unexpected(invalid(
        std::string("unsupported depth encoding ") +
        holoscan::schema::EnumNameImageEncoding(depth_image.encoding) + "; use MONO16 or MONO32F"));
  }
  const DLDataType dtype = depth_tensor.dtype();
  const bool dtype_ok = params.depth_type == DepthType::kUint16
                            ? dtype.code == kDLUInt && dtype.bits == 16 && dtype.lanes == 1
                            : dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1;
  const std::span<const std::int64_t> shape = depth_tensor.shape_span();
  if (!dtype_ok) {
    return holoscan::make_unexpected(
        invalid("depth tensor element type disagrees with its encoding"));
  }
  if (!(shape.size() == 2 || (shape.size() == 3 && shape[2] == 1))) {
    return holoscan::make_unexpected(invalid("depth tensor must be [H, W] or [H, W, 1]"));
  }
  const int height = static_cast<int>(shape[0]);
  const int width = static_cast<int>(shape[1]);
  if (width < 1 || height < 1 || width > config_.max_width || height > config_.max_height) {
    return holoscan::make_unexpected(invalid("depth image " + std::to_string(width) + "x" +
                                             std::to_string(height) + " exceeds the configured " +
                                             std::to_string(config_.max_width) + "x" +
                                             std::to_string(config_.max_height)));
  }
  if (std::cmp_not_equal(width, camera.width) || std::cmp_not_equal(height, camera.height)) {
    return holoscan::make_unexpected(
        invalid("the depth image is " + std::to_string(width) + "x" + std::to_string(height) +
                " but camera info '" + camera.header.frame_id + "' is for " +
                std::to_string(camera.width) + "x" + std::to_string(camera.height) + " images"));
  }
  if (!packed(depth_tensor, depth_bytes)) {
    return holoscan::make_unexpected(invalid("depth rows must be packed"));
  }
  if (!aligned(depth_tensor, depth_bytes)) {
    return holoscan::make_unexpected(invalid("depth data must be aligned to its element size"));
  }
  if (auto ready = wait_for_host(depth_tensor); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  params.intrinsics = *intrinsics_;
  params.depth_scale = config_.depth_scale;
  params.depth_min = config_.depth_min;
  params.depth_max = config_.depth_max;
  // The PointCloud message marks an invalid point with NaN.
  params.invalid_value = std::numeric_limits<float>::quiet_NaN();

  // The color image. The sample stays alive until compute() returns, as the
  // kernel may read it in place.
  ColorInput color_input;
  std::optional<holoscan::Sample<holoscan::schema::ImageT>> color_sample;
  if (config_.color) {
    auto sample = color.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    color_sample.emplace(std::move(*sample));
    const holoscan::schema::ImageT& image = color_sample->data;
    if (!image.data) {
      return holoscan::make_unexpected(invalid("color image has no tensor"));
    }
    switch (image.encoding) {
      case holoscan::schema::ImageEncoding_RGB8:
        color_input = {nullptr, 3, false};
        break;
      case holoscan::schema::ImageEncoding_BGR8:
        color_input = {nullptr, 3, true};
        break;
      case holoscan::schema::ImageEncoding_RGBA8:
        color_input = {nullptr, 4, false};
        break;
      case holoscan::schema::ImageEncoding_BGRA8:
        color_input = {nullptr, 4, true};
        break;
      default:
        return holoscan::make_unexpected(
            invalid(std::string("unsupported color encoding ") +
                    holoscan::schema::EnumNameImageEncoding(image.encoding) +
                    "; use RGB8, BGR8, RGBA8 or BGRA8"));
    }
    const holoscan::Tensor& tensor = *image.data;
    const DLDataType cdt = tensor.dtype();
    const std::span<const std::int64_t> cshape = tensor.shape_span();
    if (cdt.code != kDLUInt || cdt.bits != 8 || cshape.size() != 3 ||
        cshape[2] != color_input.channels) {
      return holoscan::make_unexpected(
          invalid("color tensor must be uint8 [H, W, channels] matching its encoding"));
    }
    if (cshape[0] != height || cshape[1] != width) {
      return holoscan::make_unexpected(invalid("color image size must match the depth image"));
    }
    if (!packed(tensor, 1)) {
      return holoscan::make_unexpected(invalid("color rows must be packed"));
    }
    // Four channels are read as one uchar4.
    if (!aligned(tensor, color_input.channels == 4 ? 4U : 1U)) {
      return holoscan::make_unexpected(
          invalid("color data must be aligned to 4 bytes for four channels"));
    }
    if (auto ready = wait_for_host(tensor); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    auto data = on_device(tensor, pixels * static_cast<std::size_t>(color_input.channels),
                          color_staging_, stream);
    if (!data) {
      return holoscan::make_unexpected(std::move(data).error());
    }
    color_input.data = *data;
  }
  auto depth_data = on_device(depth_tensor, pixels * depth_bytes, depth_staging_, stream);
  if (!depth_data) {
    return holoscan::make_unexpected(std::move(depth_data).error());
  }

  // The cloud: one kernel writes the points and the colors, so the colors'
  // write runs inside the points'.
  const hm::PointCloudShape cloud_shape = hm::organized_cloud(height, width);
  auto cloud = hm::allocate_point_cloud(point_cloud, cloud_shape, {.colors = config_.color},
                                        context.cuda_stream());
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  const auto deproject = [&](float3* xyz,
                             uchar3* rgb) -> holoscan::expected<void, holoscan::Error> {
    if (const cudaError_t status =
            launch_deproject(*depth_data, params, color_input, xyz, rgb, width, height, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("deproject kernel", status));
    }
    return {};
  };
  auto written = cloud->write_and_commit(hm::PointCloudFields::points, context.cuda_stream(),
                                         [&](const holoscan::TensorOutputWriteGuard& points)
                                             -> holoscan::expected<void, holoscan::Error> {
                                           if (!config_.color) {
                                             return deproject(points.data_as<float3>(), nullptr);
                                           }
                                           return cloud->write_and_commit(
                                               hm::PointCloudFields::colors, context.cuda_stream(),
                                               [&](const holoscan::TensorOutputWriteGuard& colors) {
                                                 return deproject(points.data_as<float3>(),
                                                                  colors.data_as<uchar3>());
                                               });
                                         });
  if (!written) {
    return written;
  }

  hm::Header header{.frame_id = camera.header.frame_id};
  if (depth_sample->metadata.source_clock_id.valid()) {
    header.capture_timestamp_ns = depth_sample->metadata.capture_timestamp_ns;
  }
  if (depth_image.header) {
    header.device_sequence = depth_image.header->device_sequence;
  }
  // Counting the valid points would wait for the kernel: -1, not counted.
  return hm::emit_point_cloud(point_cloud, std::move(*cloud), header, cloud_shape, -1,
                              holoscan::sensor_io::derive_from(*depth_sample));
}

}  // namespace holoscan::point_cloud
