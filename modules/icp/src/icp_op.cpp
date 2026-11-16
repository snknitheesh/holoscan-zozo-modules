// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/icp/icp_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/messages/camera_info.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace holoscan::icp {
namespace {

namespace hm = holoscan::messages;

using std::chrono_literals::operator""ns;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kFloat64{kDLFloat, 64U, 1U};

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "IcpOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         std::string("IcpOp: ") + what + ": " + cudaGetErrorString(status)};
}

holoscan::TensorOutputSpec host_tensor(std::uint32_t rank, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = kFloat64,
              .rank = rank,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
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

// Whether the tensor is packed: each element directly after the last, any
// stride on an axis of extent 1.
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

// The points of a cloud that check_point_cloud() accepted: N, or H * W.
std::uint64_t point_count(const holoscan::Tensor& points) {
  const std::span<const std::int64_t> shape = points.shape_span();
  std::uint64_t count = 1;
  for (std::size_t axis = 0; axis + 1 < shape.size(); ++axis) {
    count *= static_cast<std::uint64_t>(shape[axis]);
  }
  return count;
}

// Whether every rotation row is a unit vector orthogonal to the others, and
// the rotation is proper: a rigid transform within float32 rounding.
bool rigid(const Matrix4& m) {
  if (!is_rigid_shaped(m)) {
    return false;
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const double dot =
          m[4 * a] * m[4 * b] + m[4 * a + 1] * m[4 * b + 1] + m[4 * a + 2] * m[4 * b + 2];
      if (std::fabs(dot - (a == b ? 1.0 : 0.0)) > 1e-4) {
        return false;
      }
    }
  }
  const double det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
                     m[2] * (m[4] * m[9] - m[5] * m[8]);
  return det > 0.0;
}

}  // namespace

void validate_config(const IcpConfig& config) {
  validate_options(config.options);
  if (config.mode != IcpMode::kPairs && config.mode != IcpMode::kMap &&
      config.mode != IcpMode::kOdometry) {
    throw std::invalid_argument("mode must be pairs, map or odometry");
  }
  if (config.max_points < 1U || config.max_points > kMaxPointsLimit) {
    throw std::invalid_argument("max_points must be 1.." + std::to_string(kMaxPointsLimit));
  }
  if (config.map_max_age <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("map_max_age must be positive");
  }
  if (!config.transform_output && !config.stats_output && !config.aligned_output &&
      !(config.mode == IcpMode::kOdometry && config.pose_output)) {
    throw std::invalid_argument("enable at least one output");
  }
}

IcpResult result_from(std::span<const double> stats, const Matrix4& transform) {
  if (stats.size() != kStatsSize) {
    throw std::invalid_argument("stats must hold " + std::to_string(kStatsSize) + " values");
  }
  const double status = stats[kStatStatus];
  if (!(status >= 0.0 && status <= static_cast<double>(IcpStatus::kDegenerate)) ||
      status != std::floor(status)) {
    throw std::invalid_argument("stats hold an unknown status");
  }
  IcpResult result;
  result.transform = transform;
  result.fitness = stats[kStatFitness];
  result.rmse = stats[kStatRmse];
  result.iterations = static_cast<int>(stats[kStatIterations]);
  result.converged = stats[kStatConverged] != 0.0;
  result.status = static_cast<IcpStatus>(static_cast<int>(status));
  result.correspondences = static_cast<std::uint32_t>(stats[kStatCorrespondences]);
  result.source_points = static_cast<std::uint32_t>(stats[kStatSourcePoints]);
  result.target_points = static_cast<std::uint32_t>(stats[kStatTargetPoints]);
  return result;
}

IcpOp::IcpOp(IcpConfig config) : config_(std::move(config)) { validate_config(config_); }

IcpOp::~IcpOp() { release(); }

void IcpOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: one cloud in compute() while the next one arrives.
  spec.input(source, "source").queue_depth(2U);
  if (config_.mode != IcpMode::kOdometry) {
    spec.input(target, "target").queue_depth(2U);
  }
  if (config_.initial_guess_input) {
    spec.input(initial_guess, "initial_guess").queue_depth(2U);
  }
  // Optional ports exist only when configured: an emit into an unconnected
  // output fails.
  if (config_.transform_output) {
    spec.output(transform, "transform").produces_tensor(host_tensor(2U, 16U * sizeof(double)));
  }
  if (config_.stats_output) {
    spec.output(stats, "stats").produces_tensor(host_tensor(1U, kStatsSize * sizeof(double)));
  }
  if (config_.mode == IcpMode::kOdometry && config_.pose_output) {
    spec.output(pose, "pose").produces_tensor(host_tensor(2U, 16U * sizeof(double)));
  }
  if (config_.aligned_output) {
    hm::declare_point_cloud_output(spec, aligned, "aligned",
                                   {.max_points = config_.max_points,
                                    .colors = config_.aligned_fields.colors,
                                    .normals = config_.aligned_fields.normals,
                                    .intensity = config_.aligned_fields.intensity});
  }
}

holoscan::Contract IcpOp::contract() const {
  holoscan::Contract result;
  // The inputs of one registration carry the source's capture time.
  std::vector<holoscan::PortRef> aligned_ports{source};
  if (config_.mode == IcpMode::kPairs) {
    aligned_ports.emplace_back(target);
  }
  if (config_.initial_guess_input) {
    aligned_ports.emplace_back(initial_guess);
  }
  if (aligned_ports.size() == 1U) {
    result.trigger(holoscan::OnEach{source});
  } else {
    result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{aligned_ports}, .window = 0ns});
    result.input(source, holoscan::Exact{});
    if (config_.mode == IcpMode::kPairs) {
      result.input(target, holoscan::Exact{});
    }
    if (config_.initial_guess_input) {
      result.input(initial_guess, holoscan::Exact{});
    }
  }
  if (config_.mode == IcpMode::kMap) {
    // The newest target captured at or before the source, when there is one
    // within map_max_age; otherwise the stored one.
    result.input(target, holoscan::LatestWithin{.max_age = config_.map_max_age});
    result.on_missing(target, holoscan::SafeDefault{});
  }
  return result;
}

void IcpOp::start() {
  if (config_.mode == IcpMode::kOdometry) {
    odometry_.emplace(config_.options, config_.constant_velocity);
  } else {
    registration_.emplace(config_.options);
  }
  stored_target_time_.reset();
  stored_target_frame_ = 0;
  stored_target_data_ = nullptr;
  target_frame_id_.clear();
  staging_pending_ = false;
}

void IcpOp::stop() { release(); }

void IcpOp::release() noexcept {
  if (own_stream_ != nullptr) {
    cudaStreamSynchronize(own_stream_);
  }
  // set_target() and promote_source_to_target() only queue work, which may
  // still run on the runtime's stream when the run stops: cudaFree() waits
  // for the device to finish it before freeing. compute() itself waits for
  // its stream before it returns an error with work queued.
  registration_.reset();
  odometry_.reset();
  for (Staging* staging : {&source_staging_, &target_staging_, &colors_staging_, &normals_staging_,
                           &intensity_staging_}) {
    if (staging->data != nullptr) {
      cudaFree(staging->data);
    }
    *staging = {};
  }
  if (staging_free_ != nullptr) {
    cudaEventDestroy(staging_free_);
    staging_free_ = nullptr;
  }
  staging_pending_ = false;
  if (own_stream_ != nullptr) {
    cudaStreamDestroy(own_stream_);
    own_stream_ = nullptr;
  }
}

holoscan::expected<cudaStream_t, holoscan::Error> IcpOp::stream_for(
    holoscan::ExecutionContext& context) {
  cudaStream_t stream = context.cuda_stream().get();
  if (stream != nullptr) {
    return stream;
  }
  // Without a runtime stream (no device output), never the legacy default
  // stream, which serializes with every other stream in the process.
  if (own_stream_ == nullptr) {
    if (const cudaError_t status = cudaStreamCreateWithFlags(&own_stream_, cudaStreamNonBlocking);
        status != cudaSuccess) {
      own_stream_ = nullptr;
      return holoscan::make_unexpected(cuda_failure("cudaStreamCreate", status));
    }
  }
  return own_stream_;
}

holoscan::expected<const void*, holoscan::Error> IcpOp::on_device(const holoscan::Tensor& tensor,
                                                                  std::size_t bytes,
                                                                  std::size_t alignment,
                                                                  Staging& staging,
                                                                  cudaStream_t stream) {
  if (on_gpu(tensor)) {
    if (reinterpret_cast<std::uintptr_t>(tensor.data()) % alignment != 0U) {
      return holoscan::make_unexpected(
          invalid("cloud data must be aligned to " + std::to_string(alignment) + " bytes"));
    }
    int device = 0;
    if (const cudaError_t status = cudaStreamGetDevice(stream, &device); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamGetDevice", status));
    }
    if (tensor.device().device_type == kDLCUDA && tensor.device().device_id != device) {
      return holoscan::make_unexpected(invalid("a cloud is on another GPU than the operator"));
    }
    // The runtime orders device input on its own stream only; on the
    // operator's own, the GPU waits for the input's producer.
    if (stream == own_stream_) {
      if (auto ready = tensor.synchronize_for_consumer_stream(stream); !ready) {
        return holoscan::make_unexpected(std::move(ready).error());
      }
    }
    return static_cast<const void*>(tensor.data());
  }
  // Host memory: wait for its producer, then copy it. The registration waits
  // for the stream before it returns (see registration.hpp), so the copy is
  // done before compute() returns and the input is released.
  if (auto ready = tensor.synchronize_for_host(); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  if (!logged_host_copy_) {
    logged_host_copy_ = true;
    HOLOSCAN_LOG_INFO(
        "IcpOp: clouds arrive in host memory and are copied to the GPU; send them in device "
        "memory to avoid the copy");
  }
  // The `aligned` kernels of the frame before may still read the staging
  // buffers: the copy waits for them on the GPU.
  if (staging_pending_) {
    if (const cudaError_t status = cudaStreamWaitEvent(stream, staging_free_);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamWaitEvent", status));
    }
  }
  if (staging.bytes < bytes) {
    // Rare: only when a cloud is larger than every one before it.
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
    }
    cudaFree(staging.data);
    staging = {};
    const std::size_t grown = std::max(bytes, std::size_t{1});
    if (const cudaError_t status = cudaMalloc(&staging.data, grown); status != cudaSuccess) {
      staging.data = nullptr;
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    staging.bytes = grown;
  }
  if (bytes != 0U) {
    if (const cudaError_t status =
            cudaMemcpyAsync(staging.data, tensor.data(), bytes, cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
    }
  }
  return static_cast<const void*>(staging.data);
}

holoscan::expected<IcpOp::DeviceCloud, holoscan::Error> IcpOp::cloud_on_device(
    const hm::schema::PointCloudT& cloud, const char* name, bool fields, Staging& staging,
    cudaStream_t stream) {
  if (auto checked = hm::check_point_cloud(cloud); !checked) {
    return holoscan::make_unexpected(
        invalid(std::string(name) + ": " + std::move(checked).error().message));
  }
  const holoscan::Tensor& points = *cloud.points;
  const std::uint64_t count = point_count(points);
  if (count > config_.max_points) {
    return holoscan::make_unexpected(invalid(std::string(name) + " has " + std::to_string(count) +
                                             " points, more than the configured maximum of " +
                                             std::to_string(config_.max_points)));
  }
  DeviceCloud out;
  out.count = static_cast<std::uint32_t>(count);
  out.staged = !on_gpu(points);
  auto data = on_device(points, count * 3U * sizeof(float), sizeof(float), staging, stream);
  if (!data) {
    return holoscan::make_unexpected(std::move(data).error());
  }
  out.points = static_cast<const float*>(*data);
  if (!fields) {
    return out;
  }
  // The optional fields the aligned cloud carries; check_point_cloud() found
  // them in the points' memory, of their shape and type.
  const auto field =
      [&](const std::shared_ptr<holoscan::Tensor>& tensor, bool allowed, const char* what,
          std::size_t element_bytes, std::size_t alignment,
          Staging& field_staging) -> holoscan::expected<const void*, holoscan::Error> {
    if (!hm::is_present(tensor)) {
      return static_cast<const void*>(nullptr);
    }
    if (!allowed) {
      return holoscan::make_unexpected(invalid(std::string(name) + " carries " + what +
                                               ", which aligned_fields leaves out of the "
                                               "aligned output"));
    }
    return on_device(*tensor, count * element_bytes, alignment, field_staging, stream);
  };
  auto colors =
      field(cloud.colors, config_.aligned_fields.colors, "colors", 3U, 1U, colors_staging_);
  if (!colors) {
    return holoscan::make_unexpected(std::move(colors).error());
  }
  auto normals = field(cloud.normals, config_.aligned_fields.normals, "normals", 3U * sizeof(float),
                       sizeof(float), normals_staging_);
  if (!normals) {
    return holoscan::make_unexpected(std::move(normals).error());
  }
  auto intensity = field(cloud.intensity, config_.aligned_fields.intensity, "intensity",
                         sizeof(float), sizeof(float), intensity_staging_);
  if (!intensity) {
    return holoscan::make_unexpected(std::move(intensity).error());
  }
  out.colors = *colors;
  out.normals = static_cast<const float*>(*normals);
  out.intensity = static_cast<const float*>(*intensity);
  return out;
}

holoscan::expected<void, holoscan::Error> IcpOp::emit_aligned(
    const hm::schema::PointCloudT& cloud, const DeviceCloud& in, const Matrix4& moved,
    const std::string& frame_id, const holoscan::EmitOptions& emit, cudaStream_t stream) {
  const std::span<const std::int64_t> shape = cloud.points->shape_span();
  const hm::PointCloudShape cloud_shape =
      cloud.organized ? hm::organized_cloud(shape[0], shape[1]) : hm::unorganized_cloud(shape[0]);
  const hm::PointCloudContents contents{.colors = in.colors != nullptr,
                                        .normals = in.normals != nullptr,
                                        .intensity = in.intensity != nullptr};
  auto loan = hm::allocate_point_cloud(aligned, cloud_shape, contents, stream);
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  const auto kernel = [](const char* what,
                         cudaError_t status) -> holoscan::expected<void, holoscan::Error> {
    if (status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure(what, status));
    }
    return {};
  };
  if (auto written = loan->write_and_commit(
          hm::PointCloudFields::points, stream,
          [&](const holoscan::TensorOutputWriteGuard& out) {
            return kernel(
                "transform kernel",
                launch_transform_points(in.points, out.data_as<float>(), in.count, moved, stream));
          });
      !written) {
    return written;
  }
  if (in.normals != nullptr) {
    if (auto written = loan->write_and_commit(
            hm::PointCloudFields::normals, stream,
            [&](const holoscan::TensorOutputWriteGuard& out) {
              return kernel("rotate kernel", launch_rotate_vectors(in.normals, out.data_as<float>(),
                                                                   in.count, moved, stream));
            });
        !written) {
      return written;
    }
  }
  // Colors and intensities do not change with the pose.
  const auto copy = [&](const holoscan::TensorFieldSelector<hm::schema::PointCloudT>& field,
                        const void* from, std::size_t bytes) {
    return loan->write_and_commit(field, stream, [&](const holoscan::TensorOutputWriteGuard& out) {
      return kernel(
          "cudaMemcpyAsync",
          bytes == 0U ? cudaSuccess
                      : cudaMemcpyAsync(out.data(), from, bytes, cudaMemcpyDeviceToDevice, stream));
    });
  };
  if (in.colors != nullptr) {
    if (auto written = copy(hm::PointCloudFields::colors, in.colors, in.count * 3U); !written) {
      return written;
    }
  }
  if (in.intensity != nullptr) {
    if (auto written =
            copy(hm::PointCloudFields::intensity, in.intensity, in.count * sizeof(float));
        !written) {
      return written;
    }
  }
  // The kernels read the staging buffers after compute() returns: the next
  // copy into them waits for this event on the GPU.
  if (in.staged) {
    if (staging_free_ == nullptr) {
      if (const cudaError_t status =
              cudaEventCreateWithFlags(&staging_free_, cudaEventDisableTiming);
          status != cudaSuccess) {
        staging_free_ = nullptr;
        return holoscan::make_unexpected(cuda_failure("cudaEventCreate", status));
      }
    }
    if (const cudaError_t status = cudaEventRecord(staging_free_, stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaEventRecord", status));
    }
    staging_pending_ = true;
  }
  hm::Header header{.frame_id = frame_id};
  if (cloud.header) {
    header.capture_timestamp_ns = cloud.header->capture_timestamp_ns;
    header.device_sequence = cloud.header->device_sequence;
  }
  return hm::emit_point_cloud(aligned, std::move(*loan), header, cloud_shape, cloud.valid_count,
                              emit);
}

holoscan::expected<void, holoscan::Error> IcpOp::compute(holoscan::ExecutionContext& context) {
  auto source_sample = source.receive();
  if (!source_sample) {
    return holoscan::make_unexpected(std::move(source_sample).error());
  }
  auto stream_or = stream_for(context);
  if (!stream_or) {
    return holoscan::make_unexpected(std::move(stream_or).error());
  }
  cudaStream_t stream = *stream_or;
  // An error after work was queued: wait for it first, so that nothing still
  // reads an input or a buffer when the run stops.
  const auto failed = [stream](holoscan::Error error) -> holoscan::expected<void, holoscan::Error> {
    cudaStreamSynchronize(stream);
    return holoscan::make_unexpected(std::move(error));
  };

  // The source.
  const hm::schema::PointCloudT& source_cloud = source_sample->data;
  auto source_points =
      cloud_on_device(source_cloud, "source", config_.aligned_output, source_staging_, stream);
  if (!source_points) {
    return failed(std::move(source_points).error());
  }
  const std::uint32_t count = source_points->count;
  const std::string source_frame_id =
      source_cloud.header ? source_cloud.header->frame_id : std::string{};

  // The initial guess.
  std::optional<Matrix4> guess;
  if (config_.initial_guess_input) {
    auto sample = initial_guess.receive();
    if (!sample) {
      return failed(std::move(sample).error());
    }
    const holoscan::Tensor& tensor = sample->data;
    const std::span<const std::int64_t> shape = tensor.shape_span();
    const bool f32 = is_dtype(tensor.dtype(), kFloat32);
    if (!(f32 || is_dtype(tensor.dtype(), kFloat64)) || shape.size() != 2 || shape[0] != 4 ||
        shape[1] != 4 || !packed(tensor)) {
      return failed(invalid("initial_guess must be float32 or float64 [4, 4]"));
    }
    if (on_gpu(tensor) && !logged_guess_copy_) {
      logged_guess_copy_ = true;
      HOLOSCAN_LOG_INFO(
          "IcpOp: initial guesses arrive in device memory and are copied to the host, waiting "
          "for the GPU, each frame; send them in host memory to avoid the copy");
    }
    std::array<std::byte, 16U * sizeof(double)> bytes{};
    const std::size_t size = 16U * (f32 ? sizeof(float) : sizeof(double));
    if (auto copied = tensor.copy_to_host(bytes.data(), size, stream); !copied) {
      return failed(std::move(copied).error());
    }
    Matrix4 m{};
    for (std::size_t i = 0; i < 16U; ++i) {
      if (f32) {
        float value = 0.0F;
        std::memcpy(&value, bytes.data() + i * sizeof(float), sizeof(float));
        m[i] = value;
      } else {
        std::memcpy(&m[i], bytes.data() + i * sizeof(double), sizeof(double));
      }
    }
    if (!rigid(m)) {
      return failed(invalid("initial_guess is not a rigid transform"));
    }
    guess = m;
  }

  // The registration.
  IcpResult result;
  Matrix4 pose_out = identity_transform();
  // The frame the aligned cloud is in: the target's.
  std::string aligned_frame_id;
  if (config_.mode == IcpMode::kOdometry) {
    OdometryStep step;
    if (const cudaError_t status =
            odometry_->track(source_points->points, count, stream, step, guess ? &*guess : nullptr);
        status != cudaSuccess) {
      return failed(cuda_failure("odometry", status));
    }
    result = step.result;
    pose_out = step.pose;
    aligned_frame_id = step.first ? source_frame_id : target_frame_id_;
    target_frame_id_ = source_frame_id;
  } else {
    // The target: this source's own in kPairs; in kMap a new one when it
    // came, otherwise the stored one.
    if (config_.mode == IcpMode::kPairs || !target.is_missing()) {
      auto target_sample = target.receive();
      if (!target_sample) {
        return failed(std::move(target_sample).error());
      }
      const holoscan::SampleMetadata& metadata = target_sample->metadata;
      const hm::schema::PointCloudT& target_cloud = target_sample->data;
      const void* target_data = target_cloud.points ? target_cloud.points->data() : nullptr;
      const bool stored =
          config_.mode == IcpMode::kMap && stored_target_time_ &&
          *stored_target_time_ == static_cast<std::uint64_t>(metadata.capture_timestamp_ns) &&
          stored_target_frame_ == metadata.frame_id && stored_target_data_ == target_data;
      if (!stored) {
        auto target_points =
            cloud_on_device(target_cloud, "target", false, target_staging_, stream);
        if (!target_points) {
          return failed(std::move(target_points).error());
        }
        if (const cudaError_t status =
                registration_->set_target(target_points->points, target_points->count, stream);
            status != cudaSuccess) {
          return failed(cuda_failure("set_target", status));
        }
        stored_target_time_ = static_cast<std::uint64_t>(metadata.capture_timestamp_ns);
        stored_target_frame_ = metadata.frame_id;
        stored_target_data_ = target_data;
        target_frame_id_ = target_cloud.header ? target_cloud.header->frame_id : std::string{};
      }
    }
    if (const cudaError_t status = registration_->align(
            source_points->points, count, guess.value_or(identity_transform()), stream, result);
        status != cudaSuccess) {
      return failed(cuda_failure("align", status));
    }
    aligned_frame_id = target_frame_id_;
  }

  // The outputs, with the source's provenance.
  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*source_sample);
  const auto emit_host =
      [&emit](holoscan::Output<holoscan::Tensor>& port, std::span<const double> values,
              std::span<const std::int64_t> shape) -> holoscan::expected<void, holoscan::Error> {
    auto loan =
        port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kFloat64});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    if (auto written =
            loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                            -> holoscan::expected<void, holoscan::Error> {
              std::memcpy(writer.data_as<double>(), values.data(), values.size_bytes());
              return {};
            });
        !written) {
      return written;
    }
    return port.emit(std::move(*loan), emit);
  };

  constexpr std::array<std::int64_t, 2> kMatrixShape{4, 4};
  constexpr std::array<std::int64_t, 1> kStatsShape{static_cast<std::int64_t>(kStatsSize)};
  std::array<double, kStatsSize> values{};
  values[kStatFitness] = result.fitness;
  values[kStatRmse] = result.rmse;
  values[kStatIterations] = result.iterations;
  values[kStatConverged] = result.converged ? 1.0 : 0.0;
  values[kStatStatus] = static_cast<double>(result.status);
  values[kStatCorrespondences] = result.correspondences;
  values[kStatSourcePoints] = result.source_points;
  values[kStatTargetPoints] = result.target_points;

  // The aligned cloud first: its kernels run while the host emits the rest.
  if (config_.aligned_output) {
    if (auto sent = emit_aligned(source_cloud, *source_points, result.transform, aligned_frame_id,
                                 emit, stream);
        !sent) {
      return failed(std::move(sent).error());
    }
  }
  if (config_.transform_output) {
    if (auto sent = emit_host(transform, result.transform, kMatrixShape); !sent) {
      return sent;
    }
  }
  if (config_.stats_output) {
    if (auto sent = emit_host(stats, values, kStatsShape); !sent) {
      return sent;
    }
  }
  if (config_.mode == IcpMode::kOdometry && config_.pose_output) {
    return emit_host(pose, pose_out, kMatrixShape);
  }
  return {};
}

}  // namespace holoscan::icp
