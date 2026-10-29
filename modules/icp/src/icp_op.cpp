// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/icp/icp_op.hpp"

#include <algorithm>
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

#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace holoscan::icp {
namespace {

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

// The point count of a [N, 3] or [H, W, 3] float32 cloud.
holoscan::expected<std::uint64_t, holoscan::Error> point_count(const holoscan::Tensor& tensor,
                                                               const char* name) {
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (!is_dtype(tensor.dtype(), kFloat32) || shape.size() < 2 || shape.size() > 3 ||
      shape.back() != 3) {
    return holoscan::make_unexpected(
        invalid(std::string(name) + " must be float32 [N, 3] or [H, W, 3]"));
  }
  std::uint64_t count = 1;
  for (std::size_t axis = 0; axis + 1 < shape.size(); ++axis) {
    if (shape[axis] < 0) {
      return holoscan::make_unexpected(invalid(std::string(name) + " has a negative extent"));
    }
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
    spec.output(aligned, "aligned")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    .dtype = kFloat32,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                },
            .bounds = holoscan::TensorBounds{.max_rank = 3U,
                                             .max_byte_span = std::uint64_t{config_.max_points} *
                                                              3U * sizeof(float)},
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
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
    // The newest target, when there is one; otherwise the stored one.
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
}

void IcpOp::stop() { release(); }

void IcpOp::release() noexcept {
  if (own_stream_ != nullptr) {
    cudaStreamSynchronize(own_stream_);
  }
  // The registration's buffers are freed only after its work is done: every
  // call returns with its stream synchronized.
  registration_.reset();
  odometry_.reset();
  for (Staging* staging : {&source_staging_, &target_staging_}) {
    if (staging->data != nullptr) {
      cudaFree(staging->data);
    }
    *staging = {};
  }
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

holoscan::expected<const float*, holoscan::Error> IcpOp::cloud_on_device(
    const holoscan::Tensor& tensor, std::uint32_t count, Staging& staging, cudaStream_t stream) {
  if (!packed(tensor)) {
    return holoscan::make_unexpected(invalid("clouds must be packed"));
  }
  const std::size_t bytes = static_cast<std::size_t>(count) * 3U * sizeof(float);
  if (on_gpu(tensor)) {
    if (reinterpret_cast<std::uintptr_t>(tensor.data()) % sizeof(float) != 0U) {
      return holoscan::make_unexpected(invalid("cloud data must be aligned to 4 bytes"));
    }
    int device = 0;
    if (const cudaError_t status = cudaStreamGetDevice(stream, &device); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamGetDevice", status));
    }
    if (tensor.device().device_type == kDLCUDA && tensor.device().device_id != device) {
      return holoscan::make_unexpected(invalid("a cloud is on another GPU than the operator"));
    }
    // The runtime orders device input on its own stream only.
    if (stream == own_stream_) {
      if (auto ready = tensor.synchronize_for_host(); !ready) {
        return holoscan::make_unexpected(std::move(ready).error());
      }
    }
    return static_cast<const float*>(tensor.data());
  }
  // Host memory: wait for its producer, then copy it. Every registration
  // synchronizes its stream before it returns, so the copy is done before
  // compute() returns and the input is released.
  if (auto ready = tensor.synchronize_for_host(); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  if (!logged_host_copy_) {
    logged_host_copy_ = true;
    HOLOSCAN_LOG_INFO(
        "IcpOp: clouds arrive in host memory and are copied to the GPU; send them in device "
        "memory to avoid the copy");
  }
  if (staging.bytes < bytes) {
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
  return static_cast<const float*>(staging.data);
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

  // The source.
  const holoscan::Tensor& source_tensor = source_sample->data;
  auto source_count = point_count(source_tensor, "source");
  if (!source_count) {
    return holoscan::make_unexpected(std::move(source_count).error());
  }
  if (*source_count > config_.max_points) {
    return holoscan::make_unexpected(invalid("source has " + std::to_string(*source_count) +
                                             " points, more than the configured maximum of " +
                                             std::to_string(config_.max_points)));
  }
  const auto count = static_cast<std::uint32_t>(*source_count);
  auto source_points = cloud_on_device(source_tensor, count, source_staging_, stream);
  if (!source_points) {
    return holoscan::make_unexpected(std::move(source_points).error());
  }

  // The initial guess.
  std::optional<Matrix4> guess;
  if (config_.initial_guess_input) {
    auto sample = initial_guess.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const holoscan::Tensor& tensor = sample->data;
    const std::span<const std::int64_t> shape = tensor.shape_span();
    const bool f32 = is_dtype(tensor.dtype(), kFloat32);
    if (!(f32 || is_dtype(tensor.dtype(), kFloat64)) || shape.size() != 2 || shape[0] != 4 ||
        shape[1] != 4 || !packed(tensor)) {
      return holoscan::make_unexpected(invalid("initial_guess must be float32 or float64 [4, 4]"));
    }
    std::array<std::byte, 16U * sizeof(double)> bytes{};
    const std::size_t size = 16U * (f32 ? sizeof(float) : sizeof(double));
    if (auto copied = tensor.copy_to_host(bytes.data(), size, stream); !copied) {
      return holoscan::make_unexpected(std::move(copied).error());
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
      return holoscan::make_unexpected(invalid("initial_guess is not a rigid transform"));
    }
    guess = m;
  }

  // The registration.
  IcpResult result;
  Matrix4 pose_out = identity_transform();
  if (config_.mode == IcpMode::kOdometry) {
    OdometryStep step;
    if (const cudaError_t status =
            odometry_->track(*source_points, count, stream, step, guess ? &*guess : nullptr);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("odometry", status));
    }
    result = step.result;
    pose_out = step.pose;
  } else {
    // The target: this source's own in kPairs; in kMap a new one when it
    // came, otherwise the stored one.
    if (config_.mode == IcpMode::kPairs || !target.is_missing()) {
      auto target_sample = target.receive();
      if (!target_sample) {
        return holoscan::make_unexpected(std::move(target_sample).error());
      }
      const holoscan::SampleMetadata& metadata = target_sample->metadata;
      const holoscan::Tensor& tensor = target_sample->data;
      const bool stored =
          config_.mode == IcpMode::kMap && stored_target_time_ &&
          *stored_target_time_ == static_cast<std::uint64_t>(metadata.capture_timestamp_ns) &&
          stored_target_frame_ == metadata.frame_id && stored_target_data_ == tensor.data();
      if (!stored) {
        auto target_count = point_count(tensor, "target");
        if (!target_count) {
          return holoscan::make_unexpected(std::move(target_count).error());
        }
        if (*target_count > config_.max_points) {
          return holoscan::make_unexpected(invalid("target has " + std::to_string(*target_count) +
                                                   " points, more than the configured maximum of " +
                                                   std::to_string(config_.max_points)));
        }
        const auto n = static_cast<std::uint32_t>(*target_count);
        auto target_points = cloud_on_device(tensor, n, target_staging_, stream);
        if (!target_points) {
          return holoscan::make_unexpected(std::move(target_points).error());
        }
        if (const cudaError_t status = registration_->set_target(*target_points, n, stream);
            status != cudaSuccess) {
          return holoscan::make_unexpected(cuda_failure("set_target", status));
        }
        stored_target_time_ = static_cast<std::uint64_t>(metadata.capture_timestamp_ns);
        stored_target_frame_ = metadata.frame_id;
        stored_target_data_ = tensor.data();
      }
    }
    if (const cudaError_t status = registration_->align(
            *source_points, count, guess.value_or(identity_transform()), stream, result);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("align", status));
    }
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

  // The aligned cloud first: its kernel runs while the host emits the rest.
  if (config_.aligned_output) {
    const std::span<const std::int64_t> shape = source_tensor.shape_span();
    auto loan = aligned.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::vector<std::int64_t>(shape.begin(), shape.end()), .dtype = kFloat32});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (const cudaError_t status = launch_transform_points(*source_points, writer->data_as<float>(),
                                                           count, result.transform, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("transform kernel", status));
    }
    if (stream == own_stream_) {
      // The runtime knows nothing of this stream; finish before the output goes.
      if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
      }
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    if (auto sent = aligned.emit(std::move(*loan), emit); !sent) {
      return sent;
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
