// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// IcpOp: registers point clouds (PointCloud messages of holoscan-messages)
// with iterative closest point on the GPU (see registration.hpp), in one of
// three modes:
//
// - kPairs: each source cloud is aligned to the target cloud with the same
//   capture time.
// - kMap: each source cloud is aligned to the newest target cloud captured at
//   or before it (and at most map_max_age before it), which the operator
//   keeps until a newer one is taken up: a stored map.
// - kOdometry: each source cloud is aligned to the one before it, and the
//   transforms accumulate into a pose relative to the first cloud: depth
//   odometry.
//
// Every source cloud gives one transform and one set of statistics, also when
// the registration fails, in kOdometry one pose, and optionally the source
// cloud moved by the transform, with its colors, normals and intensities.
// Each output exists only when its config flag is set, and every output that
// exists must be connected: an emit into an unconnected output fails the run.

#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/icp/registration.hpp>
#include <holoscan/messages/point_cloud.hpp>

namespace holoscan::icp {

enum class IcpMode : std::uint8_t { kPairs, kMap, kOdometry };

// The largest cloud the operator accepts.
inline constexpr std::uint32_t kMaxPointsLimit = std::uint32_t{1} << 30;

struct IcpConfig {
  IcpOptions options;
  IcpMode mode{IcpMode::kPairs};
  // Declare the `initial_guess` input: a [4, 4] float32 or float64 rigid
  // transform with the source's capture time, which replaces the identity
  // (kPairs, kMap) or the motion model's guess (kOdometry) as the source's
  // starting point.
  bool initial_guess_input{false};
  // In kOdometry, guess each frame's motion as the previous frame's
  // (constant velocity), or as none.
  bool constant_velocity{true};
  // Declare the `transform`, `stats`, `pose` (kOdometry) and `aligned`
  // outputs.
  bool transform_output{true};
  bool stats_output{true};
  bool pose_output{true};
  bool aligned_output{false};
  // The optional fields the `aligned` output can carry: a source cloud's
  // colors and intensities go out as they are, its normals turned by the
  // transform. A source carrying a field left out here fails the run.
  holoscan::messages::PointCloudContents aligned_fields{
      .colors = true, .normals = true, .intensity = true};
  // Largest cloud accepted, 1..kMaxPointsLimit points; a larger one fails
  // the run (compute() returns an error). It bounds host staging and sizes
  // the `aligned` output's pool: each of its slots holds max_points points,
  // 12 bytes each, plus 3, 12 and 4 bytes for colors, normals and
  // intensities when `aligned_fields` has them (31 bytes in all: 124 MiB a
  // slot at the default), whatever the size of the clouds sent. Set it to
  // the largest cloud the graph sends.
  std::uint32_t max_points{std::uint32_t{1} << 22};
  // In kMap, a target captured more than this before the source is not taken
  // up, and the stored one stays.
  std::chrono::nanoseconds map_max_age{std::chrono::hours{24}};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const IcpConfig& config);

// The `stats` output: float64 [kStatsSize], in this order.
inline constexpr std::size_t kStatFitness = 0;  // 0..1.
inline constexpr std::size_t kStatRmse = 1;     // In the clouds' units.
inline constexpr std::size_t kStatIterations = 2;
inline constexpr std::size_t kStatConverged = 3;  // 1 or 0.
inline constexpr std::size_t kStatStatus = 4;     // An IcpStatus.
inline constexpr std::size_t kStatCorrespondences = 5;
inline constexpr std::size_t kStatSourcePoints = 6;  // After subsampling.
inline constexpr std::size_t kStatTargetPoints = 7;  // After subsampling.
inline constexpr std::size_t kStatsSize = 8;

// The registration result a `stats` and `transform` pair describes.
// @throws std::invalid_argument if `stats` is not kStatsSize long or holds an
// unknown status.
[[nodiscard]] IcpResult result_from(std::span<const double> stats, const Matrix4& transform);

class IcpOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit IcpOp(IcpConfig config);
  ~IcpOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // A cloud: float32 points [N, 3], or an organized [H, W, 3] taken as
  // H * W points, in host, pinned or CUDA device memory on the operator's
  // GPU, as check_point_cloud() accepts them. Non-finite points are skipped.
  holoscan::Input<holoscan::messages::schema::PointCloudT> source;
  // kPairs and kMap: the cloud to align to, as `source`.
  holoscan::Input<holoscan::messages::schema::PointCloudT> target;
  // With `initial_guess_input`: float32 or float64 [4, 4], row-major,
  // mapping source points into the target's frame; host or device memory.
  holoscan::Input<holoscan::Tensor> initial_guess;
  // With `transform_output`: float64 [4, 4] row-major, host: maps source
  // points into the target's frame (in kOdometry, the previous cloud's).
  holoscan::Output<holoscan::Tensor> transform;
  // With `stats_output`: float64 [kStatsSize], host: see kStatFitness.
  holoscan::Output<holoscan::Tensor> stats;
  // kOdometry with `pose_output`: float64 [4, 4] row-major, host: maps the
  // source's points into the first cloud's frame.
  holoscan::Output<holoscan::Tensor> pose;
  // With `aligned_output`: the source cloud moved by `transform`, in its
  // shape, on the GPU: its points moved (non-finite ones as they were), its
  // normals turned, its colors and intensities as they were, absent fields
  // absent. The header has the target's frame id (in kOdometry, the
  // previous cloud's) and the source's capture time and device sequence;
  // valid_count is the source's.
  holoscan::Output<holoscan::messages::schema::PointCloudT> aligned;

 private:
  // A device copy of a host tensor, grown as needed.
  struct Staging {
    void* data{};
    std::size_t bytes{};
  };
  // A received cloud on the GPU: its points, their count, and with
  // `aligned_output` the optional fields it carries (else null); whether any
  // of them is a staging copy.
  struct DeviceCloud {
    const float* points{};
    std::uint32_t count{};
    const void* colors{};
    const float* normals{};
    const float* intensity{};
    bool staged{};
  };

  [[nodiscard]] holoscan::expected<DeviceCloud, holoscan::Error> cloud_on_device(
      const holoscan::messages::schema::PointCloudT& cloud, const char* name, bool fields,
      Staging& staging, cudaStream_t stream);
  [[nodiscard]] holoscan::expected<const void*, holoscan::Error> on_device(
      const holoscan::Tensor& tensor, std::size_t bytes, std::size_t alignment, Staging& staging,
      cudaStream_t stream);
  [[nodiscard]] holoscan::expected<cudaStream_t, holoscan::Error> stream_for(
      holoscan::ExecutionContext& context);
  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_aligned(
      const holoscan::messages::schema::PointCloudT& cloud, const DeviceCloud& on_device,
      const Matrix4& transform, const std::string& frame_id, const holoscan::EmitOptions& emit,
      cudaStream_t stream);
  void release() noexcept;

  IcpConfig config_;
  std::optional<IcpRegistration> registration_;
  std::optional<IcpOdometry> odometry_;
  // kMap: which target sample the stored target came from.
  std::optional<std::uint64_t> stored_target_time_;
  std::uint64_t stored_target_frame_{};
  const void* stored_target_data_{};
  // The frame the stored target (kOdometry: the previous cloud) is in.
  std::string target_frame_id_;
  Staging source_staging_;
  Staging target_staging_;
  Staging colors_staging_;
  Staging normals_staging_;
  Staging intensity_staging_;
  // Recorded after the `aligned` kernels that read the staging buffers: the
  // next copy into them waits for it on the GPU, not on the host.
  cudaEvent_t staging_free_{};
  bool staging_pending_{};
  // Used when the runtime gives this operator no CUDA stream.
  cudaStream_t own_stream_{};
  bool logged_host_copy_{};
  bool logged_guess_copy_{};
};

}  // namespace holoscan::icp
