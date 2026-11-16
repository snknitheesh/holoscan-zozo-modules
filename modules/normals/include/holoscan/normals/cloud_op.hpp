// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CloudOp: what the module's operators share. Each takes a point cloud as a
// PointCloud message of holoscan-messages (DepthToPointCloudOp's
// `point_cloud`, for one):
//
//   points     float32 [N, 3], or organized [H, W, 3]; a point with a
//              non-finite coordinate is invalid.
//   colors     uint8 RGB of the points' shape, or absent.
//   normals    float32 of the points' shape, or absent.
//   intensity  float32, one per point, or absent.
//
// in host, pinned host or device memory on the operator's GPU, and emits
// device tensors with the cloud's capture time and frame id. The base
// receives and checks the cloud, puts the fields it carries on the GPU
// (copying host ones every frame, which it logs once), and allocates, commits
// and emits the outputs; a refused emit fails the frame with an error that
// names the port.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/point_cloud.hpp>

namespace holoscan::normals {

enum class OutputLayout : std::uint8_t {
  kKeep,   // The input's shape; points without a result NaN. Organized clouds stay organized.
  kDense,  // [M, 3] of the points with a result, in input order.
};

class CloudOp : public holoscan::Operator<> {
 public:
  ~CloudOp() override;
  CloudOp(const CloudOp&) = delete;
  CloudOp& operator=(const CloudOp&) = delete;

  void stop() override;

 protected:
  // A received cloud, its fields on the GPU.
  struct Cloud {
    holoscan::Sample<holoscan::messages::schema::PointCloudT> sample;
    // The cloud's header: its frame, capture time and device sequence.
    holoscan::messages::Header header;
    const float* xyz{};
    // The optional fields, null when the cloud does not carry them or has no
    // points.
    const std::uint8_t* rgb{};
    const float* normals{};
    const float* intensity{};
    bool has_colors{};
    bool has_normals{};
    bool has_intensity{};
    std::uint32_t count{};
    // The points' shape, and the image grid: [H, W, 3] is organized, W x H;
    // [N, 3] is N x 1.
    std::vector<std::int64_t> shape;
    bool organized{};
    int width{};
    int height{};
    cudaStream_t stream{};

    // The shape of one value per point: the points' shape without the 3.
    [[nodiscard]] std::vector<std::int64_t> point_shape() const {
      return {shape.begin(), shape.end() - 1};
    }
    // The shape as the messages module's helpers take it.
    [[nodiscard]] holoscan::messages::PointCloudShape cloud_shape() const {
      return organized ? holoscan::messages::organized_cloud(height, width)
                       : holoscan::messages::unorganized_cloud(count);
    }
  };

  // A device tensor output being written: allocate() makes it, send() emits it.
  struct OutputTensor {
    std::optional<holoscan::TensorOutputLoan> loan;
    std::optional<holoscan::TensorOutputWriteGuard> writer;  // Borrows `loan`.

    template <typename T>
    [[nodiscard]] T* data() const noexcept {
      return writer ? writer->data_as<T>() : nullptr;
    }
  };

  // `max_points`: the largest cloud accepted, 1..kMaxPointsLimit.
  // @throws std::invalid_argument if max_points is not usable.
  CloudOp(std::string name, std::uint32_t max_points);

  // Receives and checks the next cloud on `port`, named `port_name` in
  // errors, and puts the fields it carries on the GPU: in place when they are
  // in device memory, else copied through a staging buffer.
  [[nodiscard]] holoscan::expected<Cloud, holoscan::Error> receive_cloud(
      holoscan::Input<holoscan::messages::schema::PointCloudT>& port, const char* port_name,
      holoscan::ExecutionContext& context);

  // Allocates `out` on `port` and begins its write on the cloud's stream.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> allocate(
      OutputTensor& out, holoscan::Output<holoscan::Tensor>& port,
      std::span<const std::int64_t> shape, DLDataType dtype, const Cloud& cloud);
  // Commits `out` and emits it on `port` with the cloud's capture time and
  // frame id; a refused emit names `name`.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> send(
      OutputTensor& out, holoscan::Output<holoscan::Tensor>& port, const char* name,
      const Cloud& cloud);

  [[nodiscard]] holoscan::Error invalid(const std::string& message) const;
  [[nodiscard]] holoscan::Error cuda_failure(const char* what, cudaError_t status) const;
  // The error of a failed emit on `port`, naming the operator and the port.
  [[nodiscard]] holoscan::unexpected<holoscan::Error> emit_failure(const char* port,
                                                                   holoscan::Error error) const;
  // Copies `device_count` to the host after the work queued so far on the
  // cloud's stream, waits for that copy (not for later work on the stream),
  // and returns the count.
  [[nodiscard]] holoscan::expected<std::uint32_t, holoscan::Error> wait_for_count(
      const std::uint32_t* device_count, const Cloud& cloud);

  [[nodiscard]] std::uint32_t max_points() const noexcept { return max_points_; }
  [[nodiscard]] const std::string& op_name() const noexcept { return name_; }

  // A device output of `dtype` of any shape up to rank 3, of at most `bytes`.
  [[nodiscard]] static holoscan::TensorOutputSpec device_output(DLDataType dtype,
                                                                std::uint64_t bytes);

 private:
  struct Impl;

  // The cloud field `field` (0 points, 1 colors, 2 normals, 3 intensity), of
  // `bytes`, on the GPU.
  [[nodiscard]] holoscan::expected<const void*, holoscan::Error> on_gpu(
      const holoscan::Tensor& tensor, std::size_t bytes, int field, const Cloud& cloud,
      const char* port_name);

  std::string name_;
  std::uint32_t max_points_;
  std::unique_ptr<Impl> impl_;
};

inline constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
inline constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
inline constexpr DLDataType kUInt32{kDLUInt, 32U, 1U};

}  // namespace holoscan::normals
