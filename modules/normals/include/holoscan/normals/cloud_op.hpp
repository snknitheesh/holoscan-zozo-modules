// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CloudOp: what the module's operators share. Each takes a point cloud in
// holoscan-point-cloud's layout, with per-point tensors that go with it:
//
//   points  float32 [N, 3], or organized [H, W, 3]; packed, in host, pinned or
//           device memory on the operator's GPU. A point with a non-finite
//           coordinate is invalid.
//
// and emits device tensors with the cloud's capture time and frame id. The
// base receives and checks the cloud and its companions, copies host input to
// the GPU, and allocates, commits and emits the outputs; a refused emit fails
// the frame with an error that names the port.

#pragma once

#include <cuda_runtime_api.h>

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
  // A received cloud, on the GPU.
  struct Cloud {
    holoscan::Sample<holoscan::Tensor> points;
    const float* xyz{};
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
  };

  // A device output being written: allocate() makes it, send() emits it.
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

  // Receives and checks the next cloud on `port`, copying host input to the
  // GPU.
  [[nodiscard]] holoscan::expected<Cloud, holoscan::Error> receive_cloud(
      holoscan::Input<holoscan::Tensor>& port, holoscan::ExecutionContext& context);
  // Receives the tensor on `port` that goes with `cloud`: `dtype` of the
  // points' shape (three values per point). Keeps it in `sample` and returns
  // its data on the GPU; `slot` (0 or 1) names its staging buffer.
  [[nodiscard]] holoscan::expected<const void*, holoscan::Error> receive_matching(
      holoscan::Input<holoscan::Tensor>& port, const char* name, DLDataType dtype,
      const Cloud& cloud, std::optional<holoscan::Sample<holoscan::Tensor>>& sample, int slot);

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
  // Waits for `device_count` on the cloud's stream and returns it.
  [[nodiscard]] holoscan::expected<std::uint32_t, holoscan::Error> wait_for_count(
      const std::uint32_t* device_count, const Cloud& cloud);

  [[nodiscard]] std::uint32_t max_points() const noexcept { return max_points_; }
  [[nodiscard]] const std::string& op_name() const noexcept { return name_; }

  // A device output of `dtype` of any shape up to rank 3, of at most `bytes`.
  [[nodiscard]] static holoscan::TensorOutputSpec device_output(DLDataType dtype,
                                                                std::uint64_t bytes);

 private:
  struct Impl;

  std::string name_;
  std::uint32_t max_points_;
  std::unique_ptr<Impl> impl_;
};

inline constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
inline constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
inline constexpr DLDataType kUInt32{kDLUInt, 32U, 1U};

}  // namespace holoscan::normals
