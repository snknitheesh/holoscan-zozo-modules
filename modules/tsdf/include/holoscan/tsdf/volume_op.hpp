// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// detail::VolumeOp: what TsdfIntegrateOp and TsdfMeshOp share. Not a stable
// API: it is installed only because the operators derive from it, and may
// change in any release; derive from holoscan::Operator<> and use TsdfVolume
// instead.
//
// Each operator works on a TsdfVolume it shares with the other through a
// std::shared_ptr; the volume orders their GPU work (see tsdf_volume.hpp), so
// they may run on different streams and threads. The base finds the CUDA
// stream to work on, receives and checks a camera pose, stages host input on
// the GPU, and allocates, commits and emits device outputs; a refused emit
// fails the frame with an error that names the operator and the port.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>

namespace holoscan::tsdf::detail {

inline constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
inline constexpr DLDataType kFloat64{kDLFloat, 64U, 1U};
inline constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
inline constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};
inline constexpr DLDataType kUInt32{kDLUInt, 32U, 1U};

class VolumeOp : public holoscan::Operator<> {
 public:
  ~VolumeOp() override;
  VolumeOp(const VolumeOp&) = delete;
  VolumeOp& operator=(const VolumeOp&) = delete;

  void stop() override;

  [[nodiscard]] const std::shared_ptr<TsdfVolume>& volume() const noexcept { return volume_; }

 protected:
  // A device output being written: allocate() makes it, send() emits it.
  struct OutputTensor {
    std::optional<holoscan::TensorOutputLoan> loan;
    std::optional<holoscan::TensorOutputWriteGuard> writer;  // Borrows `loan`.

    template <typename T>
    [[nodiscard]] T* data() const noexcept {
      return writer ? writer->data_as<T>() : nullptr;
    }
  };

  // `kind` is the operator's class, which errors name with the instance's
  // path once compute() has run.
  // @throws std::invalid_argument if `volume` is null.
  VolumeOp(std::string kind, std::shared_ptr<TsdfVolume> volume);

  // Names this activation's operator in errors; call first in compute().
  void bind(const holoscan::ExecutionContext& context);

  // The runtime's CUDA stream for this activation, which must be on the
  // volume's GPU; when the runtime gives none (every input and output in
  // host memory), a stream of the operator's own.
  [[nodiscard]] holoscan::expected<cudaStream_t, holoscan::Error> stream_for(
      holoscan::ExecutionContext& context);

  // A [4, 4] float32 or float64 rigid transform from `tensor`, in host or
  // device memory; `name` names it in errors. Device memory is copied to the
  // host, which waits for the stream (logged once).
  [[nodiscard]] holoscan::expected<Matrix4, holoscan::Error> read_pose(
      const holoscan::Tensor& tensor, const char* name, cudaStream_t stream);

  // `bytes` of `tensor` on the volume's GPU: its own data when it is in
  // device or managed memory of that GPU, otherwise a copy in staging buffer
  // `slot` (0 or 1), enqueued on `stream` (the first copy is logged once).
  // The staging buffer is allocated once, of `capacity` bytes (at least
  // `bytes`), so it never grows. Pageable host memory: the copy returns once
  // the data is staged, which waits for it. A tensor on another GPU, or in
  // memory of another kind, is refused.
  [[nodiscard]] holoscan::expected<const void*, holoscan::Error> on_device(
      const holoscan::Tensor& tensor, std::size_t bytes, std::size_t capacity, int slot,
      cudaStream_t stream);

  // Call after the activation's GPU work is enqueued on `stream`: on the
  // operator's own stream, which the runtime does not order its inputs'
  // release after, waits for the work to finish when it reads an input in
  // place. Otherwise nothing.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> settle(cudaStream_t stream);

  // Allocates `out` on `port` and begins its write on `stream`.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> allocate(
      OutputTensor& out, holoscan::Output<holoscan::Tensor>& port,
      std::span<const std::int64_t> shape, DLDataType dtype, cudaStream_t stream);
  // Commits `out` and emits it on `port` with `options`; a refused emit names
  // `name`.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> send(
      OutputTensor& out, holoscan::Output<holoscan::Tensor>& port, const char* name,
      const holoscan::EmitOptions& options);

  [[nodiscard]] holoscan::Error invalid(const std::string& message) const;
  [[nodiscard]] holoscan::Error cuda_failure(const char* what, cudaError_t status) const;
  // The error of a failed emit on `port`, naming the operator and the port.
  [[nodiscard]] holoscan::unexpected<holoscan::Error> emit_failure(const char* port,
                                                                   holoscan::Error error) const;

  // "TsdfIntegrateOp 'integrate'", or the class alone before compute().
  [[nodiscard]] const std::string& op_name() const noexcept { return name_; }

  // A device output of `dtype` of any shape up to rank 3, of at most `bytes`.
  [[nodiscard]] static holoscan::TensorOutputSpec device_output(DLDataType dtype,
                                                                std::uint64_t bytes);
  // A host output of `dtype` of any shape up to rank 2, of at most `bytes`.
  [[nodiscard]] static holoscan::TensorOutputSpec host_output(DLDataType dtype,
                                                              std::uint64_t bytes);

 private:
  struct Impl;

  std::string kind_;
  std::string path_;
  std::string name_;
  std::shared_ptr<TsdfVolume> volume_;
  std::unique_ptr<Impl> impl_;
};

// Checks a tensor's dtype.
[[nodiscard]] bool is_dtype(const DLDataType& dtype, const DLDataType& expected);
// Whether a tensor is packed (row-major, no gaps); any stride is accepted on
// an axis of extent 1.
[[nodiscard]] bool packed(const holoscan::Tensor& tensor);

}  // namespace holoscan::tsdf::detail
