// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// GridOp: what the module's operators share. Each emits map layers, float32
// or uint8 [rows, cols] tensors in device memory (grid.hpp), the map's
// `map_info` (float64 [kMapInfoSize] in host memory) and optionally a color
// image for HoloViz, every one with the capture time and frame id of the
// input it was made from. The base gets the operator's CUDA stream,
// allocates, commits and emits the outputs; a refused emit fails the frame
// with an error that names the port.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/height_map/grid.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::height_map {

inline constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
inline constexpr DLDataType kFloat64{kDLFloat, 64U, 1U};
inline constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// Reads a `map_info` tensor (float64 [kMapInfoSize], host or device memory).
// @returns the geometry, or an error naming what is wrong with it.
[[nodiscard]] holoscan::expected<MapGeometry, holoscan::Error> read_map_info(
    const holoscan::Tensor& tensor, cudaStream_t stream);

class GridOp : public holoscan::Operator<> {
 public:
  ~GridOp() override = default;
  GridOp(const GridOp&) = delete;
  GridOp& operator=(const GridOp&) = delete;

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

  explicit GridOp(std::string name) : name_(std::move(name)) {}

  // The stream the runtime gives the operator for its device outputs.
  [[nodiscard]] holoscan::expected<cudaStream_t, holoscan::Error> stream_of(
      holoscan::ExecutionContext& context) const;

  // Allocates `out` on `port` (a tensor or image output) and begins its
  // write on `stream`.
  template <typename Port>
  [[nodiscard]] static holoscan::expected<void, holoscan::Error> allocate(
      OutputTensor& out, Port& port, std::span<const std::int64_t> shape, DLDataType dtype,
      cudaStream_t stream) {
    auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    out.loan.emplace(std::move(*loan));
    auto writer = out.loan->write(stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    out.writer.emplace(std::move(*writer));
    return {};
  }
  // Commits `out` and emits it on `port` with `emit`; a refused emit names
  // `name`.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> send(
      OutputTensor& out, holoscan::Output<holoscan::Tensor>& port, const char* name,
      const holoscan::EmitOptions& emit) const;
  // Emits `geometry` on `port` as float64 [kMapInfoSize] in host memory.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> send_info(
      holoscan::Output<holoscan::Tensor>& port, const char* name, const MapGeometry& geometry,
      const holoscan::EmitOptions& emit) const;
  // Commits `out`, an RGB8 [rows, cols, 3] image, and emits it on `port`
  // with a header holding `frame_id` and the capture time of `emit`.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> send_image(
      OutputTensor& out, holoscan::Output<holoscan::schema::ImageT>& port, const char* name,
      int rows, int cols, const std::string& frame_id, const holoscan::EmitOptions& emit) const;

  [[nodiscard]] holoscan::Error invalid(const std::string& message) const;
  [[nodiscard]] holoscan::Error cuda_failure(const char* what, cudaError_t status) const;
  // The error of a failed emit on `port`, naming the operator and the port.
  [[nodiscard]] holoscan::unexpected<holoscan::Error> emit_failure(const char* port,
                                                                   holoscan::Error error) const;

  [[nodiscard]] const std::string& op_name() const noexcept { return name_; }

  // A device layer of `dtype`, rank 2, of at most `bytes`.
  [[nodiscard]] static holoscan::TensorOutputSpec layer_output(DLDataType dtype,
                                                               std::uint64_t bytes);
  // The `map_info` output.
  [[nodiscard]] static holoscan::TensorOutputSpec info_output();
  // A device RGB8 image of at most `bytes`, as a schema::ImageT.
  static void declare_image(holoscan::OperatorSpec& spec,
                            holoscan::Output<holoscan::schema::ImageT>& port, const char* name,
                            std::uint64_t bytes);

 private:
  std::string name_;
};

}  // namespace holoscan::height_map
