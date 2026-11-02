// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/height_map/grid_op.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>

#include "tensor_checks.hpp"

namespace holoscan::height_map {
holoscan::expected<MapGeometry, holoscan::Error> read_map_info(const holoscan::Tensor& tensor,
                                                               cudaStream_t stream) {
  const auto fail = [](const char* message) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, message});
  };
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (!detail::is_dtype(tensor.dtype(), kFloat64) || shape.size() != 1 ||
      shape[0] != static_cast<std::int64_t>(kMapInfoSize)) {
    return fail("map_info must be float64 [5]");
  }
  std::array<double, kMapInfoSize> values{};
  if (auto copied = tensor.copy_to_host(values.data(), sizeof(values), stream); !copied) {
    return holoscan::make_unexpected(std::move(copied).error());
  }
  const double rows = values[kInfoRows];
  const double cols = values[kInfoCols];
  const double resolution = values[kInfoResolution];
  if (!(rows >= 1.0 && rows <= kMaxGridSide && rows == std::floor(rows)) ||
      !(cols >= 1.0 && cols <= kMaxGridSide && cols == std::floor(cols)) ||
      !(resolution > 0.0 && std::isfinite(resolution)) || !std::isfinite(values[kInfoOriginX]) ||
      !std::isfinite(values[kInfoOriginY])) {
    return fail("map_info holds an unusable grid");
  }
  return MapGeometry{.resolution = static_cast<float>(resolution),
                     .rows = static_cast<int>(rows),
                     .cols = static_cast<int>(cols),
                     .origin_x = values[kInfoOriginX],
                     .origin_y = values[kInfoOriginY]};
}

holoscan::Error GridOp::invalid(const std::string& message) const {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, name_ + ": " + message};
}

holoscan::Error GridOp::cuda_failure(const char* what, cudaError_t status) const {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         name_ + ": " + what + ": " + cudaGetErrorString(status)};
}

holoscan::unexpected<holoscan::Error> GridOp::emit_failure(const char* port,
                                                           holoscan::Error error) const {
  // Keeps the code, so that a caller can still tell backpressure from failure.
  const char* hint = error.code == holoscan::ErrorCode::kBackpressured
                         ? " (its queue is full: the consumer is behind)"
                         : " (is the port connected?)";
  error.message = name_ + ": emitting " + port + " failed" +
                  (error.message.empty() ? std::string(hint) : ": " + error.message);
  return holoscan::make_unexpected(std::move(error));
}

holoscan::expected<cudaStream_t, holoscan::Error> GridOp::stream_of(
    holoscan::ExecutionContext& context) const {
  // The runtime assigns a stream to an operator with device outputs, and
  // writes their loans on that stream only.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        name_ + ": the runtime gave no CUDA stream for the device outputs"});
  }
  return stream;
}

holoscan::TensorOutputSpec GridOp::layer_output(DLDataType dtype, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = dtype,
              .rank = 2U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

holoscan::TensorOutputSpec GridOp::info_output() {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = kFloat64,
              .rank = 1U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(kMapInfoSize * sizeof(double)),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

void GridOp::declare_image(holoscan::OperatorSpec& spec,
                           holoscan::Output<holoscan::schema::ImageT>& port, const char* name,
                           std::uint64_t bytes) {
  spec.output(port, name)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kUInt8,
                  .rank = 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(bytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::expected<void, holoscan::Error> GridOp::send(OutputTensor& out,
                                                       holoscan::Output<holoscan::Tensor>& port,
                                                       const char* name,
                                                       const holoscan::EmitOptions& emit) const {
  if (!out.loan || !out.writer) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, name_ + ": " + name + " was not written"});
  }
  auto committed = std::move(*out.writer).commit();
  out.writer.reset();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  auto sent = port.emit(std::move(*out.loan), emit);
  out.loan.reset();
  if (!sent) {
    return emit_failure(name, std::move(sent).error());
  }
  return {};
}

holoscan::expected<void, holoscan::Error> GridOp::send_info(
    holoscan::Output<holoscan::Tensor>& port, const char* name, const MapGeometry& geometry,
    const holoscan::EmitOptions& emit) const {
  std::array<double, kMapInfoSize> values{};
  values[kInfoOriginX] = geometry.origin_x;
  values[kInfoOriginY] = geometry.origin_y;
  values[kInfoResolution] = geometry.resolution;
  values[kInfoRows] = geometry.rows;
  values[kInfoCols] = geometry.cols;
  constexpr std::array<std::int64_t, 1> kShape{static_cast<std::int64_t>(kMapInfoSize)};
  auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{.shape = kShape, .dtype = kFloat64});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  if (auto written =
          loan->write_host_and_commit([&values](const holoscan::TensorOutputWriteGuard& writer)
                                          -> holoscan::expected<void, holoscan::Error> {
            std::memcpy(writer.data_as<double>(), values.data(), sizeof(values));
            return {};
          });
      !written) {
    return written;
  }
  if (auto sent = port.emit(std::move(*loan), emit); !sent) {
    return emit_failure(name, std::move(sent).error());
  }
  return {};
}

holoscan::expected<void, holoscan::Error> GridOp::send_image(
    OutputTensor& out, holoscan::Output<holoscan::schema::ImageT>& port, const char* name, int rows,
    int cols, const std::string& frame_id, const holoscan::EmitOptions& emit) const {
  if (!out.loan || !out.writer) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, name_ + ": " + name + " was not written"});
  }
  auto committed = std::move(*out.writer).commit();
  out.writer.reset();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = cols;
  descriptor.height = rows;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
  if (emit.capture_time && emit.capture_time->valid()) {
    descriptor.header->capture_timestamp_ns = emit.capture_time->timestamp_ns;
  }
  descriptor.header->frame_id = frame_id;
  auto sent = port.emit_tensor(std::move(*out.loan), descriptor, emit);
  out.loan.reset();
  if (!sent) {
    return emit_failure(name, std::move(sent).error());
  }
  return {};
}

}  // namespace holoscan::height_map
