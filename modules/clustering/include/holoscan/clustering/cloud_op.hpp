// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CloudOp: what the module's operators share. Each takes a PointCloud message
// (holoscan-messages) on `point_cloud`:
//
//   points     float32 [N, 3], or organized [H, W, 3]. A point with a
//              non-finite coordinate is invalid.
//   colors     uint8 RGB, normals float32 xyz, intensity float32: optional,
//              of the points' shape; read only when the operator's outputs
//              carry them.
//
// in host, pinned host or device memory on the operator's GPU, and runs once
// per cloud. Its outputs are in device memory and carry the cloud's capture
// time and frame id. A per-point output keeps the input's layout with
// OutputLayout::kKeep, or packs the points it keeps with kDense.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/point_cloud.hpp>

namespace holoscan::clustering {

enum class OutputLayout : std::uint8_t {
  kKeep,   // The input's shape, one entry per input point. Organized stays organized.
  kDense,  // Only the points kept, packed.
};

// The settings every operator of the module has.
struct CloudSettings {
  OutputLayout layout{OutputLayout::kKeep};
  // Largest cloud accepted, 1..kMaxPointsLimit points; a larger one fails the
  // frame. It sizes the outputs. The default fits 1920x1080.
  std::uint32_t max_points{1920U * 1080U};
  // The optional fields read from a cloud that has them: those the
  // operator's outputs carry. A cloud without one is accepted all the same.
  holoscan::messages::PointCloudContents fields{};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_settings(const CloudSettings& settings);

class CloudOp : public holoscan::Operator<> {
 public:
  ~CloudOp() override;
  CloudOp(const CloudOp&) = delete;
  CloudOp& operator=(const CloudOp&) = delete;

  // Declares the input; a derived operator adds its outputs.
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;

  holoscan::Input<holoscan::messages::schema::PointCloudT> point_cloud;

 protected:
  // A received cloud, on the GPU: the fields in `settings().fields` that the
  // cloud has, the others null.
  struct Cloud {
    holoscan::Sample<holoscan::messages::schema::PointCloudT> sample;
    const float* xyz{};
    const std::uint8_t* rgb{};
    const float* normals{};
    const float* intensity{};
    std::uint32_t count{};
    // The organized cloud's height and width; height 0 for [N, 3].
    std::uint32_t height{};
    std::uint32_t width{};
    // The cloud's header: its frame and capture time.
    holoscan::messages::Header header;
    cudaStream_t stream{};
  };

  // @throws std::invalid_argument if `settings` are not valid.
  CloudOp(std::string name, CloudSettings settings);

  // Receives and checks the next cloud, copying host input to the GPU.
  [[nodiscard]] holoscan::expected<Cloud, holoscan::Error> receive(
      holoscan::ExecutionContext& context);

  // Writes a device tensor of `shape` on `port` with write(data) on the
  // cloud's stream, and emits it with the cloud's capture time and frame id.
  // An empty tensor is emitted without calling `write`.
  using Writer = std::function<cudaError_t(void* data)>;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_tensor(
      holoscan::Output<holoscan::Tensor>& port, const char* port_name, const Cloud& cloud,
      std::span<const std::int64_t> shape, DLDataType dtype, const Writer& write);

  [[nodiscard]] holoscan::Error invalid(const std::string& message) const;
  [[nodiscard]] holoscan::Error cuda_failure(const char* what, cudaError_t status) const;
  // The error of a failed emit on `port`, naming the operator and the port.
  [[nodiscard]] holoscan::unexpected<holoscan::Error> emit_failure(const char* port,
                                                                   holoscan::Error error) const;
  [[nodiscard]] const CloudSettings& settings() const noexcept { return settings_; }

 private:
  struct Impl;

  std::string name_;
  CloudSettings settings_;
  std::unique_ptr<Impl> impl_;
};

// The output spec of a device tensor of up to `max_rank` dimensions and
// `max_bytes`, from the runtime's pool.
[[nodiscard]] holoscan::TensorOutputSpec device_tensor_spec(DLDataType dtype,
                                                            std::uint32_t max_rank,
                                                            std::uint64_t max_bytes);

}  // namespace holoscan::clustering
