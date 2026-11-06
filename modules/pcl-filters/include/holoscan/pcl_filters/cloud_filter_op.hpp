// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CloudFilterOp: what the module's operators share. Each takes a point cloud
// and sends the filtered cloud, both holoscan-messages PointCloud messages:
//
//   point_cloud           points float32 [N, 3], or organized [H, W, 3], with
//                         optional colors, normals and intensity; packed, in
//                         host or device memory on the operator's GPU. A
//                         point with a non-finite coordinate is invalid.
//   filtered_point_cloud  the filtered cloud in device memory, with every
//                         field the input has (an absent one stays the
//                         messages module's empty [0]), the input's header
//                         (capture time and frame id) and its sample's
//                         capture time and frame number. See OutputLayout for
//                         its shape and valid_count.
//
// Waits: a kDense output's size is known only once the filter has run, so the
// operator waits for its GPU work once per frame before allocating it. kKeep
// and kPadded never wait. Input in host memory costs up to one wait per frame
// too, for its copy to the GPU: its producer may reuse that memory once
// compute() returns, unless its allocator can defer the reuse until the copy
// is done.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/pcl_filters/filters.hpp>

namespace holoscan::pcl_filters {

enum class OutputLayout : std::uint8_t {
  // The input's shape, organized or not; removed points NaN (see CloudOutput).
  // valid_count -1. No wait.
  kKeep,
  // [M, 3] of the M kept points, in input order. valid_count M. One wait per
  // frame for M.
  kDense,
  // [N, 3] for an input of N points: the kept points first, in input order,
  // then removed points. valid_count -1. No wait.
  kPadded,
};

// The settings every operator of the module has.
struct CloudSettings {
  OutputLayout layout{OutputLayout::kKeep};
  // Largest cloud accepted, 1..kMaxPointsLimit points; a larger one fails the
  // run (compute() returns its error). It sizes the outputs. The default fits
  // 1920x1080.
  std::uint32_t max_points{1920U * 1080U};
  // The optional fields the output has room for: each takes max_points
  // values in every output buffer. A cloud carrying another fails the run.
  holoscan::messages::PointCloudContents fields{.colors = true, .normals = true, .intensity = true};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_settings(const CloudSettings& settings);

class CloudFilterOp : public holoscan::Operator<> {
 public:
  ~CloudFilterOp() override;
  CloudFilterOp(const CloudFilterOp&) = delete;
  CloudFilterOp& operator=(const CloudFilterOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;

  holoscan::Input<holoscan::messages::schema::PointCloudT> point_cloud;
  holoscan::Output<holoscan::messages::schema::PointCloudT> filtered_point_cloud;

 protected:
  // A received cloud, its fields on the GPU.
  struct Cloud {
    holoscan::Sample<holoscan::messages::schema::PointCloudT> sample;
    // The optional fields it carries.
    holoscan::messages::PointCloudContents contents;
    // Null for an absent field, and may be for a cloud of no points.
    CloudFields fields;
    holoscan::messages::PointCloudShape shape;
    std::uint32_t count{};
    cudaStream_t stream{};
  };

  // @throws std::invalid_argument if `settings` are not valid.
  CloudFilterOp(std::string name, CloudSettings settings);

  // Receives and checks the next cloud, copying host fields to the GPU.
  [[nodiscard]] holoscan::expected<Cloud, holoscan::Error> receive(
      holoscan::ExecutionContext& context);
  // A device buffer of one byte per point, for a keep mask.
  [[nodiscard]] holoscan::expected<std::uint8_t*, holoscan::Error> mask(const Cloud& cloud);
  // Sends the points `keep` marks, in the configured layout.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_kept(const Cloud& cloud,
                                                                    const std::uint8_t* keep);
  // Sends a packed cloud of the `device_count` points that `write` writes on
  // the cloud's stream, into buffers of at least that many points: with
  // kDense, waits for the count and allocates that many; with kPadded,
  // allocates the input's size and pads after the count on the device.
  using PackedWriter = std::function<cudaError_t(const CloudOutput& out)>;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_packed(
      const Cloud& cloud, const std::uint32_t* device_count, const PackedWriter& write);

  [[nodiscard]] holoscan::Error invalid(const std::string& message) const;
  [[nodiscard]] holoscan::Error cuda_failure(const char* what, cudaError_t status) const;
  // The error of a failed emit on `port`, naming the operator and the port;
  // a stop's RUNTIME_PUBLICATION_STOPPED error unchanged, so that the run
  // still ends stopped.
  [[nodiscard]] holoscan::unexpected<holoscan::Error> emit_failure(const char* port,
                                                                   holoscan::Error error) const;
  [[nodiscard]] const CloudSettings& settings() const noexcept { return settings_; }

 private:
  struct Impl;

  // Says, once, that host input is copied to the GPU every frame, and how not to.
  void log_host_copy(const std::string& fields, std::size_t bytes) const;
  // Allocates a cloud of `shape` with the input's fields, runs `write` on
  // all of them on the cloud's stream and sends it.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_cloud(
      const Cloud& cloud, const holoscan::messages::PointCloudShape& shape,
      std::int64_t valid_count, const PackedWriter& write);

  std::string name_;
  CloudSettings settings_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::pcl_filters
