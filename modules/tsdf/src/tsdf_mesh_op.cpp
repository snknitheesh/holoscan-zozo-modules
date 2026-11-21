// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/tsdf/tsdf_mesh_op.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "pinhole_camera.hpp"

namespace holoscan::tsdf {
namespace {

namespace hm = holoscan::messages;
using detail::kFloat32;
using detail::kUInt32;
using detail::kUInt8;

holoscan::PayloadOptions image_payload() {
  return holoscan::PayloadOptions{
      .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
  };
}

holoscan::schema::ImageT descriptor(int width, int height,
                                    holoscan::schema::ImageEncoding encoding) {
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding;
  return image;
}

}  // namespace

void validate_config(const TsdfMeshConfig& config) {
  if (!config.volume) {
    throw std::invalid_argument("volume is required");
  }
  if (!config.mesh && !config.raycast) {
    throw std::invalid_argument("enable mesh, raycast or both");
  }
  if (config.mesh) {
    if (config.mesh_every < 1U) {
      throw std::invalid_argument("mesh_every must be at least 1");
    }
    if (config.max_vertices < 1U || config.max_vertices > kMaxMeshCapacity ||
        config.max_triangles < 1U || config.max_triangles > kMaxMeshCapacity) {
      throw std::invalid_argument("max_vertices and max_triangles must be 1.." +
                                  std::to_string(kMaxMeshCapacity));
    }
  }
  if (config.raycast) {
    if (!(std::isfinite(config.raycast_depth_min) && std::isfinite(config.raycast_depth_max) &&
          config.raycast_depth_min > 0.0F && config.raycast_depth_max > config.raycast_depth_min)) {
      throw std::invalid_argument(
          "the ray-cast depth range needs 0 < raycast_depth_min < raycast_depth_max, finite");
    }
    if (config.max_width < 1 || config.max_height < 1 || config.max_width > 8192 ||
        config.max_height > 8192) {
      throw std::invalid_argument("max_width and max_height must be 1..8192");
    }
    if (config.camera_info.max_age <= std::chrono::nanoseconds::zero()) {
      throw std::invalid_argument("camera_info.max_age must be > 0");
    }
  }
}

TsdfMeshOp::TsdfMeshOp(TsdfMeshConfig config)
    : VolumeOp("TsdfMeshOp", config.volume), config_(std::move(config)) {
  validate_config(config_);
  if (config_.mesh) {
    extractor_ = std::make_unique<MeshExtractor>(config_.max_vertices, config_.max_triangles);
  }
}

TsdfMeshOp::~TsdfMeshOp() = default;

bool TsdfMeshOp::color() const noexcept { return volume()->config().color; }

void TsdfMeshOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(pose, "pose");
  // Optional ports exist only when configured: an unconnected input would
  // never be satisfied, and an emit into an unconnected output fails.
  if (config_.mesh) {
    const std::uint64_t v = config_.max_vertices;
    const std::uint64_t t = config_.max_triangles;
    spec.output(vertices, "vertices").produces_tensor(device_output(kFloat32, v * 12U));
    spec.output(normals, "normals").produces_tensor(device_output(kFloat32, v * 12U));
    if (color()) {
      spec.output(colors, "colors").produces_tensor(device_output(kUInt8, v * 3U));
    }
    spec.output(triangles, "triangles").produces_tensor(device_output(kUInt32, t * 12U));
    spec.output(mesh_counts, "mesh_counts")
        .produces_tensor(device_output(kUInt32, kMeshCountsSize * 4U));
  }
  if (config_.raycast) {
    // Mandatory: an unconnected camera_info fails compile().
    spec.input(camera_info, "camera_info");
    const auto pixels = static_cast<std::uint64_t>(config_.max_width) *
                        static_cast<std::uint64_t>(config_.max_height);
    spec.output(raycast_depth, "raycast_depth")
        .produces_tensor(device_output(kFloat32, pixels * 4U))
        .payload_options(image_payload());
    hm::declare_point_cloud_output(spec, raycast_cloud, "raycast_cloud",
                                   {.max_points = pixels, .colors = color(), .normals = true});
  }
}

holoscan::Contract TsdfMeshOp::contract() const {
  if (config_.raycast) {
    return hm::camera_info_contract(config_.camera_info, camera_info, pose);
  }
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{pose});
  return result;
}

void TsdfMeshOp::start() {
  poses_ = 0;
  info_.reset();
  intrinsics_.reset();
  if (config_.mesh) {
    // The scratch, now rather than on the first mesh: extract() then
    // allocates nothing. Fresh buffers wait for no stream.
    if (const cudaError_t status = extractor_->prepare(*volume(), nullptr); status != cudaSuccess) {
      throw std::runtime_error(
          op_name() + ": allocating the mesh scratch failed: " + cudaGetErrorString(status));
    }
  }
}

void TsdfMeshOp::stop() {
  if (config_.mesh) {
    // Frees the scratch; cudaFree waits for the work that may still use it.
    extractor_ = std::make_unique<MeshExtractor>(config_.max_vertices, config_.max_triangles);
  }
  VolumeOp::stop();
}

holoscan::expected<void, holoscan::Error> TsdfMeshOp::compute(holoscan::ExecutionContext& context) {
  bind(context);
  auto sample = pose.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  // The camera to ray cast through.
  const hm::CameraInfo* camera = nullptr;
  if (config_.raycast) {
    auto info = info_.update(camera_info);
    if (!info) {
      return holoscan::make_unexpected(std::move(info).error());
    }
    if (*info == nullptr) {
      return {};  // No camera info yet: this pose is skipped.
    }
    camera = *info;
    if (info_.changed() || !intrinsics_) {
      intrinsics_.reset();
      auto intrinsics = detail::pinhole_intrinsics(*camera, op_name());
      if (!intrinsics) {
        return holoscan::make_unexpected(std::move(intrinsics).error());
      }
      intrinsics_ = *intrinsics;
    }
    if (std::cmp_greater(camera->width, config_.max_width) ||
        std::cmp_greater(camera->height, config_.max_height)) {
      return holoscan::make_unexpected(invalid(
          "camera info '" + camera->header.frame_id + "' is for " + std::to_string(camera->width) +
          "x" + std::to_string(camera->height) + " images, larger than the configured " +
          std::to_string(config_.max_width) + "x" + std::to_string(config_.max_height)));
    }
  }
  auto stream = stream_for(context);
  if (!stream) {
    return holoscan::make_unexpected(std::move(stream).error());
  }
  auto camera_pose = read_pose(sample->data, "pose", *stream);
  if (!camera_pose) {
    return holoscan::make_unexpected(std::move(camera_pose).error());
  }
  ++poses_;
  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*sample);
  const bool mesh_now = config_.mesh && poses_ % config_.mesh_every == 0U;

  OutputTensor out_vertices;
  OutputTensor out_normals;
  OutputTensor out_colors;
  OutputTensor out_triangles;
  OutputTensor out_counts;
  if (mesh_now) {
    const std::array<std::int64_t, 2> vertex_shape{config_.max_vertices, 3};
    const std::array<std::int64_t, 2> triangle_shape{config_.max_triangles, 3};
    const std::array<std::int64_t, 1> count_shape{kMeshCountsSize};
    if (auto made = allocate(out_vertices, vertices, vertex_shape, kFloat32, *stream); !made) {
      return made;
    }
    if (auto made = allocate(out_normals, normals, vertex_shape, kFloat32, *stream); !made) {
      return made;
    }
    if (color()) {
      if (auto made = allocate(out_colors, colors, vertex_shape, kUInt8, *stream); !made) {
        return made;
      }
    }
    if (auto made = allocate(out_triangles, triangles, triangle_shape, kUInt32, *stream); !made) {
      return made;
    }
    if (auto made = allocate(out_counts, mesh_counts, count_shape, kUInt32, *stream); !made) {
      return made;
    }
    const MeshBuffers buffers{out_vertices.data<float>(), out_normals.data<float>(),
                              out_colors.data<std::uint8_t>(), out_triangles.data<std::uint32_t>(),
                              out_counts.data<std::uint32_t>()};
    if (const cudaError_t status = extractor_->extract(*volume(), buffers, *stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("mesh extraction", status));
    }
  }

  // The ray cast: one kernel writes the depth image and the cloud's points,
  // normals and colors.
  std::optional<holoscan::TensorOutputLoan> depth_loan;
  std::optional<holoscan::TensorOutputWriteGuard> depth_writer;
  std::optional<hm::PointCloudLoan> cloud;
  hm::PointCloudShape cloud_shape;
  if (config_.raycast) {
    const int width = static_cast<int>(camera->width);
    const int height = static_cast<int>(camera->height);
    const RaycastParams params{.intrinsics = *intrinsics_,
                               .width = width,
                               .height = height,
                               .depth_min = config_.raycast_depth_min,
                               .depth_max = config_.raycast_depth_max};
    auto loan = raycast_depth.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{height, width}, .dtype = kFloat32});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    depth_loan.emplace(std::move(*loan));
    auto writer = depth_loan->write(*stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    depth_writer.emplace(std::move(*writer));
    cloud_shape = hm::organized_cloud(height, width);
    auto allocated = hm::allocate_point_cloud(
        raycast_cloud, cloud_shape, {.colors = color(), .normals = true}, context.cuda_stream());
    if (!allocated) {
      return holoscan::make_unexpected(std::move(allocated).error());
    }
    cloud.emplace(std::move(*allocated));
    const auto raycast =
        [&](float* points, float* normals_out,
            std::uint8_t* colors_out) -> holoscan::expected<void, holoscan::Error> {
      if (const cudaError_t status = volume()->raycast(
              *camera_pose, params,
              RaycastImages{depth_writer->data_as<float>(), normals_out, colors_out, points},
              *stream);
          status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("ray cast", status));
      }
      return {};
    };
    using Guard = holoscan::TensorOutputWriteGuard;
    auto written = cloud->write_and_commit(
        hm::PointCloudFields::points, context.cuda_stream(),
        [&](const Guard& points) -> holoscan::expected<void, holoscan::Error> {
          return cloud->write_and_commit(
              hm::PointCloudFields::normals, context.cuda_stream(),
              [&](const Guard& normals_out) -> holoscan::expected<void, holoscan::Error> {
                if (!color()) {
                  return raycast(points.data_as<float>(), normals_out.data_as<float>(), nullptr);
                }
                return cloud->write_and_commit(hm::PointCloudFields::colors, context.cuda_stream(),
                                               [&](const Guard& colors_out) {
                                                 return raycast(points.data_as<float>(),
                                                                normals_out.data_as<float>(),
                                                                colors_out.data_as<std::uint8_t>());
                                               });
              });
        });
    if (!written) {
      return written;
    }
  }

  if (mesh_now) {
    if (auto sent = send(out_vertices, vertices, "vertices", emit); !sent) {
      return sent;
    }
    if (auto sent = send(out_normals, normals, "normals", emit); !sent) {
      return sent;
    }
    if (color()) {
      if (auto sent = send(out_colors, colors, "colors", emit); !sent) {
        return sent;
      }
    }
    if (auto sent = send(out_triangles, triangles, "triangles", emit); !sent) {
      return sent;
    }
    if (auto sent = send(out_counts, mesh_counts, "mesh_counts", emit); !sent) {
      return sent;
    }
  }
  if (config_.raycast) {
    if (auto committed = std::move(*depth_writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    depth_writer.reset();
    if (auto sent = raycast_depth.emit_tensor(
            std::move(*depth_loan),
            descriptor(static_cast<int>(camera->width), static_cast<int>(camera->height),
                       holoscan::schema::ImageEncoding_MONO32F),
            emit);
        !sent) {
      return emit_failure("raycast_depth", std::move(sent).error());
    }
    hm::Header header{.frame_id = camera->header.frame_id};
    if (sample->metadata.source_clock_id.valid()) {
      header.capture_timestamp_ns = sample->metadata.capture_timestamp_ns;
    }
    // Counting the hits would wait for the kernel: -1, not counted.
    if (auto sent =
            hm::emit_point_cloud(raycast_cloud, std::move(*cloud), header, cloud_shape, -1, emit);
        !sent) {
      return emit_failure("raycast_cloud", std::move(sent).error());
    }
  }
  return {};
}

}  // namespace holoscan::tsdf
