// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/tsdf/tsdf_mesh_op.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace holoscan::tsdf {
namespace {

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
    validate_params(config.raycast_params);
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
  spec.input(pose, "pose").queue_depth(2U);
  // Optional ports exist only when configured: an emit into an unconnected
  // output fails.
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
    const auto pixels = static_cast<std::uint64_t>(config_.raycast_params.width) *
                        static_cast<std::uint64_t>(config_.raycast_params.height);
    spec.output(raycast_depth, "raycast_depth")
        .produces_tensor(device_output(kFloat32, pixels * 4U))
        .payload_options(image_payload());
    spec.output(raycast_normals, "raycast_normals")
        .produces_tensor(device_output(kFloat32, pixels * 12U));
    if (color()) {
      spec.output(raycast_colors, "raycast_colors")
          .produces_tensor(device_output(kUInt8, pixels * 3U))
          .payload_options(image_payload());
    }
  }
}

holoscan::Contract TsdfMeshOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{pose});
  return result;
}

void TsdfMeshOp::start() { poses_ = 0; }

void TsdfMeshOp::stop() {
  if (config_.mesh) {
    // Frees the scratch; cudaFree waits for the work that may still use it.
    extractor_ = std::make_unique<MeshExtractor>(config_.max_vertices, config_.max_triangles);
  }
  VolumeOp::stop();
}

holoscan::expected<void, holoscan::Error> TsdfMeshOp::compute(holoscan::ExecutionContext& context) {
  auto sample = pose.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
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

  std::optional<holoscan::TensorOutputLoan> depth_loan;
  std::optional<holoscan::TensorOutputWriteGuard> depth_writer;
  std::optional<holoscan::TensorOutputLoan> color_loan;
  std::optional<holoscan::TensorOutputWriteGuard> color_writer;
  OutputTensor out_ray_normals;
  const RaycastParams& rp = config_.raycast_params;
  if (config_.raycast) {
    const std::array<std::int64_t, 2> image_shape{rp.height, rp.width};
    const std::array<std::int64_t, 3> normal_shape{rp.height, rp.width, 3};
    auto loan = raycast_depth.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = image_shape, .dtype = kFloat32});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    depth_loan.emplace(std::move(*loan));
    auto writer = depth_loan->write(*stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    depth_writer.emplace(std::move(*writer));
    if (auto made = allocate(out_ray_normals, raycast_normals, normal_shape, kFloat32, *stream);
        !made) {
      return made;
    }
    if (color()) {
      auto rgb_loan = raycast_colors.allocate_tensor(
          holoscan::TensorLoanRequest{.shape = normal_shape, .dtype = kUInt8});
      if (!rgb_loan) {
        return holoscan::make_unexpected(std::move(rgb_loan).error());
      }
      color_loan.emplace(std::move(*rgb_loan));
      auto rgb_writer = color_loan->write(*stream);
      if (!rgb_writer) {
        return holoscan::make_unexpected(std::move(rgb_writer).error());
      }
      color_writer.emplace(std::move(*rgb_writer));
    }
    const RaycastImages images{depth_writer->data_as<float>(), out_ray_normals.data<float>(),
                               color_writer ? color_writer->data_as<std::uint8_t>() : nullptr};
    if (const cudaError_t status = volume()->raycast(*camera_pose, rp, images, *stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("ray cast", status));
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
            descriptor(rp.width, rp.height, holoscan::schema::ImageEncoding_MONO32F), emit);
        !sent) {
      return emit_failure("raycast_depth", std::move(sent).error());
    }
    if (auto sent = send(out_ray_normals, raycast_normals, "raycast_normals", emit); !sent) {
      return sent;
    }
    if (color()) {
      if (auto committed = std::move(*color_writer).commit(); !committed) {
        return holoscan::make_unexpected(std::move(committed).error());
      }
      color_writer.reset();
      if (auto sent = raycast_colors.emit_tensor(
              std::move(*color_loan),
              descriptor(rp.width, rp.height, holoscan::schema::ImageEncoding_RGB8), emit);
          !sent) {
        return emit_failure("raycast_colors", std::move(sent).error());
      }
    }
  }
  return {};
}

}  // namespace holoscan::tsdf
