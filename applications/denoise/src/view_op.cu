// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "view_op.hpp"

namespace denoise_demo {
namespace {

using std::chrono_literals::operator""ns;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

struct Sources {
  const unsigned char* image[6];  // noisy, denoised, bracket0..2, fused.
};

__global__ void view_kernel(Sources s, int width, int height, unsigned char* out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int out_w = 2 * width;
  if (x >= out_w || y >= height + height / 2) return;
  unsigned char* o = out + (static_cast<std::size_t>(y) * out_w + x) * 3;
  if (y < height) {
    // Noisy | denoised.
    const unsigned char* i =
        s.image[x < width ? 0 : 1] + (static_cast<std::size_t>(y) * width + x % width) * 3;
    o[0] = i[0];
    o[1] = i[1];
    o[2] = i[2];
    return;
  }
  // Brackets and fused at half size, each the mean of a 2x2 block.
  const int tile = x / (width / 2);
  if (tile > 3) {
    o[0] = o[1] = o[2] = 0;
    return;
  }
  const int sx = 2 * (x % (width / 2));
  const int sy = 2 * (y - height);
  const unsigned char* image = s.image[2 + tile];
  for (int c = 0; c < 3; ++c) {
    int sum = 0;
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        sum += image[(static_cast<std::size_t>(min(sy + dy, height - 1)) * width +
                      min(sx + dx, width - 1)) *
                         3 +
                     c];
      }
    }
    o[c] = static_cast<unsigned char>((sum + 2) / 4);
  }
}

}  // namespace

ViewOp::ViewOp(ViewConfig config) : config_(config) {
  if (config_.width < 2 || config_.height < 2 || config_.width % 2 != 0 ||
      config_.height % 2 != 0) {
    throw std::invalid_argument("ViewOp needs an even width and height");
  }
}

void ViewOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(noisy, "noisy");
  spec.input(denoised, "denoised");
  spec.input(bracket0, "bracket0");
  spec.input(bracket1, "bracket1");
  spec.input(bracket2, "bracket2");
  spec.input(fused, "fused");
  const auto bytes = static_cast<std::uint64_t>(config_.width) * 2U *
                     static_cast<std::uint64_t>(config_.height + config_.height / 2) * 3U;
  spec.output(view, "view")
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

holoscan::Contract ViewOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                                         noisy, denoised, bracket0, bracket1, bracket2, fused}},
                                     .window = 0ns});
  result.input(noisy, holoscan::Exact{});
  result.input(denoised, holoscan::Exact{});
  result.input(bracket0, holoscan::Exact{});
  result.input(bracket1, holoscan::Exact{});
  result.input(bracket2, holoscan::Exact{});
  result.input(fused, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> ViewOp::compute(holoscan::ExecutionContext& context) {
  const std::array<holoscan::Input<holoscan::schema::ImageT>*, 6> ports{
      &noisy, &denoised, &bracket0, &bracket1, &bracket2, &fused};
  std::vector<holoscan::Sample<holoscan::schema::ImageT>> samples;
  samples.reserve(ports.size());
  Sources sources{};
  for (std::size_t k = 0; k < ports.size(); ++k) {
    auto sample = ports[k]->receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    const holoscan::schema::ImageT& image = sample->data;
    if (!image.data || image.width != config_.width || image.height != config_.height ||
        image.encoding != holoscan::schema::ImageEncoding_RGB8) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument, "ViewOp: expected RGB8 images of the scene"});
    }
    sources.image[k] = image.data->data_as<unsigned char>();
    samples.push_back(std::move(*sample));
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "ViewOp: the runtime gave no CUDA stream"});
  }
  const int out_h = config_.height + config_.height / 2;
  auto loan = view.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{out_h, 2 * config_.width, 3}, .dtype = kUInt8});
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  auto writer = loan->write(stream);
  if (!writer) return holoscan::make_unexpected(std::move(writer).error());
  const dim3 block(32, 8);
  const dim3 grid((2 * config_.width + 31) / 32, (out_h + 7) / 8);
  view_kernel<<<grid, block, 0, stream>>>(sources, config_.width, config_.height,
                                          writer->data_as<unsigned char>());
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        std::string("ViewOp: kernel: ") + cudaGetErrorString(status)});
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = 2 * config_.width;
  descriptor.height = out_h;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  return view.emit_tensor(std::move(*loan), std::move(descriptor),
                          holoscan::sensor_io::derive_from(samples.front()));
}

}  // namespace denoise_demo
