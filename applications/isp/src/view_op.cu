// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "view_op.hpp"

namespace isp_demo {
namespace {

using std::chrono_literals::operator""ns;
namespace isp = holoscan::isp;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

struct ViewParams {
  int width;
  int height;
  int format;
  std::size_t pitch;
  float black;
  float scale;  // 1 / (white - black).
};

// One raw sample, unpacked.
__device__ int sample(const unsigned char* raw, const ViewParams& p, int x, int y) {
  const unsigned char* row = raw + static_cast<std::size_t>(y) * p.pitch;
  switch (p.format) {
    case static_cast<int>(isp::RawFormat::kUint8):
      return row[x];
    case static_cast<int>(isp::RawFormat::kUint16):
      return reinterpret_cast<const std::uint16_t*>(row)[x];
    case static_cast<int>(isp::RawFormat::kRaw10): {
      const unsigned char* group = row + 5 * (x >> 2);
      return (group[x & 3] << 2) | ((group[4] >> (2 * (x & 3))) & 3);
    }
    default: {
      const unsigned char* group = row + 3 * (x >> 1);
      return (group[x & 1] << 4) | ((group[2] >> (4 * (x & 1))) & 15);
    }
  }
}

__global__ void view_kernel(ViewParams p, const unsigned char* raw, const unsigned char* rgb,
                            unsigned char* out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= 2 * p.width || y >= p.height) {
    return;
  }
  unsigned char* o = out + (static_cast<std::size_t>(y) * 2 * p.width + x) * 3;
  if (x < p.width) {
    // The mosaic as gray, square-rooted so that its dark half shows.
    float v = (static_cast<float>(sample(raw, p, x, y)) - p.black) * p.scale;
    v = v < 0.0F ? 0.0F : v > 1.0F ? 1.0F : v;
    const auto g = static_cast<unsigned char>(sqrtf(v) * 255.0F + 0.5F);
    o[0] = o[1] = o[2] = g;
  } else {
    const unsigned char* i = rgb + (static_cast<std::size_t>(y) * p.width + x - p.width) * 3;
    o[0] = i[0];
    o[1] = i[1];
    o[2] = i[2];
  }
}

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "ViewOp: " + message};
}

}  // namespace

ViewOp::ViewOp(ViewConfig config) : config_(config) {}

void ViewOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(raw, "raw");
  spec.input(image, "image");
  const auto bytes = static_cast<std::uint64_t>(config_.width) * 2U *
                     static_cast<std::uint64_t>(config_.height) * 3U;
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
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{raw, image}}, .window = 0ns});
  result.input(raw, holoscan::Exact{});
  result.input(image, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> ViewOp::compute(holoscan::ExecutionContext& context) {
  auto raw_sample = raw.receive();
  if (!raw_sample) {
    return holoscan::make_unexpected(std::move(raw_sample).error());
  }
  auto image_sample = image.receive();
  if (!image_sample) {
    return holoscan::make_unexpected(std::move(image_sample).error());
  }
  const holoscan::schema::ImageT& r = raw_sample->data;
  const holoscan::schema::ImageT& i = image_sample->data;
  if (!r.data || !i.data || r.width != config_.width || r.height != config_.height ||
      i.width != config_.width || i.height != config_.height ||
      i.encoding != holoscan::schema::ImageEncoding_RGB8) {
    return holoscan::make_unexpected(invalid("expected the camera's raw frame and its RGB8 image"));
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(invalid("the runtime gave no CUDA stream"));
  }
  auto loan = view.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{config_.height, 2 * config_.width, 3}, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  const float white = static_cast<float>((1U << static_cast<unsigned>(config_.bit_depth)) - 1U);
  const ViewParams p{config_.width,
                     config_.height,
                     static_cast<int>(config_.format),
                     static_cast<std::size_t>(r.data->strides()[0]),
                     config_.black_level,
                     1.0F / (white - config_.black_level)};
  const dim3 block(32, 8);
  const dim3 grid((2 * config_.width + 31) / 32, (config_.height + 7) / 8);
  // The raw samples may be uint16: read them as bytes, not through data_as<>, which checks the
  // type.
  view_kernel<<<grid, block, 0, stream>>>(p, static_cast<const unsigned char*>(r.data->data()),
                                          i.data->data_as<unsigned char>(),
                                          writer->data_as<unsigned char>());
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return holoscan::make_unexpected(invalid(std::string("kernel: ") + cudaGetErrorString(status)));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = 2 * config_.width;
  descriptor.height = config_.height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  return view.emit_tensor(std::move(*loan), std::move(descriptor),
                          holoscan::sensor_io::derive_from(*image_sample));
}

}  // namespace isp_demo
