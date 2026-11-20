// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "scene_source_op.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace denoise_demo {
namespace {

namespace dn = holoscan::denoise;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
// One bit per output: noisy, bracket0, bracket1, bracket2.
constexpr std::uint32_t kAllSent = 0xFU;

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "SceneSourceOp: " + message};
}

// Whether `error` says a consumer is still busy with earlier frames: an emit
// refused before delivery, or every slot of the output's pool still held.
bool consumers_busy(const holoscan::Error& error) {
  return error.code == holoscan::ErrorCode::kBackpressured ||
         error.code == holoscan::ErrorCode::kResourceExhausted;
}

}  // namespace

SceneSourceOp::SceneSourceOp(SceneSourceConfig config) : config_(std::move(config)) {
  const dn::LowLightScene& scene = config_.scene;
  if (scene.format != dn::PixelFormat::kRgb8 || scene.width < 1 || scene.height < 1 ||
      config_.hdr.width != scene.width || config_.hdr.height != scene.height ||
      config_.hdr.channels != 3) {
    throw std::invalid_argument("the source renders RGB8 scenes of one size");
  }
  if (config_.count == 0 || config_.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the source needs a positive count and period");
  }
}

void SceneSourceOp::setup(holoscan::OperatorSpec& spec) {
  const auto bytes = static_cast<std::uint64_t>(config_.scene.width) *
                     static_cast<std::uint64_t>(config_.scene.height) * 3U;
  const holoscan::TensorOutputSpec tensor{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = kUInt8,
              .rank = 3U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
  const holoscan::PayloadOptions image{.schema =
                                           holoscan::schema_identity<holoscan::schema::ImageT>()};
  spec.output(noisy, "noisy").produces_tensor(tensor).payload_options(image);
  spec.output(bracket0, "bracket0").produces_tensor(tensor).payload_options(image);
  spec.output(bracket1, "bracket1").produces_tensor(tensor).payload_options(image);
  spec.output(bracket2, "bracket2").produces_tensor(tensor).payload_options(image);
}

holoscan::Contract SceneSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SceneSourceOp::start() {
  emitted_ = 0;
  capture_.reset();
  sent_ = 0U;
}

holoscan::expected<void, holoscan::Error> SceneSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the checker completes the run.
  }
  if (config_.finished && config_.finished->load(std::memory_order_acquire) < emitted_) {
    return {};  // The previous frame is still in flight.
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(failure("the runtime gave no CUDA stream"));
  }
  const int width = config_.scene.width;
  const int height = config_.scene.height;
  const int frame = static_cast<int>(emitted_);
  const std::array<std::int64_t, 3> shape{height, width, 3};
  if (!capture_) {
    capture_ = context.activation_time();  // Kept until every output is out.
  }
  const holoscan::EmitOptions options{.capture_time = *capture_, .frame_id = emitted_ + 1U};
  // Renders one image into a loan of `port` and sends it.
  const auto send = [&](holoscan::Output<holoscan::schema::ImageT>& port,
                        const auto& render) -> holoscan::expected<void, holoscan::Error> {
    auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) return holoscan::make_unexpected(std::move(loan).error());
    auto writer = loan->write(stream);
    if (!writer) return holoscan::make_unexpected(std::move(writer).error());
    dn::ImageView view{.format = dn::PixelFormat::kRgb8, .width = width, .height = height};
    view.planes[0] = dn::Plane{.data = writer->data_as<std::uint8_t>(),
                               .pitch = static_cast<std::size_t>(width) * 3U};
    if (const cudaError_t status = render(view); status != cudaSuccess) {
      return holoscan::make_unexpected(
          failure(std::string("render: ") + cudaGetErrorString(status)));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = width;
    descriptor.height = height;
    descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
    descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
    descriptor.header->frame_id = "synthetic_camera";
    descriptor.header->capture_timestamp_ns = capture_->timestamp_ns;
    return port.emit_tensor(std::move(*loan), std::move(descriptor), options);
  };
  // Sends output `bit` unless it already went out; a busy consumer leaves it
  // for the next tick, any other failure fails the run.
  bool busy = false;
  const auto send_once = [&](std::uint32_t bit, holoscan::Output<holoscan::schema::ImageT>& port,
                             const auto& render) -> holoscan::expected<void, holoscan::Error> {
    if ((sent_ & bit) != 0U || busy) return {};
    auto sent = send(port, render);
    if (sent) {
      sent_ |= bit;
      return {};
    }
    if (consumers_busy(sent.error())) {
      busy = true;
      return {};
    }
    return sent;
  };
  if (auto sent = send_once(1U, noisy,
                            [&](const dn::ImageView& view) {
                              return dn::render_noisy(config_.scene, config_.noise, frame, view,
                                                      stream);
                            });
      !sent) {
    return sent;
  }
  const std::array<holoscan::Output<holoscan::schema::ImageT>*, 3> brackets{&bracket0, &bracket1,
                                                                            &bracket2};
  for (int k = 0; k < 3; ++k) {
    if (auto sent = send_once(2U << k, *brackets[k],
                              [&](const dn::ImageView& view) {
                                return dn::render_bracket(config_.hdr, frame,
                                                          dn::kBracketExposures[k], view, stream);
                              });
        !sent) {
      return sent;
    }
  }
  if (sent_ != kAllSent) {
    return {};  // The rest of this frame goes on the next tick.
  }
  sent_ = 0U;
  capture_.reset();
  ++emitted_;
  return {};
}

}  // namespace denoise_demo
