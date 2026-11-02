// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/denoise/denoise_op.hpp"

#include <cuda_runtime_api.h>

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "image_ports.hpp"

namespace holoscan::denoise {
namespace {

constexpr std::string_view kOp = "DenoiseOp";
constexpr int kMaxSide = 16384;

std::string_view auto_reason(const PlatformFacts& facts, const DenoiseParams& params) {
  if (!facts.vpi_compiled) return "this build has no vpi backend";
  if (!facts.vic_tnr) return "VPI cannot run TNR on a VIC here";
  if (!params.temporal || params.spatial != SpatialFilter::kNone) {
    return "the settings need the cuda backend (a spatial filter, or no temporal filter)";
  }
  return "VPI runs TNR on the VIC, which leaves the GPU free";
}

}  // namespace

void validate_config(const DenoiseOpConfig& config) {
  validate_params(config.params);
  check_available(config.backend);
  if (const std::string_view reason = unsupported_reason(config.backend, config.params);
      !reason.empty()) {
    throw std::invalid_argument(std::string(reason));
  }
  if (config.max_width < 1 || config.max_height < 1 || config.max_width > kMaxSide ||
      config.max_height > kMaxSide) {
    throw std::invalid_argument("max_width and max_height must be 1..16384");
  }
}

struct DenoiseOp::State {
  std::unique_ptr<Denoiser> denoiser;
  std::string start_error;
  detail::DeviceStaging staging;
};

DenoiseOp::DenoiseOp(DenoiseOpConfig config)
    : config_(std::move(config)), backend_(Backend::kCuda), state_(std::make_unique<State>()) {
  validate_config(config_);
  backend_ = config_.backend == Backend::kAuto ? resolve_auto(probe_platform(), config_.params)
                                               : config_.backend;
}

DenoiseOp::~DenoiseOp() = default;

void DenoiseOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots: one frame in compute() while the next one arrives.
  spec.input(image, "image").queue_depth(2U);
  spec.output(denoised, "denoised")
      .produces_tensor(detail::image_output_spec(config_.max_width, config_.max_height))
      .payload_options(detail::image_payload());
}

holoscan::Contract DenoiseOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void DenoiseOp::start() {
  if (config_.backend == Backend::kAuto) {
    HOLOSCAN_LOG_INFO("DenoiseOp: backend auto chose {}: {}", backend_name(backend_),
                      auto_reason(probe_platform(), config_.params));
  }
  state_->start_error.clear();
  try {
    state_->denoiser = std::make_unique<Denoiser>(backend_, config_.params, config_.vpi);
  } catch (const std::exception& error) {
    // Reported by compute(), which can fail the frame.
    state_->start_error = error.what();
  }
}

void DenoiseOp::stop() {
  state_->denoiser.reset();
  state_->staging.release();
}

holoscan::expected<void, holoscan::Error> DenoiseOp::compute(holoscan::ExecutionContext& context) {
  State& state = *state_;
  if (!state.denoiser) {
    return holoscan::make_unexpected(
        detail::failure(kOp, "the denoiser could not start: " + state.start_error));
  }
  auto sample = image.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  const holoscan::schema::ImageT& input = sample->data;
  auto described = detail::describe_image(input, kOp, true, config_.max_width, config_.max_height);
  if (!described) {
    return holoscan::make_unexpected(std::move(described).error());
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    // Not reached in a compiled graph: the output is in device memory, so the
    // runtime always assigns a stream.
    return holoscan::make_unexpected(detail::failure(kOp, "the runtime gave no CUDA stream"));
  }
  const holoscan::Tensor& tensor = *input.data;
  if (auto ready = detail::wait_for_host_input(tensor); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  auto source = state.staging.on_device(*described, tensor, stream, kOp, "image");
  if (!source) {
    return holoscan::make_unexpected(std::move(source).error());
  }
  const PixelFormat format = described->view.format;
  const int width = described->view.width;
  const int height = described->view.height;
  auto loan = denoised.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = detail::image_shape(format, width, height), .dtype = DLDataType{kDLUInt, 8U, 1U}});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  try {
    state.denoiser->process(
        *source, detail::packed_view(format, width, height, writer->data_as<std::uint8_t>()),
        stream);
  } catch (const std::invalid_argument& error) {
    return holoscan::make_unexpected(detail::invalid(kOp, error.what()));
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(detail::failure(kOp, error.what()));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return denoised.emit_tensor(std::move(*loan),
                              detail::image_descriptor(format, width, height, input),
                              holoscan::sensor_io::derive_from(*sample));
}

}  // namespace holoscan::denoise
