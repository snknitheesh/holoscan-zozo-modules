// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/denoise/exposure_fusion_op.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "image_ports.hpp"

namespace holoscan::denoise {
namespace {

constexpr std::string_view kOp = "ExposureFusionOp";
constexpr int kMaxSide = 16384;
constexpr std::array<std::string_view, 3> kBracketNames{"bracket0", "bracket1", "bracket2"};

PixelFormat tone_mapped_format(HdrFormat format) {
  const int channels = hdr_channels(format);
  return channels == 1   ? PixelFormat::kMono8
         : channels == 3 ? PixelFormat::kRgb8
                         : PixelFormat::kRgba8;
}

}  // namespace

void validate_config(const ExposureFusionOpConfig& config) {
  if (config.mode != FusionMode::kFusion && config.mode != FusionMode::kToneMap) {
    throw std::invalid_argument("mode must be fusion or tone map");
  }
  if (config.mode == FusionMode::kFusion) {
    if (config.brackets != 2 && config.brackets != 3) {
      throw std::invalid_argument("brackets must be 2 or 3");
    }
    if (config.match_window < std::chrono::nanoseconds::zero() ||
        config.match_window > std::chrono::seconds(1)) {
      throw std::invalid_argument("match_window must be 0..1 s");
    }
    validate_fusion_params(config.fusion);
  } else {
    validate_tone_map_params(config.tone_map);
  }
  if (config.max_width < 1 || config.max_height < 1 || config.max_width > kMaxSide ||
      config.max_height > kMaxSide) {
    throw std::invalid_argument("max_width and max_height must be 1..16384");
  }
}

struct ExposureFusionOp::State {
  std::unique_ptr<ExposureFusion> fusion;
  std::unique_ptr<ToneMapper> tone_mapper;
  std::array<detail::DeviceStaging, 3> staging;
};

ExposureFusionOp::ExposureFusionOp(ExposureFusionOpConfig config)
    : config_(std::move(config)), state_(std::make_unique<State>()) {
  validate_config(config_);
}

ExposureFusionOp::~ExposureFusionOp() = default;

void ExposureFusionOp::setup(holoscan::OperatorSpec& spec) {
  // Optional ports exist only when configured: an unconnected input would never
  // be satisfied. Two slots each: one set in compute() while the next arrives.
  if (config_.mode == FusionMode::kFusion) {
    spec.input(bracket0, "bracket0").queue_depth(2U);
    spec.input(bracket1, "bracket1").queue_depth(2U);
    if (config_.brackets == 3) spec.input(bracket2, "bracket2").queue_depth(2U);
  } else {
    spec.input(hdr, "hdr").queue_depth(2U);
  }
  spec.output(image, "image")
      .produces_tensor(detail::image_output_spec(config_.max_width, config_.max_height))
      .payload_options(detail::image_payload());
}

holoscan::Contract ExposureFusionOp::contract() const {
  holoscan::Contract result;
  if (config_.mode == FusionMode::kToneMap) {
    result.trigger(holoscan::OnEach{hdr});
    return result;
  }
  // One bracket of each exposure, by capture time.
  std::vector<holoscan::PortRef> ports{bracket0, bracket1};
  if (config_.brackets == 3) ports.emplace_back(bracket2);
  result.trigger(
      holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = config_.match_window});
  result.input(bracket0, holoscan::Exact{});
  result.input(bracket1, holoscan::Exact{});
  if (config_.brackets == 3) result.input(bracket2, holoscan::Exact{});
  return result;
}

void ExposureFusionOp::start() {
  // Settings were validated at construction, so these do not throw.
  if (config_.mode == FusionMode::kFusion) {
    state_->fusion = std::make_unique<ExposureFusion>(config_.fusion);
  } else {
    state_->tone_mapper = std::make_unique<ToneMapper>(config_.tone_map);
  }
}

void ExposureFusionOp::stop() {
  state_->fusion.reset();
  state_->tone_mapper.reset();
  for (detail::DeviceStaging& staging : state_->staging) staging.release();
}

holoscan::expected<void, holoscan::Error> ExposureFusionOp::compute(
    holoscan::ExecutionContext& context) {
  State& state = *state_;
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    // Not reached in a compiled graph: the output is in device memory.
    return holoscan::make_unexpected(detail::failure(kOp, "the runtime gave no CUDA stream"));
  }
  const auto emit_options = [](const auto& sample) {
    return holoscan::sensor_io::derive_from(sample);
  };

  if (config_.mode == FusionMode::kToneMap) {
    auto sample = hdr.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    const holoscan::schema::ImageT& input = sample->data;
    auto described = detail::describe_hdr(input, kOp, config_.max_width, config_.max_height);
    if (!described) return holoscan::make_unexpected(std::move(described).error());
    if (auto ready = detail::wait_for_host_input(*input.data); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    auto data = state.staging[0].on_device(*described, *input.data, stream, kOp, "hdr");
    if (!data) return holoscan::make_unexpected(std::move(data).error());
    HdrImageView view = described->view;
    view.data = *data;
    const PixelFormat format = tone_mapped_format(view.format);
    auto loan = image.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = detail::image_shape(format, view.width, view.height),
                                    .dtype = DLDataType{kDLUInt, 8U, 1U}});
    if (!loan) return holoscan::make_unexpected(std::move(loan).error());
    auto writer = loan->write(stream);
    if (!writer) return holoscan::make_unexpected(std::move(writer).error());
    try {
      state.tone_mapper->map(
          view,
          detail::packed_view(format, view.width, view.height, writer->data_as<std::uint8_t>()),
          stream);
    } catch (const std::invalid_argument& error) {
      return holoscan::make_unexpected(detail::invalid(kOp, error.what()));
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(detail::failure(kOp, error.what()));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    return image.emit_tensor(std::move(*loan),
                             detail::image_descriptor(format, view.width, view.height, input),
                             emit_options(*sample));
  }

  // Fusion: every bracket is received, so that each is consumed once.
  std::array<holoscan::Input<holoscan::schema::ImageT>*, 3> ports{&bracket0, &bracket1, &bracket2};
  const auto count = static_cast<std::size_t>(config_.brackets);
  std::vector<holoscan::Sample<holoscan::schema::ImageT>> samples;
  samples.reserve(count);
  for (std::size_t k = 0; k < count; ++k) {
    auto sample = ports[k]->receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    samples.push_back(std::move(*sample));
  }
  std::vector<ImageView> views;
  for (std::size_t k = 0; k < count; ++k) {
    const holoscan::schema::ImageT& input = samples[k].data;
    auto described =
        detail::describe_image(input, kOp, false, config_.max_width, config_.max_height);
    if (!described) return holoscan::make_unexpected(std::move(described).error());
    if (auto ready = detail::wait_for_host_input(*input.data); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    auto view = state.staging[k].on_device(*described, *input.data, stream, kOp, kBracketNames[k]);
    if (!view) return holoscan::make_unexpected(std::move(view).error());
    views.push_back(*view);
  }
  const ImageView& first = views.front();
  for (const ImageView& view : views) {
    if (view.format != first.format || view.width != first.width || view.height != first.height) {
      return holoscan::make_unexpected(
          detail::invalid(kOp, "the brackets must share format and size"));
    }
  }
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = detail::image_shape(first.format, first.width, first.height),
      .dtype = DLDataType{kDLUInt, 8U, 1U}});
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  auto writer = loan->write(stream);
  if (!writer) return holoscan::make_unexpected(std::move(writer).error());
  try {
    state.fusion->fuse(views,
                       detail::packed_view(first.format, first.width, first.height,
                                           writer->data_as<std::uint8_t>()),
                       stream);
  } catch (const std::invalid_argument& error) {
    return holoscan::make_unexpected(detail::invalid(kOp, error.what()));
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(detail::failure(kOp, error.what()));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return image.emit_tensor(
      std::move(*loan),
      detail::image_descriptor(first.format, first.width, first.height, samples.front().data),
      emit_options(samples.front()));
}

}  // namespace holoscan::denoise
