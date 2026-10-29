// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/rectify/undistort_op.hpp"

#include <cuda_runtime_api.h>

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/sensor_io/provenance.hpp>

#include "op_support.hpp"

namespace holoscan::rectify {
namespace {

constexpr std::string_view kOp = "UndistortOp";

}  // namespace

struct UndistortOp::State {
  explicit State(const Undistortion& undistortion) : model(undistortion_remap(undistortion)) {}

  RemapModel model;
  detail::RemapperSlot remapper;
  detail::Staging staging;
  bool copy_logged{};
};

void validate_config(const UndistortConfig& config) {
  check_available(config.backend);
  (void)compute_undistortion(config.camera, config.options);
}

UndistortOp::UndistortOp(UndistortConfig config)
    : config_(std::move(config)),
      undistortion_(compute_undistortion(config_.camera, config_.options)),
      state_(std::make_unique<State>(undistortion_)) {
  check_available(config_.backend);
}

UndistortOp::~UndistortOp() = default;

void UndistortOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots: one image in compute() while the next one arrives.
  spec.input(image, "image").queue_depth(2U);
  detail::declare_output(spec, undistorted, "undistorted", undistortion_.output.width,
                         undistortion_.output.height);
}

holoscan::Contract UndistortOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void UndistortOp::stop() {
  state_->remapper.reset();
  state_->staging.release();
}

holoscan::expected<void, holoscan::Error> UndistortOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) return holoscan::make_unexpected(std::move(sample).error());
  auto input = detail::read_input(sample->data, kOp, "input");
  if (!input) return holoscan::make_unexpected(std::move(input).error());
  const CameraModel& camera = undistortion_.input;
  if (input->width != camera.width || input->height != camera.height) {
    return holoscan::make_unexpected(detail::invalid(
        kOp, "the image is " + std::to_string(input->width) + "x" + std::to_string(input->height) +
                 ", the camera " + std::to_string(camera.width) + "x" +
                 std::to_string(camera.height)));
  }
  auto stream = detail::stream_of(context, kOp);
  if (!stream) return holoscan::make_unexpected(std::move(stream).error());
  State& state = *state_;
  const CameraModel& out = undistortion_.output;
  auto remapper = state.remapper.get(config_.backend, state.model, out.width, out.height,
                                     input->format, config_.interpolation, kOp);
  if (!remapper) return holoscan::make_unexpected(std::move(remapper).error());
  if (!input->on_device && !state.copy_logged) {
    state.copy_logged = true;
    detail::log_host_copy(kOp, (*remapper)->name(), input->row_bytes() * input->height);
  }
  auto source = state.staging.on_device(*input, *sample->data.data, *stream, kOp);
  if (!source) return holoscan::make_unexpected(std::move(source).error());
  detail::OutputImage output;
  if (auto allocated =
          detail::allocate_output(undistorted, *input, out.width, out.height, *stream, output);
      !allocated) {
    return allocated;
  }
  try {
    (*remapper)->submit(*source, output.image, *stream);
    (*remapper)->finish();
  } catch (const std::exception& error) {
    // Nothing queued may still read the input after compute() returns.
    cudaStreamSynchronize(*stream);
    return holoscan::make_unexpected(dynamic_cast<const std::invalid_argument*>(&error) != nullptr
                                         ? detail::invalid(kOp, error.what())
                                         : detail::failure(kOp, error.what()));
  }
  return detail::send_output(undistorted, output, sample->data,
                             holoscan::sensor_io::derive_from(*sample));
}

}  // namespace holoscan::rectify
