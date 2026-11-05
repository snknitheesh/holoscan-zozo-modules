// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/rectify/undistort_op.hpp"

#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "op_support.hpp"

namespace holoscan::rectify {
namespace {

constexpr std::string_view kOp = "UndistortOp";

messages::CameraInfo read_calibration(const UndistortConfig& config) {
  if (config.camera_info_path.empty()) {
    throw std::invalid_argument(std::string(kOp) + ": camera_info_path is required");
  }
  return messages::read_camera_info_yaml(config.camera_info_path);
}

}  // namespace

struct UndistortOp::State {
  explicit State(const Undistortion& undistortion)
      : model(undistortion_remap(undistortion)),
        camera_info(messages::to_message(undistortion.output)) {}

  RemapModel model;
  detail::RemapperSlot remapper;
  detail::Staging staging;
  // Declared after the remapper, so that it goes first, waiting for its work.
  std::optional<RemapQueue> queue;
  // The camera info sent with every image, stamped with its capture time.
  messages::schema::CameraInfoT camera_info;
  bool copy_logged{};
};

void validate_config(const UndistortConfig& config) {
  check_available(config.backend);
  (void)compute_undistortion(read_calibration(config), config.options);
}

UndistortOp::UndistortOp(UndistortConfig config)
    : config_(std::move(config)),
      undistortion_(compute_undistortion(read_calibration(config_), config_.options)),
      state_(std::make_unique<State>(undistortion_)) {
  check_available(config_.backend);
}

UndistortOp::~UndistortOp() = default;

void UndistortOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots: one image in compute() while the next one arrives.
  spec.input(image, "image").queue_depth(2U);
  detail::declare_output(spec, undistorted, "undistorted",
                         static_cast<int>(undistortion_.output.width),
                         static_cast<int>(undistortion_.output.height));
  spec.output(camera_info, "camera_info");
}

holoscan::Contract UndistortOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void UndistortOp::stop() {
  state_->queue.reset();
  state_->remapper.reset();
  state_->staging.release();
}

holoscan::expected<void, holoscan::Error> UndistortOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) return holoscan::make_unexpected(std::move(sample).error());
  auto input = detail::read_input(sample->data, kOp, "input");
  if (!input) return holoscan::make_unexpected(std::move(input).error());
  const messages::CameraInfo& camera = undistortion_.input;
  if (input->width != static_cast<int>(camera.width) ||
      input->height != static_cast<int>(camera.height)) {
    return holoscan::make_unexpected(detail::invalid(
        kOp, "the image is " + std::to_string(input->width) + "x" + std::to_string(input->height) +
                 ", the camera " + std::to_string(camera.width) + "x" +
                 std::to_string(camera.height)));
  }
  auto stream = detail::stream_of(context, kOp);
  if (!stream) return holoscan::make_unexpected(std::move(stream).error());
  State& state = *state_;
  const int out_width = static_cast<int>(undistortion_.output.width);
  const int out_height = static_cast<int>(undistortion_.output.height);
  auto remapper = state.remapper.get(config_.backend, state.model, out_width, out_height,
                                     input->format, config_.interpolation, kOp);
  if (!remapper) return holoscan::make_unexpected(std::move(remapper).error());
  detail::OutputImage output;
  if (auto allocated =
          detail::allocate_output(undistorted, *input, out_width, out_height, *stream, output);
      !allocated) {
    return allocated;
  }
  auto source = state.staging.on_device(*input, *sample->data.data, *stream, kOp);
  if (!source) return holoscan::make_unexpected(std::move(source).error());
  if (source->copied && !state.copy_logged) {
    state.copy_logged = true;
    detail::log_host_copy(kOp, (*remapper)->name(), input->row_bytes() * input->height,
                          state.staging.reads_pinned_in_place());
  }
  bool waited = false;
  try {
    RemapQueue& queue = detail::queue_on(state.queue, *stream);
    (*remapper)->submit(source->image, output.image, queue);
    waited = queue.finish();
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(detail::remap_failed(state.queue, *stream, kOp, error));
  }
  if (auto released = detail::release_inputs(source->wait_before_return, waited, *stream, kOp);
      !released) {
    return released;
  }
  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*sample);
  if (auto sent = detail::send_output(undistorted, output, sample->data, emit); !sent) {
    return sent;
  }
  detail::stamp(*state.camera_info.header, emit);
  return camera_info.emit_ref(state.camera_info, emit);
}

}  // namespace holoscan::rectify
