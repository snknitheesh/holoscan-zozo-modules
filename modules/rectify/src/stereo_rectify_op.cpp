// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/rectify/stereo_rectify_op.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "op_support.hpp"

namespace holoscan::rectify {
namespace {

using std::chrono_literals::operator""ns;

constexpr std::string_view kOp = "StereoRectifyOp";

messages::StereoCameraInfo read_calibration(const StereoRectifyConfig& config) {
  if (config.left_camera_info_path.empty() || config.right_camera_info_path.empty()) {
    throw std::invalid_argument(std::string(kOp) +
                                ": left_camera_info_path and right_camera_info_path are required");
  }
  return messages::read_stereo_camera_info_yaml(config.left_camera_info_path,
                                                config.right_camera_info_path);
}

}  // namespace

struct StereoRectifyOp::State {
  State(const messages::StereoCameraInfo& calibration, const StereoRectification& rectification)
      : left_model(left_remap(calibration, rectification)),
        right_model(right_remap(calibration, rectification)),
        camera_info(messages::to_message(rectification.camera_info)) {}

  RemapModel left_model;
  RemapModel right_model;
  detail::RemapperSlot left;
  detail::RemapperSlot right;
  detail::Staging left_staging;
  detail::Staging right_staging;
  // Declared after the remappers, so that it goes first, waiting for their work.
  std::optional<RemapQueue> queue;
  // The camera info sent with every pair, stamped with its capture time.
  messages::schema::StereoCameraInfoT camera_info;
  bool copy_logged{};
};

void validate_config(const StereoRectifyConfig& config) {
  check_available(config.backend);
  (void)compute_stereo_rectification(read_calibration(config), config.options);
}

StereoRectifyOp::StereoRectifyOp(StereoRectifyConfig config)
    : config_(std::move(config)),
      calibration_(read_calibration(config_)),
      rectification_(compute_stereo_rectification(calibration_, config_.options)),
      state_(std::make_unique<State>(calibration_, rectification_)) {
  check_available(config_.backend);
}

StereoRectifyOp::~StereoRectifyOp() = default;

void StereoRectifyOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(left, "left").queue_depth(2U);
  spec.input(right, "right").queue_depth(2U);
  detail::declare_output(spec, rectified_left, "rectified_left", rectification_.width,
                         rectification_.height);
  detail::declare_output(spec, rectified_right, "rectified_right", rectification_.width,
                         rectification_.height);
  spec.output(camera_info, "camera_info");
}

holoscan::Contract StereoRectifyOp::contract() const {
  // The two images of a pair carry the same capture time.
  holoscan::Contract result;
  const std::vector<holoscan::PortRef> ports{left, right};
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
  result.input(left, holoscan::Exact{});
  result.input(right, holoscan::Exact{});
  return result;
}

void StereoRectifyOp::stop() {
  State& state = *state_;
  state.queue.reset();
  state.left.reset();
  state.right.reset();
  state.left_staging.release();
  state.right_staging.release();
}

holoscan::expected<void, holoscan::Error> StereoRectifyOp::compute(
    holoscan::ExecutionContext& context) {
  auto left_sample = left.receive();
  if (!left_sample) return holoscan::make_unexpected(std::move(left_sample).error());
  auto right_sample = right.receive();
  if (!right_sample) return holoscan::make_unexpected(std::move(right_sample).error());
  auto left_input = detail::read_input(left_sample->data, kOp, "left");
  if (!left_input) return holoscan::make_unexpected(std::move(left_input).error());
  auto right_input = detail::read_input(right_sample->data, kOp, "right");
  if (!right_input) return holoscan::make_unexpected(std::move(right_input).error());
  if (left_input->format != right_input->format) {
    return holoscan::make_unexpected(
        detail::invalid(kOp, "the left and right images have different encodings"));
  }
  const messages::CameraInfo& camera = calibration_.left;
  for (const detail::InputImage* input : {&*left_input, &*right_input}) {
    if (input->width != static_cast<int>(camera.width) ||
        input->height != static_cast<int>(camera.height)) {
      return holoscan::make_unexpected(detail::invalid(
          kOp, "an image is " + std::to_string(input->width) + "x" + std::to_string(input->height) +
                   ", the cameras " + std::to_string(camera.width) + "x" +
                   std::to_string(camera.height)));
    }
  }
  auto stream = detail::stream_of(context, kOp);
  if (!stream) return holoscan::make_unexpected(std::move(stream).error());
  State& state = *state_;
  const int width = rectification_.width;
  const int height = rectification_.height;
  auto left_remapper = state.left.get(config_.backend, state.left_model, width, height,
                                      left_input->format, config_.interpolation, kOp);
  if (!left_remapper) return holoscan::make_unexpected(std::move(left_remapper).error());
  // The right eye on the left eye's backend, which for auto is what it chose:
  // the two eyes of a pair never differ in latency or quality, and never mix
  // CUDA kernels with VPI's work on the one stream.
  auto right_remapper = state.right.get((*left_remapper)->backend(), state.right_model, width,
                                        height, right_input->format, config_.interpolation, kOp);
  if (!right_remapper) return holoscan::make_unexpected(std::move(right_remapper).error());
  detail::OutputImage left_output;
  detail::OutputImage right_output;
  if (auto allocated =
          detail::allocate_output(rectified_left, *left_input, width, height, *stream, left_output);
      !allocated) {
    return allocated;
  }
  if (auto allocated = detail::allocate_output(rectified_right, *right_input, width, height,
                                               *stream, right_output);
      !allocated) {
    return allocated;
  }
  auto left_source =
      state.left_staging.on_device(*left_input, *left_sample->data.data, *stream, kOp);
  if (!left_source) return holoscan::make_unexpected(std::move(left_source).error());
  auto right_source =
      state.right_staging.on_device(*right_input, *right_sample->data.data, *stream, kOp);
  if (!right_source) {
    // The left image may already be read from its producer's pinned memory.
    (void)detail::release_inputs(left_source->wait_before_return, false, *stream, kOp);
    return holoscan::make_unexpected(std::move(right_source).error());
  }
  const std::size_t copied_bytes =
      (left_source->copied ? left_input->row_bytes() * left_input->height : 0U) +
      (right_source->copied ? right_input->row_bytes() * right_input->height : 0U);
  if (copied_bytes != 0 && !state.copy_logged) {
    state.copy_logged = true;
    detail::log_host_copy(kOp, (*left_remapper)->name(), copied_bytes,
                          state.left_staging.reads_pinned_in_place());
  }
  bool waited = false;
  try {
    // One queue for both eyes, so that VPI's work for the pair is waited for once.
    RemapQueue& queue = detail::queue_on(state.queue, *stream);
    (*left_remapper)->submit(left_source->image, left_output.image, queue);
    (*right_remapper)->submit(right_source->image, right_output.image, queue);
    waited = queue.finish();
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(detail::remap_failed(state.queue, *stream, kOp, error));
  }
  if (auto released = detail::release_inputs(
          left_source->wait_before_return || right_source->wait_before_return, waited, *stream,
          kOp);
      !released) {
    return released;
  }

  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*left_sample);
  if (auto sent = detail::send_output(rectified_left, left_output, left_sample->data, emit);
      !sent) {
    return sent;
  }
  if (auto sent = detail::send_output(rectified_right, right_output, right_sample->data, emit);
      !sent) {
    return sent;
  }
  messages::schema::StereoCameraInfoT& info = state.camera_info;
  for (holoscan::schema::HeaderT* header :
       {info.header.get(), info.left->header.get(), info.right->header.get()}) {
    detail::stamp(*header, emit);
  }
  return camera_info.emit_ref(info, emit);
}

}  // namespace holoscan::rectify
