// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/rectify/stereo_rectify_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/sensor_io/provenance.hpp>

#include "op_support.hpp"

namespace holoscan::rectify {
namespace {

using std::chrono_literals::operator""ns;

constexpr std::string_view kOp = "StereoRectifyOp";

schema::PixelRect to_message(const PixelRect& r) {
  return schema::PixelRect(r.x, r.y, r.width, r.height);
}

schema::Matrix3 to_message(const Matrix3& m) {
  return schema::Matrix3(::flatbuffers::span<const double, 9>(m));
}

}  // namespace

struct StereoRectifyOp::State {
  State(const StereoCalibration& calibration, const StereoRectification& rectification)
      : left_model(left_remap(calibration, rectification)),
        right_model(right_remap(calibration, rectification)) {}

  RemapModel left_model;
  RemapModel right_model;
  detail::RemapperSlot left;
  detail::RemapperSlot right;
  detail::Staging left_staging;
  detail::Staging right_staging;
  bool copy_logged{};
};

void validate_config(const StereoRectifyConfig& config) {
  check_available(config.backend);
  (void)compute_stereo_rectification(config.calibration, config.options);
}

schema::StereoGeometryT make_geometry_message(const StereoRectification& r,
                                              std::string_view backend) {
  schema::StereoGeometryT message;
  message.backend = std::string(backend);
  message.width = r.width;
  message.height = r.height;
  message.fx = r.fx;
  message.cx_left = r.cx_left;
  message.cx_right = r.cx_right;
  message.cy = r.cy;
  message.baseline = r.baseline;
  message.doffs = r.doffs;
  message.disparity_to_depth = std::make_unique<schema::Matrix4>(
      ::flatbuffers::span<const double, 16>(r.disparity_to_depth));
  message.rect_from_left = std::make_unique<schema::Matrix3>(to_message(r.rect_from_left));
  message.rect_from_right = std::make_unique<schema::Matrix3>(to_message(r.rect_from_right));
  message.valid_left = std::make_unique<schema::PixelRect>(to_message(r.valid_left));
  message.valid_right = std::make_unique<schema::PixelRect>(to_message(r.valid_right));
  return message;
}

StereoRectifyOp::StereoRectifyOp(StereoRectifyConfig config)
    : config_(std::move(config)),
      rectification_(compute_stereo_rectification(config_.calibration, config_.options)),
      state_(std::make_unique<State>(config_.calibration, rectification_)) {
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
  // An unconnected output fails every emit, so it exists only when asked for.
  if (config_.geometry_output) {
    spec.output(geometry, "geometry");
  }
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
  const CameraModel& camera = config_.calibration.left;
  for (const detail::InputImage* input : {&*left_input, &*right_input}) {
    if (input->width != camera.width || input->height != camera.height) {
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
  auto right_remapper = state.right.get(config_.backend, state.right_model, width, height,
                                        right_input->format, config_.interpolation, kOp,
                                        /*log_auto=*/false);
  if (!right_remapper) return holoscan::make_unexpected(std::move(right_remapper).error());
  const std::size_t host_bytes =
      (left_input->on_device ? 0U : left_input->row_bytes() * left_input->height) +
      (right_input->on_device ? 0U : right_input->row_bytes() * right_input->height);
  if (host_bytes != 0 && !state.copy_logged) {
    state.copy_logged = true;
    detail::log_host_copy(kOp, (*left_remapper)->name(), host_bytes);
  }
  auto left_source =
      state.left_staging.on_device(*left_input, *left_sample->data.data, *stream, kOp);
  if (!left_source) return holoscan::make_unexpected(std::move(left_source).error());
  auto right_source =
      state.right_staging.on_device(*right_input, *right_sample->data.data, *stream, kOp);
  if (!right_source) return holoscan::make_unexpected(std::move(right_source).error());

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
  try {
    // Both submitted before either is waited for, so that the eyes overlap.
    (*left_remapper)->submit(*left_source, left_output.image, *stream);
    (*right_remapper)->submit(*right_source, right_output.image, *stream);
    (*left_remapper)->finish();
    (*right_remapper)->finish();
  } catch (const std::exception& error) {
    try {
      (*left_remapper)->finish();
      (*right_remapper)->finish();
    } catch (const std::exception&) {
      // Already failing; the first error is the one to report.
    }
    cudaStreamSynchronize(*stream);
    return holoscan::make_unexpected(dynamic_cast<const std::invalid_argument*>(&error) != nullptr
                                         ? detail::invalid(kOp, error.what())
                                         : detail::failure(kOp, error.what()));
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
  if (config_.geometry_output) {
    return geometry.emit(make_geometry_message(rectification_, (*left_remapper)->name()), emit);
  }
  return {};
}

}  // namespace holoscan::rectify
