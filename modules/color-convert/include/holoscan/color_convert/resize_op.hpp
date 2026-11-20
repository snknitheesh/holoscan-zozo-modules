// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ResizeOp: an image in, a resized (and optionally cropped, letterboxed and
// converted) image out, in one pass on the GPU or the VIC; with
// `with_camera_info`, the camera of the input images in and the camera of the
// resized ones out.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include <holoscan/color_convert/backend.hpp>
#include <holoscan/color_convert/color_convert_op.hpp>
#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/color_convert/pixel_format.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_schema_traits.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::color_convert {

struct ResizeConfig {
  // The output size, 1..kMaxImageSide.
  int width{640};
  int height{480};
  ResizeMode mode{ResizeMode::kBilinear};
  // The part of the input to resize; the whole input without one. It must
  // lie inside every input image.
  std::optional<Rect> crop;
  // Keep the aspect ratio: fit the input (or crop) inside the output,
  // centered, and fill the rest with `fill_rgb`.
  bool letterbox{false};
  std::array<std::uint8_t, 3> fill_rgb{0, 0, 0};
  // Convert to this format in the same pass; without one, the output keeps
  // each input's format.
  std::optional<PixelFormat> output_format;
  // The format the input images are in, which is then the only one taken.
  // Without `output_format`, it sizes the output's memory pool for that format
  // alone; without either, the pool fits the format of most bytes per pixel,
  // RGB16 (6), up to 6 times what a MONO8 stream needs.
  std::optional<PixelFormat> input_format;
  // As in ColorConvertConfig.
  ColorMatrix matrix{ColorMatrix::kBt601};
  ColorRange range{ColorRange::kLimited};
  bool use_input_color_space{true};
  Backend backend{Backend::kCuda};
  std::optional<PixelFormat> custom_format;
  std::function<void(Backend)> on_backend;
  // Declare the `camera_info` input and the `resized_camera_info` output.
  bool with_camera_info{false};
  // How `camera_info` is taken (messages::camera_info_contract()): by default
  // the newest one, at any rate, for each image.
  messages::CameraInfoInputOptions camera_info_input;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const ResizeConfig& config);

class ResizeOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit ResizeOp(ResizeConfig config);
  ~ResizeOp() override;
  ResizeOp(const ResizeOp&) = delete;
  ResizeOp& operator=(const ResizeOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // As ColorConvertOp::image.
  holoscan::Input<holoscan::schema::ImageT> image;
  // The `width` x `height` image, in `output_format` or the input's, laid out
  // as ColorConvertOp::converted. Its ROI offsets are reset to 0: the crop is
  // recorded in `resized_camera_info`.
  holoscan::Output<holoscan::schema::ImageT> resized;
  // With `with_camera_info`, mandatory: the camera of the input images, of
  // their size, taken as `camera_info_input` says. Images that come before the
  // first camera info are skipped.
  holoscan::Input<messages::schema::CameraInfoT> camera_info;
  // With `with_camera_info`: the camera of each resized image
  // (resized_camera_info() in camera_info.hpp), sent before it with its
  // capture time and frame id, so that a consumer taking the newest camera
  // info (messages::camera_info_contract()) gets this one.
  holoscan::Output<messages::schema::CameraInfoT> resized_camera_info;

 private:
  struct CameraState;

  ResizeConfig config_;
  std::unique_ptr<detail::ConversionRunner> runner_;
  std::unique_ptr<CameraState> camera_;
};

}  // namespace holoscan::color_convert
