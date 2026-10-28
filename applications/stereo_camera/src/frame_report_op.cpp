// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "frame_report_op.hpp"

#include <chrono>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/stereo_camera/synthetic.hpp>

namespace stereo_camera_app {
namespace {

namespace sc = holoscan::stereo_camera;
using std::chrono_literals::operator""ns;
using Result = holoscan::expected<void, holoscan::Error>;

std::optional<sc::PixelFormat> format_of(holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_MONO8:
      return sc::PixelFormat::kMono8;
    case holoscan::schema::ImageEncoding_MONO16:
      return sc::PixelFormat::kMono16;
    case holoscan::schema::ImageEncoding_RGB8:
      return sc::PixelFormat::kRgb8;
    case holoscan::schema::ImageEncoding_BGR8:
      return sc::PixelFormat::kBgr8;
    default:
      return std::nullopt;
  }
}

std::string describe(const holoscan::schema::ImageT& image) {
  const auto format = format_of(image.encoding);
  return std::to_string(image.width) + "x" + std::to_string(image.height) + " " +
         std::string(format ? sc::pixel_format_name(*format) : "unknown");
}

// Whether `image` is `width` x `height`, with a packed tensor of that size in a
// format the module sends.
bool well_formed(const holoscan::schema::ImageT& image, int width, int height) {
  const auto format = format_of(image.encoding);
  if (!image.data || !format || image.width != width || image.height != height) {
    return false;
  }
  const std::span<const std::int64_t> shape = image.data->shape_span();
  const bool shape_ok =
      sc::channels(*format) == 1
          ? shape.size() == 2 && shape[0] == height && shape[1] == width
          : shape.size() == 3 && shape[0] == height && shape[1] == width && shape[2] == 3;
  return shape_ok && image.data->nbytes() ==
                         static_cast<std::int64_t>(width) * height * sc::bytes_per_pixel(*format);
}

// Copies `image` to the host into `out`.
Result to_host(const holoscan::schema::ImageT& image, sc::Image& out,
               holoscan::ExecutionContext& context) {
  out.resize(image.width, image.height, *format_of(image.encoding));
  return image.data->copy_to_host(out.pixels.data(), out.pixels.size(), context.cuda_stream());
}

}  // namespace

FrameReportOp::FrameReportOp(FrameReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("FrameReportOp needs a report");
  }
}

void FrameReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(left, "left").queue_depth(2U);
  spec.input(right, "right").queue_depth(2U);
  if (config_.rgb) spec.input(rgb, "rgb").queue_depth(2U);
  if (config_.depth) spec.input(depth, "depth").queue_depth(2U);
  spec.input(calibration, "calibration").queue_depth(2U);
}

holoscan::Contract FrameReportOp::contract() const {
  // Every port of a frame carries the same capture time.
  std::vector<holoscan::PortRef> ports{left, right};
  if (config_.rgb) ports.emplace_back(rgb);
  if (config_.depth) ports.emplace_back(depth);
  ports.emplace_back(calibration);
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
  result.input(left, holoscan::Exact{});
  result.input(right, holoscan::Exact{});
  if (config_.rgb) result.input(rgb, holoscan::Exact{});
  if (config_.depth) result.input(depth, holoscan::Exact{});
  result.input(calibration, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> FrameReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto l = left.receive();
  if (!l) return holoscan::make_unexpected(std::move(l).error());
  auto r = right.receive();
  if (!r) return holoscan::make_unexpected(std::move(r).error());
  auto c = calibration.receive();
  if (!c) return holoscan::make_unexpected(std::move(c).error());
  std::optional<holoscan::Sample<holoscan::schema::ImageT>> rgb_sample;
  std::optional<holoscan::Sample<holoscan::schema::ImageT>> depth_sample;
  if (config_.rgb) {
    auto sample = rgb.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    rgb_sample.emplace(std::move(*sample));
  }
  if (config_.depth) {
    auto sample = depth.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    depth_sample.emplace(std::move(*sample));
  }
  FrameReport& report = *config_.report;
  const sc::schema::StereoCalibrationT& message = c->data;
  sc::StereoCalibration calibration_now;
  try {
    calibration_now = sc::from_message(message);
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        std::string("FrameReportOp: unusable calibration: ") + error.what()});
  }
  const std::string calibration_text = sc::format_calibration(calibration_now);

  const std::uint64_t frame_id = l->metadata.frame_id;
  const std::int64_t capture = l->metadata.capture_timestamp_ns;
  bool sizes_ok = well_formed(l->data, calibration_now.left.width, calibration_now.left.height) &&
                  well_formed(r->data, calibration_now.right.width, calibration_now.right.height) &&
                  l->data.encoding == r->data.encoding;
  if (rgb_sample) {
    sizes_ok &= calibration_now.rgb.has_value() &&
                rgb_sample->data.encoding == holoscan::schema::ImageEncoding_RGB8 &&
                well_formed(rgb_sample->data, calibration_now.rgb->model.width,
                            calibration_now.rgb->model.height);
  }
  if (depth_sample) {
    sizes_ok &= calibration_now.depth.has_value() &&
                depth_sample->data.encoding == holoscan::schema::ImageEncoding_MONO16 &&
                well_formed(depth_sample->data, calibration_now.depth->model.width,
                            calibration_now.depth->model.height);
  }
  // A looping recording starts its frames and timestamps over.
  const bool wrapped = last_index_ && message.frame < *last_index_;
  const bool increasing = (!last_capture_ || capture > *last_capture_) &&
                          (!last_timestamp_ || wrapped || message.timestamp_ns > *last_timestamp_);
  const std::uint64_t last_id = report.last_frame_id.load();
  const bool gap = !config_.realtime && frame_id != last_id + 1U;
  const bool id_increases = frame_id > last_id;
  last_capture_ = capture;
  last_timestamp_ = message.timestamp_ns;
  last_index_ = message.frame;

  // The synthetic camera's images, pixel for pixel.
  bool images_ok = true;
  if (config_.synthetic && sizes_ok) {
    const int width = calibration_now.left.width;
    const int height = calibration_now.left.height;
    sc::render_synthetic_frame(width, height, message.frame, truth_[0], truth_[1],
                               rgb_sample ? &truth_[2] : nullptr,
                               depth_sample ? &truth_[3] : nullptr);
    const holoscan::schema::ImageT* images[4] = {&l->data, &r->data,
                                                 rgb_sample ? &rgb_sample->data : nullptr,
                                                 depth_sample ? &depth_sample->data : nullptr};
    for (int i = 0; i < 4; ++i) {
      if (images[i] == nullptr) {
        continue;
      }
      if (auto copied = to_host(*images[i], host_[i], context); !copied) {
        return copied;
      }
      images_ok &= sc::same_pixels(host_[i].view(), truth_[i].view());
    }
  }
  std::uint64_t valid = 0;
  std::uint64_t pixels = 0;
  if (depth_sample && sizes_ok) {
    if (!config_.synthetic) {
      if (auto copied = to_host(depth_sample->data, host_[3], context); !copied) {
        return copied;
      }
    }
    pixels = host_[3].pixels.size() / 2;
    for (std::uint64_t i = 0; i < pixels; ++i) {
      std::uint16_t value = 0;
      std::memcpy(&value, host_[3].pixels.data() + 2 * i, 2);
      valid += value != 0U;
    }
  }

  {
    const std::lock_guard lock(report.mutex);
    if (!report.calibration) {
      report.calibration = calibration_now;
      report.backend = message.backend;
      report.eyes = describe(l->data);
      if (rgb_sample) report.rgb = describe(rgb_sample->data);
      if (depth_sample) report.depth = describe(depth_sample->data);
      first_calibration_ = calibration_text;
    }
    report.wrong_sizes += !sizes_ok;
    report.not_increasing += !(increasing && id_increases);
    report.frame_gaps += gap;
    report.calibration_changes += calibration_text != first_calibration_;
    report.wrong_images += !images_ok;
    report.verified += config_.synthetic && sizes_ok;
    report.depth_pixels += pixels;
    report.valid_depth += valid;
  }
  report.last_frame_id = frame_id;
  const std::uint32_t frames = ++report.frames;
  if (config_.expected_frames != 0 && (config_.realtime ? frame_id >= config_.expected_frames
                                                        : frames == config_.expected_frames)) {
    context.request_completion();
  }
  return {};
}

}  // namespace stereo_camera_app
