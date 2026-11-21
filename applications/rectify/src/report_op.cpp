// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "report_op.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace rectify_app {
namespace {

namespace rc = holoscan::rectify;
namespace hm = holoscan::messages;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
// Rows between two epipolar lines in the view.
constexpr int kLineSpacing = 40;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "ReportOp: " + message};
}

// The pixels of a MONO8 image, in host memory.
holoscan::expected<std::vector<std::uint8_t>, holoscan::Error> host_pixels(
    const holoscan::schema::ImageT& image, int width, int height,
    holoscan::ExecutionContext& context) {
  if (!image.data || image.encoding != holoscan::schema::ImageEncoding_MONO8 ||
      image.width != width || image.height != height) {
    return holoscan::make_unexpected(invalid("expected a " + std::to_string(width) + "x" +
                                             std::to_string(height) + " MONO8 image"));
  }
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height);
  if (auto copied = image.data->copy_to_host(pixels.data(), pixels.size(), context.cuda_stream());
      !copied) {
    return holoscan::make_unexpected(std::move(copied).error());
  }
  return pixels;
}

}  // namespace

double Report::rms_row_error() const { return dots > 0 ? std::sqrt(sum_row_error2 / dots) : 0.0; }

ReportOp::ReportOp(ReportConfig config) : config_(std::move(config)) {}

void ReportOp::setup(holoscan::OperatorSpec& spec) {
  if (config_.view_output) {
    spec.input(raw_left, "raw_left").queue_depth(2U);
    spec.input(raw_right, "raw_right").queue_depth(2U);
  }
  spec.input(rectified_left, "rectified_left").queue_depth(2U);
  spec.input(rectified_right, "rectified_right").queue_depth(2U);
  spec.input(camera_info, "camera_info").queue_depth(2U);
  if (config_.view_output) {
    const auto& r = config_.rectification;
    spec.output(view, "view")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
                    .dtype = kUInt8,
                    .rank = 3U,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                },
            .bounds = holoscan::tensor_bounds(std::uint64_t{4U} * 4U * r.width * r.height),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        })
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
        });
  }
}

holoscan::Contract ReportOp::contract() const {
  // StereoRectifyOp sends a camera info with every pair: each input of one
  // pair carries its capture time.
  const hm::CameraInfoInputOptions every_pair{.timing = hm::CameraInfoTiming::kEveryFrame};
  if (config_.view_output) {
    return hm::camera_info_contract(every_pair, camera_info, rectified_left, rectified_right,
                                    raw_left, raw_right);
  }
  return hm::camera_info_contract(every_pair, camera_info, rectified_left, rectified_right);
}

holoscan::expected<void, holoscan::Error> ReportOp::compute(holoscan::ExecutionContext& context) {
  auto xl = rectified_left.receive();
  if (!xl) return holoscan::make_unexpected(std::move(xl).error());
  auto xr = rectified_right.receive();
  if (!xr) return holoscan::make_unexpected(std::move(xr).error());
  auto info = info_.update(camera_info);
  if (!info) return holoscan::make_unexpected(std::move(info).error());

  const rc::StereoRectification& r = config_.rectification;
  auto rect_l = host_pixels(xl->data, r.width, r.height, context);
  if (!rect_l) return holoscan::make_unexpected(std::move(rect_l).error());
  auto rect_r = host_pixels(xr->data, r.width, r.height, context);
  if (!rect_r) return holoscan::make_unexpected(std::move(rect_r).error());

  Report& report = *config_.report;
  // With kEveryFrame each pair arrives with its own camera info, never without.
  report.camera_info_ok = report.camera_info_ok && *info != nullptr &&
                          hm::same_calibration(**info, r.camera_info) &&
                          (*info)->header.capture_timestamp_ns == xl->metadata.capture_timestamp_ns;
  const auto frame = static_cast<std::uint32_t>(xl->metadata.frame_id - 1U);
  if (config_.scene) {
    const rc::AlignmentReport check =
        rc::check_rectified(*config_.scene, r, rect_l->data(), rect_r->data(), r.width, frame);
    report.dots += check.dots;
    report.missing += check.missing;
    report.max_row_error = std::max(report.max_row_error, check.max_row_error);
    report.sum_row_error2 += check.rms_row_error * check.rms_row_error * check.dots;
    report.max_position_error = std::max(report.max_position_error, check.max_position_error);
  } else if (!config_.recording.empty()) {
    // The recording's pair again, from its files.
    const RecordedFrame& recorded = config_.recording[frame % config_.recording.size()];
    try {
      const GrayImage raw_l = read_pgm(recorded.left);
      const GrayImage raw_r = read_pgm(recorded.right);
      if (raw_l.pixels.size() != rect_l->size() || raw_r.pixels.size() != rect_r->size()) {
        return holoscan::make_unexpected(invalid("the rectified size differs from the recording"));
      }
      for (std::size_t i = 0; i < rect_l->size(); ++i) {
        report.max_pixel_change =
            std::max({report.max_pixel_change, std::abs(raw_l.pixels[i] - (*rect_l)[i]),
                      std::abs(raw_r.pixels[i] - (*rect_r)[i])});
      }
    } catch (const std::invalid_argument& error) {
      return holoscan::make_unexpected(invalid(error.what()));
    }
  }
  if (config_.view_output) {
    auto rl = raw_left.receive();
    if (!rl) return holoscan::make_unexpected(std::move(rl).error());
    auto rr = raw_right.receive();
    if (!rr) return holoscan::make_unexpected(std::move(rr).error());
    auto raw_l = host_pixels(rl->data, rl->data.width, rl->data.height, context);
    if (!raw_l) return holoscan::make_unexpected(std::move(raw_l).error());
    auto raw_r = host_pixels(rr->data, rr->data.width, rr->data.height, context);
    if (!raw_r) return holoscan::make_unexpected(std::move(raw_r).error());
    // Best effort: the display keeps only the newest view and never holds this back.
    (void)send_view(*raw_l, *raw_r, *rect_l, *rect_r, holoscan::sensor_io::derive_from(*xl));
  }
  // Counted last, once the pair is done with: the source sends the next one then.
  if (report.frames.fetch_add(1) + 1 == config_.expected_frames &&
      config_.remaining->fetch_sub(1) == 1) {
    context.request_completion();
  }
  return {};
}

holoscan::expected<void, holoscan::Error> ReportOp::send_view(
    const std::vector<std::uint8_t>& raw_l, const std::vector<std::uint8_t>& raw_r,
    const std::vector<std::uint8_t>& rect_l, const std::vector<std::uint8_t>& rect_r,
    const holoscan::EmitOptions& emit) {
  const rc::StereoRectification& r = config_.rectification;
  const int w = r.width;
  const int h = r.height;
  const int view_w = 2 * w;
  const int view_h = 2 * h;
  const std::array<std::int64_t, 3> shape{view_h, view_w, 4};
  auto loan = view.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
    auto* out = writer.data_as<std::uint8_t>();
    const auto put = [&](int x, int y, std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
      std::uint8_t* p = out + (static_cast<std::size_t>(y) * view_w + x) * 4;
      p[0] = red;
      p[1] = green;
      p[2] = blue;
      p[3] = 255;
    };
    // The four images, gray; the raw ones are the input's size, which may be smaller.
    const std::array<const std::vector<std::uint8_t>*, 4> panels{&raw_l, &raw_r, &rect_l, &rect_r};
    for (int panel = 0; panel < 4; ++panel) {
      const std::vector<std::uint8_t>& pixels = *panels[panel];
      const int pw = panel < 2 ? static_cast<int>(pixels.size()) / std::max(1, h) : w;
      const int ox = (panel % 2) * w;
      const int oy = (panel / 2) * h;
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const std::uint8_t v =
              x < pw ? pixels[static_cast<std::size_t>(y) * pw + x] : std::uint8_t{0};
          put(ox + x, oy + y, v, v, v);
        }
      }
    }
    // Epipolar lines: a scene point lies on one row in both rectified images.
    for (int y = kLineSpacing / 2; y < view_h; y += kLineSpacing) {
      for (int x = 0; x < view_w; ++x) put(x, y, 40, 220, 40);
    }
    // The valid regions of the rectified images.
    for (int panel = 0; panel < 2; ++panel) {
      const rc::PixelRect& v = panel == 0 ? r.valid_left : r.valid_right;
      if (v.width < 1 || v.height < 1) continue;
      const int ox = panel * w;
      for (int x = v.x; x < v.x + v.width; ++x) {
        put(ox + x, h + v.y, 240, 200, 0);
        put(ox + x, h + v.y + v.height - 1, 240, 200, 0);
      }
      for (int y = v.y; y < v.y + v.height; ++y) {
        put(ox + v.x, h + y, 240, 200, 0);
        put(ox + v.x + v.width - 1, h + y, 240, 200, 0);
      }
    }
    return {};
  });
  if (!committed) return holoscan::make_unexpected(std::move(committed).error());
  holoscan::schema::ImageT descriptor;
  descriptor.width = view_w;
  descriptor.height = view_h;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGBA8;
  return view.emit_tensor(std::move(*loan), descriptor, emit);
}

}  // namespace rectify_app
