// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "report_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

using std::chrono_literals::operator""ns;
namespace rc = holoscan::rectify;

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

bool same(const holoscan::rectify::schema::Matrix3* m, const rc::Matrix3& expected) {
  if (m == nullptr) return false;
  for (int i = 0; i < 9; ++i) {
    if (m->m()->Get(i) != expected[i]) return false;
  }
  return true;
}

}  // namespace

double Report::rms_row_error() const { return dots > 0 ? std::sqrt(sum_row_error2 / dots) : 0.0; }

ReportOp::ReportOp(ReportConfig config) : config_(std::move(config)) {}

void ReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(raw_left, "raw_left").queue_depth(2U);
  spec.input(raw_right, "raw_right").queue_depth(2U);
  spec.input(rectified_left, "rectified_left").queue_depth(2U);
  spec.input(rectified_right, "rectified_right").queue_depth(2U);
  spec.input(geometry, "geometry").queue_depth(2U);
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
  // Every input of one pair carries its capture time.
  holoscan::Contract result;
  const std::vector<holoscan::PortRef> ports{raw_left, raw_right, rectified_left, rectified_right,
                                             geometry};
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
  result.input(raw_left, holoscan::Exact{});
  result.input(raw_right, holoscan::Exact{});
  result.input(rectified_left, holoscan::Exact{});
  result.input(rectified_right, holoscan::Exact{});
  result.input(geometry, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> ReportOp::compute(holoscan::ExecutionContext& context) {
  auto rl = raw_left.receive();
  if (!rl) return holoscan::make_unexpected(std::move(rl).error());
  auto rr = raw_right.receive();
  if (!rr) return holoscan::make_unexpected(std::move(rr).error());
  auto xl = rectified_left.receive();
  if (!xl) return holoscan::make_unexpected(std::move(xl).error());
  auto xr = rectified_right.receive();
  if (!xr) return holoscan::make_unexpected(std::move(xr).error());
  auto g = geometry.receive();
  if (!g) return holoscan::make_unexpected(std::move(g).error());

  const rc::StereoRectification& r = config_.rectification;
  const int in_width = rl->data.width;
  const int in_height = rl->data.height;
  auto raw_l = host_pixels(rl->data, in_width, in_height, context);
  if (!raw_l) return holoscan::make_unexpected(std::move(raw_l).error());
  auto raw_r = host_pixels(rr->data, in_width, in_height, context);
  if (!raw_r) return holoscan::make_unexpected(std::move(raw_r).error());
  auto rect_l = host_pixels(xl->data, r.width, r.height, context);
  if (!rect_l) return holoscan::make_unexpected(std::move(rect_l).error());
  auto rect_r = host_pixels(xr->data, r.width, r.height, context);
  if (!rect_r) return holoscan::make_unexpected(std::move(rect_r).error());

  Report& report = *config_.report;
  const rc::schema::StereoGeometryT& geometry_message = g->data;
  report.geometry_ok = report.geometry_ok && geometry_message.width == r.width &&
                       geometry_message.height == r.height && geometry_message.fx == r.fx &&
                       geometry_message.cx_left == r.cx_left &&
                       geometry_message.cx_right == r.cx_right && geometry_message.cy == r.cy &&
                       geometry_message.baseline == r.baseline &&
                       geometry_message.doffs == r.doffs &&
                       same(geometry_message.rect_from_left.get(), r.rect_from_left) &&
                       same(geometry_message.rect_from_right.get(), r.rect_from_right);

  report.backend = geometry_message.backend;
  const auto frame = static_cast<std::uint32_t>(xl->metadata.frame_id - 1U);
  if (config_.scene) {
    const rc::AlignmentReport check =
        rc::check_rectified(*config_.scene, r, rect_l->data(), rect_r->data(), r.width, frame);
    report.dots += check.dots;
    report.missing += check.missing;
    report.max_row_error = std::max(report.max_row_error, check.max_row_error);
    report.sum_row_error2 += check.rms_row_error * check.rms_row_error * check.dots;
    report.max_position_error = std::max(report.max_position_error, check.max_position_error);
  } else if (r.width == in_width && r.height == in_height) {
    for (std::size_t i = 0; i < raw_l->size(); ++i) {
      report.max_pixel_change =
          std::max({report.max_pixel_change, std::abs((*raw_l)[i] - (*rect_l)[i]),
                    std::abs((*raw_r)[i] - (*rect_r)[i])});
    }
  }
  if (config_.view_output) {
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
