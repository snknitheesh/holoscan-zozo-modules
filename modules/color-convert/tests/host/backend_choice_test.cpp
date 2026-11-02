// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Backend::kAuto's choice and what the VIC can do, without the SDK or a GPU.

#include <optional>
#include <string>

#include <holoscan/color_convert/backend.hpp>

#include "test_support.hpp"

namespace holoscan::color_convert {
namespace {

bool vic_can(PixelFormat from, int sw, int sh, PixelFormat to, int w, int h,
             const ConversionParams& params = {}) {
  return !vic_limitation(from, sw, sh, to, w, h, params).has_value();
}

// The VIC wins only when the build has it, VPI reports one and it can do the
// conversion; CUDA otherwise.
void check_resolve_auto() {
  static_assert(resolve_auto({.vpi_compiled = true, .has_vic = true}, true) == Backend::kVic);
  VERIFY(resolve_auto({.vpi_compiled = true, .has_vic = true}, true) == Backend::kVic);
  VERIFY(resolve_auto({.vpi_compiled = true, .has_vic = true}, false) == Backend::kCuda);
  VERIFY(resolve_auto({.vpi_compiled = true, .has_vic = false}, true) == Backend::kCuda);
  VERIFY(resolve_auto({.vpi_compiled = false, .has_vic = true}, true) == Backend::kCuda);
  VERIFY(resolve_auto({.vpi_compiled = false, .has_vic = false}, false) == Backend::kCuda);
}

void check_vic_formats() {
  VERIFY(vic_can(PixelFormat::kNv12, 1920, 1080, PixelFormat::kRgba8, 1920, 1080));
  VERIFY(vic_can(PixelFormat::kYuyv, 1920, 1080, PixelFormat::kBgra8, 640, 360));
  VERIFY(vic_can(PixelFormat::kUyvy, 1280, 720, PixelFormat::kNv12, 1280, 720));
  VERIFY(vic_can(PixelFormat::kRgba8, 1280, 720, PixelFormat::kMono8, 1280, 720));
  // RGB8 and BGR8, I420, NV16 and the 16-bit formats are the CUDA kernel's.
  for (const PixelFormat format : {PixelFormat::kRgb8, PixelFormat::kBgr8, PixelFormat::kI420,
                                   PixelFormat::kNv16, PixelFormat::kMono16, PixelFormat::kRgb16}) {
    VERIFY(!vic_can(PixelFormat::kNv12, 640, 480, format, 640, 480));
    VERIFY(!vic_can(format, 640, 480, PixelFormat::kRgba8, 640, 480));
  }
  const auto why = vic_limitation(PixelFormat::kNv12, 64, 64, PixelFormat::kRgb8, 64, 64, {});
  VERIFY(why.has_value() && why->find("rgb8") != std::string::npos);
}

void check_vic_geometry() {
  // No area sampling.
  VERIFY(!vic_can(PixelFormat::kNv12, 1920, 1080, PixelFormat::kRgba8, 640, 360,
                  ConversionParams{.mode = ResizeMode::kArea}));
  VERIFY(vic_can(PixelFormat::kNv12, 1920, 1080, PixelFormat::kRgba8, 640, 360,
                 ConversionParams{.mode = ResizeMode::kNearest}));
  // Sides of 16 to 16384.
  VERIFY(!vic_can(PixelFormat::kRgba8, 15, 64, PixelFormat::kRgba8, 64, 64));
  VERIFY(!vic_can(PixelFormat::kRgba8, 64, 64, PixelFormat::kRgba8, 64, 8));
  // At most 16 times up or down.
  VERIFY(vic_can(PixelFormat::kRgba8, 3200, 1600, PixelFormat::kRgba8, 200, 100));
  VERIFY(!vic_can(PixelFormat::kRgba8, 3200, 1600, PixelFormat::kRgba8, 199, 100));
  VERIFY(!vic_can(PixelFormat::kRgba8, 16, 16, PixelFormat::kRgba8, 257, 16));
  // Chroma-subsampled images only at even offsets and sizes.
  VERIFY(!vic_can(PixelFormat::kNv12, 1920, 1080, PixelFormat::kRgba8, 640, 360,
                  ConversionParams{.crop = Rect{1, 0, 640, 360}}));
  VERIFY(!vic_can(PixelFormat::kNv12, 1920, 1080, PixelFormat::kRgba8, 640, 360,
                  ConversionParams{.crop = Rect{0, 0, 641, 360}}));
  VERIFY(vic_can(PixelFormat::kNv12, 1920, 1080, PixelFormat::kRgba8, 640, 360,
                 ConversionParams{.crop = Rect{2, 4, 640, 360}}));
  // RGBA has no chroma to align.
  VERIFY(vic_can(PixelFormat::kRgba8, 1920, 1080, PixelFormat::kRgba8, 641, 361,
                 ConversionParams{.crop = Rect{1, 1, 641, 361}}));
  VERIFY(!vic_can(PixelFormat::kRgba8, 1920, 1080, PixelFormat::kNv12, 641, 360));
  // Letterboxed into a subsampled destination, the content is made even.
  VERIFY(vic_can(PixelFormat::kNv12, 1920, 1080, PixelFormat::kNv12, 640, 640,
                 ConversionParams{.letterbox = true}));
  // BT.2020 only where no matrix applies.
  VERIFY(!vic_can(PixelFormat::kNv12, 1920, 1080, PixelFormat::kRgba8, 1920, 1080,
                  ConversionParams{.matrix = ColorMatrix::kBt2020}));
  VERIFY(vic_can(PixelFormat::kNv12, 1920, 1080, PixelFormat::kYuyv, 1920, 1080,
                 ConversionParams{.matrix = ColorMatrix::kBt2020}));
  VERIFY(vic_can(PixelFormat::kRgba8, 1920, 1080, PixelFormat::kBgra8, 1280, 720,
                 ConversionParams{.matrix = ColorMatrix::kBt2020}));
  // A crop outside the source is no VIC conversion either.
  VERIFY(!vic_can(PixelFormat::kRgba8, 64, 64, PixelFormat::kRgba8, 64, 64,
                  ConversionParams{.crop = Rect{40, 0, 40, 40}}));
}

void check_names() {
  for (const Backend backend : {Backend::kCuda, Backend::kVic, Backend::kAuto}) {
    VERIFY(parse_backend(backend_name(backend)) == backend);
  }
  VERIFY(!parse_backend("pva").has_value());
}

}  // namespace
}  // namespace holoscan::color_convert

int main() {
  namespace cc = holoscan::color_convert;
  return cc::test::run({
      {"auto's choice", cc::check_resolve_auto},
      {"formats the VIC handles", cc::check_vic_formats},
      {"geometry the VIC handles", cc::check_vic_geometry},
      {"backend names", cc::check_names},
  });
}
