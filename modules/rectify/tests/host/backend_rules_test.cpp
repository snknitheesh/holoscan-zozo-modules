// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The backends' names, what each takes, and what Backend::kAuto chooses, for
// the platforms CI runs on.

#include <holoscan/rectify/backend.hpp>

#include "test_support.hpp"

namespace holoscan::rectify {
namespace {

// x86_64 with VPI, a Jetson, and a build without VPI.
constexpr PlatformFacts kX86{.vpi_compiled = true, .cuda_device = true};
constexpr PlatformFacts kJetson{true, true, true, true};
constexpr PlatformFacts kNoVpi{.cuda_device = true};

void test_names() {
  for (const Backend b :
       {Backend::kCuda, Backend::kVpiCuda, Backend::kVpiVic, Backend::kVpiPva, Backend::kAuto}) {
    VERIFY(parse_backend(backend_name(b)) == b);
  }
  VERIFY(!parse_backend("vic"));
  VERIFY(parse_interpolation("nearest") == Interpolation::kNearest);
  VERIFY(!parse_interpolation("cubic"));
  VERIFY(bytes_per_pixel(PixelFormat::kMono16) == 2 && channels(PixelFormat::kBgra8) == 4);
}

void test_limitations() {
  const auto takes = [](Backend b, const PlatformFacts& f, PixelFormat p, Interpolation i, int w,
                        int h) { return !backend_limitation(b, f, p, i, w, h); };
  constexpr auto kLinear = Interpolation::kLinear;
  constexpr auto kNearest = Interpolation::kNearest;
  // Depth is never blended.
  VERIFY(!takes(Backend::kCuda, kX86, PixelFormat::kMono16, kLinear, 640, 480));
  VERIFY(takes(Backend::kCuda, kX86, PixelFormat::kMono16, kNearest, 640, 480));
  VERIFY(takes(Backend::kCuda, kX86, PixelFormat::kRgba8, kLinear, 3, 3));
  VERIFY(takes(Backend::kVpiCuda, kX86, PixelFormat::kRgb8, kLinear, 641, 479));
  VERIFY(!takes(Backend::kVpiCuda, kX86, PixelFormat::kMono8, kLinear, 63, 480));
  VERIFY(!takes(Backend::kVpiCuda, kNoVpi, PixelFormat::kMono8, kLinear, 640, 480));
  VERIFY(!takes(Backend::kVpiVic, kX86, PixelFormat::kMono8, kLinear, 640, 480));
  VERIFY(takes(Backend::kVpiVic, kJetson, PixelFormat::kMono8, kLinear, 848, 480));
  VERIFY(takes(Backend::kVpiVic, kJetson, PixelFormat::kMono16, kNearest, 848, 480));
  VERIFY(!takes(Backend::kVpiVic, kJetson, PixelFormat::kRgb8, kLinear, 848, 480));
  VERIFY(!takes(Backend::kVpiVic, kJetson, PixelFormat::kMono8, kLinear, 641, 480));
  VERIFY(!takes(Backend::kVpiVic, kJetson, PixelFormat::kMono8, kLinear, 640, 359));
  VERIFY(takes(Backend::kVpiPva, kJetson, PixelFormat::kMono8, kLinear, 1280, 720));
  VERIFY(!takes(Backend::kVpiPva, kJetson, PixelFormat::kMono8, kLinear, 848, 480));
  VERIFY(!takes(Backend::kVpiPva, kJetson, PixelFormat::kMono16, kNearest, 1280, 720));
  VERIFY(!takes(Backend::kVpiPva, kJetson, PixelFormat::kMono8, kLinear, 1280, 721));
  VERIFY(!takes(Backend::kVpiPva, kX86, PixelFormat::kMono8, kLinear, 1280, 720));
}

void test_auto() {
  constexpr auto kLinear = Interpolation::kLinear;
  VERIFY(resolve_auto(kX86, PixelFormat::kMono8, kLinear, 1280, 720) == Backend::kCuda);
  VERIFY(resolve_auto(kJetson, PixelFormat::kMono8, kLinear, 1280, 720) == Backend::kVpiVic);
  VERIFY(resolve_auto(kJetson, PixelFormat::kMono16, Interpolation::kNearest, 848, 480) ==
         Backend::kVpiVic);
  VERIFY(resolve_auto(kJetson, PixelFormat::kRgb8, kLinear, 1280, 720) == Backend::kCuda);
  VERIFY(resolve_auto(kJetson, PixelFormat::kMono8, kLinear, 641, 480) == Backend::kCuda);
  VERIFY(resolve_auto(kNoVpi, PixelFormat::kMono8, kLinear, 1280, 720) == Backend::kCuda);
}

}  // namespace
}  // namespace holoscan::rectify

int main() {
  namespace rc = holoscan::rectify;
  return rc::test::run({
      {"names", rc::test_names},
      {"what each backend takes", rc::test_limitations},
      {"auto", rc::test_auto},
  });
}
