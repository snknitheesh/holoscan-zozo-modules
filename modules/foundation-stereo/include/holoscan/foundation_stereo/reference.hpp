// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CPU references for FoundationStereoOp's processing, written from the
// definitions in kernels.hpp and stub_model.hpp in double precision, for
// tests to compare the GPU against: the resize and normalization into the
// model, the stub model, and the resize and rescale of its disparity. With
// the stub model, they give what the operator must output for any pair. No
// SDK or CUDA.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <holoscan/foundation_stereo/model_input.hpp>
#include <holoscan/foundation_stereo/stub_model.hpp>

namespace holoscan::foundation_stereo::reference {

// Bilinear sample of `at(x, y)` over an in_w x in_h grid, for output pixel
// (x, y) of an out_w x out_h grid, with half-pixel centers.
template <typename At>
double bilinear(At at, int in_w, int in_h, int out_w, int out_h, int x, int y) {
  const auto source = [](int index, int in, int out) {
    double s = (index + 0.5) * static_cast<double>(in) / out - 0.5;
    return std::clamp(s, 0.0, static_cast<double>(in - 1));
  };
  const double sx = source(x, in_w, out_w);
  const double sy = source(y, in_h, out_h);
  const int x0 = static_cast<int>(sx);
  const int y0 = static_cast<int>(sy);
  const int x1 = std::min(x0 + 1, in_w - 1);
  const int y1 = std::min(y0 + 1, in_h - 1);
  const double tx = sx - x0;
  const double ty = sy - y0;
  const double top = at(x0, y0) * (1 - tx) + at(x1, y0) * tx;
  const double bottom = at(x0, y1) * (1 - tx) + at(x1, y1) * tx;
  return top * (1 - ty) + bottom * ty;
}

// The model input for an image: planar normalized RGB [3, mh, mw].
inline std::vector<float> preprocess(const std::uint8_t* pixels, std::size_t pitch, int width,
                                     int height, PixelFormat format, int mw, int mh) {
  const int n = channels(format);
  const bool bgr = format == PixelFormat::kBgr8 || format == PixelFormat::kBgra8;
  std::vector<float> out(static_cast<std::size_t>(3) * mw * mh);
  for (int c = 0; c < 3; ++c) {
    const int source = n == 1 ? 0 : bgr ? 2 - c : c;
    const auto at = [&](int x, int y) {
      return static_cast<double>(pixels[static_cast<std::size_t>(y) * pitch + x * n + source]);
    };
    for (int y = 0; y < mh; ++y) {
      for (int x = 0; x < mw; ++x) {
        const double value = bilinear(at, width, height, mw, mh, x, y);
        out[(static_cast<std::size_t>(c) * mh + y) * mw + x] =
            static_cast<float>((value - kImageMean[c]) / kImageStd[c]);
      }
    }
  }
  return out;
}

// What the stub model gives for two preprocessed inputs: [mh, mw].
inline std::vector<float> stub(const std::vector<float>& left, const std::vector<float>& right,
                               int mw, int mh) {
  const std::size_t plane = static_cast<std::size_t>(mw) * mh;
  std::vector<float> out(plane);
  for (std::size_t i = 0; i < plane; ++i) {
    float rgb[3];
    for (int c = 0; c < 3; ++c) {
      rgb[c] = static_cast<float>(static_cast<double>(left[c * plane + i]) * kImageStd[c] +
                                  kImageMean[c]);
    }
    const auto red =
        static_cast<float>(static_cast<double>(right[i]) * kImageStd[0] + kImageMean[0]);
    out[i] = stub_disparity(rgb, red);
  }
  return out;
}

// The model's disparity [mh, mw] at the image's size and in its pixels,
// before any invalidation.
inline std::vector<double> rescale(const std::vector<float>& model, int mw, int mh, int width,
                                   int height) {
  std::vector<double> out(static_cast<std::size_t>(width) * height);
  const auto at = [&](int x, int y) {
    return static_cast<double>(model[static_cast<std::size_t>(y) * mw + x]);
  };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      out[static_cast<std::size_t>(y) * width + x] =
          bilinear(at, mw, mh, width, height, x, y) * width / mw;
    }
  }
  return out;
}

// A rescaled disparity at column x as the operator outputs it: 0 where not
// positive or, with `invalidate_out_of_view`, matched left of the image.
inline double valid(double d, int x, bool invalidate_out_of_view) {
  return !(d > 0) || (invalidate_out_of_view && x - d < 0) ? 0.0 : d;
}

// Whether `d` at column x is within `tolerance` of a validity limit, where
// the GPU's float arithmetic may decide either way.
inline bool near_limit(double d, int x, double tolerance) {
  return std::abs(d) <= tolerance || std::abs(x - d) <= tolerance;
}

}  // namespace holoscan::foundation_stereo::reference
