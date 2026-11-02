// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/foundation_stereo/stereo_geometry.hpp"

#include <cmath>
#include <stdexcept>

namespace holoscan::foundation_stereo {

void validate_geometry(const StereoGeometry& geometry) {
  constexpr int kMaxSide = 16384;
  if (!(std::isfinite(geometry.fx) && geometry.fx > 0)) {
    throw std::invalid_argument("geometry: fx must be finite and positive");
  }
  if (!(std::isfinite(geometry.baseline) && geometry.baseline > 0)) {
    throw std::invalid_argument("geometry: baseline must be finite and positive, in meters");
  }
  if (geometry.width < 1 || geometry.height < 1 || geometry.width > kMaxSide ||
      geometry.height > kMaxSide) {
    throw std::invalid_argument("geometry: width and height must be 1..16384");
  }
  if ((geometry.cx && !std::isfinite(*geometry.cx)) ||
      (geometry.cy && !std::isfinite(*geometry.cy)) || !std::isfinite(geometry.doffs)) {
    throw std::invalid_argument("geometry: cx, cy and doffs must be finite");
  }
}

}  // namespace holoscan::foundation_stereo
