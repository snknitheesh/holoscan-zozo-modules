// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_disparity/geometry_message.hpp"

namespace holoscan::stereo_disparity {

schema::StereoGeometryT to_message(const StereoGeometry& geometry) {
  schema::StereoGeometryT message;
  message.fx = geometry.fx;
  message.baseline = geometry.baseline;
  message.width = geometry.width;
  message.height = geometry.height;
  message.has_principal_point = geometry.cx.has_value() && geometry.cy.has_value();
  message.cx = geometry.cx.value_or(0.0);
  message.cy = geometry.cy.value_or(0.0);
  message.doffs = geometry.doffs;
  return message;
}

StereoGeometry from_message(const schema::StereoGeometryT& message) {
  StereoGeometry geometry{.fx = message.fx,
                          .baseline = message.baseline,
                          .width = message.width,
                          .height = message.height,
                          .doffs = message.doffs};
  if (message.has_principal_point) {
    geometry.cx = message.cx;
    geometry.cy = message.cy;
  }
  validate_geometry(geometry);
  return geometry;
}

}  // namespace holoscan::stereo_disparity
