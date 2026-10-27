// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Declares an ImageT output whose tensors the runtime pool allocates.

#pragma once

#include <holoscan/core/memory_kind.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::apriltag {

// Declares `image` as output `name`: width x height x channels uint8 frames in `memory`.
void image_output(holoscan::OperatorSpec& spec, holoscan::Output<holoscan::schema::ImageT>& image,
                  const char* name, holoscan::MemoryKind memory, int width, int height,
                  int channels);

}  // namespace holoscan::apriltag
