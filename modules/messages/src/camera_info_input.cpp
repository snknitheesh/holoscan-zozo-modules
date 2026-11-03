// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/messages/camera_info_input.hpp"

#include <stdexcept>
#include <string>

namespace holoscan::messages {

namespace detail {

void check_options(const CameraInfoInputOptions& options) {
  if (options.max_age <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("camera_info_contract: max_age must be > 0");
  }
}

std::optional<holoscan::Error> selection_error(holoscan::SelectionReason reason) {
  switch (reason) {
    case holoscan::SelectionReason::kCaptureTimeAbsent:
      return holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                             "camera info without a capture time cannot be matched to frames; "
                             "send it with EmitOptions::capture_time"};
    case holoscan::SelectionReason::kClockIncomparable:
    case holoscan::SelectionReason::kConversionUnavailable:
      return holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                             "the camera info's capture time is in another clock domain than "
                             "the frames', so it cannot be matched to them"};
    default:
      return std::nullopt;
  }
}

holoscan::Error invalid_camera_info(const std::string& what) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "invalid camera info: " + what};
}

}  // namespace detail
}  // namespace holoscan::messages
