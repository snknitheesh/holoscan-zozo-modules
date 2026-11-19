// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// How the vpi-* backends report a VPI status that is not VPI_SUCCESS: as a
// refusal of the images, the interpolation or the backend on this machine
// (std::invalid_argument, which auto falls back from and try_backend()
// reports), or as a failure (std::runtime_error), which neither hides.

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

// VPI's headers declare C linkage themselves.
#include <vpi/Status.h>  // NOLINT(build/include_order)

namespace holoscan::rectify::detail {

// Whether `status` says VPI does not take this work here: an argument, an
// image format, an operation or a backend it does not support. Any other
// status, such as running out of memory or a device fault, is a failure.
[[nodiscard]] constexpr bool is_refusal(VPIStatus status) noexcept {
  switch (status) {
    case VPI_ERROR_NOT_IMPLEMENTED:
    case VPI_ERROR_INVALID_ARGUMENT:
    case VPI_ERROR_INVALID_IMAGE_FORMAT:
    case VPI_ERROR_INVALID_OPERATION:
    case VPI_ERROR_NOT_SUPPORTED:
      return true;
    default:
      return false;
  }
}

// The status's name and VPI's last message for this thread.
inline std::string status_text(VPIStatus status) {
  char detail[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
  vpiGetLastStatusMessage(detail, sizeof(detail));
  return std::string(vpiStatusGetName(status)) + ": " + detail;
}

// Throws `what` and the status: std::invalid_argument for a refusal, else
// std::runtime_error.
[[noreturn]] inline void refuse_or_fail(VPIStatus status, std::string_view what) {
  const std::string message = std::string(what) + ": " + status_text(status);
  if (is_refusal(status)) throw std::invalid_argument(message);
  throw std::runtime_error(message);
}

}  // namespace holoscan::rectify::detail
