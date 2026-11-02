// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The VIC through VPI 4. In a build without VPI, vic_unavailable.cpp stands in.

#pragma once

#include <cuda_runtime_api.h>

#include <memory>

#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/color_convert/image_view.hpp>

namespace holoscan::color_convert::detail {

class VicEngine {
 public:
  virtual ~VicEngine() = default;
  // Converts device images on the VIC, after `stream`'s work, and returns when
  // it is done. The letterbox border is filled by the CUDA kernel on `stream`.
  // The caller has checked vic_limitation().
  virtual void convert(const ImageView& source, const ImageView& destination,
                       const ConversionParams& params, cudaStream_t stream) = 0;

 protected:
  VicEngine() = default;
};

// Whether this build has VPI.
[[nodiscard]] bool vpi_compiled() noexcept;
// Whether VPI reports a VIC; false without VPI.
// @throws std::runtime_error if VPI fails to answer.
[[nodiscard]] bool vpi_has_vic();
// @throws std::invalid_argument without VPI, std::runtime_error if VPI fails.
[[nodiscard]] std::unique_ptr<VicEngine> make_vic_engine();

}  // namespace holoscan::color_convert::detail
