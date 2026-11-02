// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Converter: one conversion engine, usable without Holoscan, that the
// operators run. It converts and resizes device images on the CUDA kernel or
// on the VIC, as conversion.hpp defines.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/color_convert/backend.hpp>
#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/color_convert/image_view.hpp>

namespace holoscan::color_convert {

namespace detail {
class VicEngine;
}  // namespace detail

// This process's facts for Backend::kAuto. A fact VPI fails to answer is false.
[[nodiscard]] PlatformFacts probe_platform() noexcept;

// Checks that `backend` is compiled into this build. kCuda and kAuto always are.
// @throws std::invalid_argument naming what is missing.
void check_available(Backend backend);

// Fills `rgb8`, a device RGB8 view, with test_pattern_rgb() (reference.hpp)
// for `frame`, on `stream`.
[[nodiscard]] cudaError_t launch_test_pattern(const ImageView& rgb8, std::uint32_t frame,
                                              cudaStream_t stream);

class Converter {
 public:
  // @throws std::invalid_argument if `backend` is not compiled in (check_available()).
  explicit Converter(Backend backend);
  ~Converter();
  Converter(const Converter&) = delete;
  Converter& operator=(const Converter&) = delete;

  // The backend that would convert these images: the configured one, or for
  // kAuto its choice, with why it was made (empty for a configured backend).
  [[nodiscard]] std::pair<Backend, std::string> choose(const ImageView& source,
                                                       const ImageView& destination,
                                                       const ConversionParams& params) const;

  // Converts `source` into `destination`, both in CUDA device memory, after
  // the work already queued on `stream`. The cuda backend queues its kernel on
  // `stream` and returns; the vic backend waits for `stream`, runs on the VIC
  // and returns when it is done. Returns the backend that ran.
  // @throws std::invalid_argument if the images or params are not usable, or
  //         the configured vic backend cannot do the conversion.
  // @throws std::runtime_error on a CUDA or VPI failure.
  Backend convert(const ImageView& source, const ImageView& destination,
                  const ConversionParams& params, cudaStream_t stream);

  [[nodiscard]] Backend backend() const noexcept { return backend_; }

 private:
  Backend backend_;
  PlatformFacts facts_{};
  // Made on the first conversion on the VIC.
  std::unique_ptr<detail::VicEngine> vic_;
};

}  // namespace holoscan::color_convert
