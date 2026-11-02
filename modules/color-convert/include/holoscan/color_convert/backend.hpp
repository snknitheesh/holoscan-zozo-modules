// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The backends a conversion runs on, what the VIC can do, and how `auto`
// chooses. Needs no SDK, CUDA or GPU.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/color_convert/pixel_format.hpp>

namespace holoscan::color_convert {

enum class Backend : std::uint8_t {
  kCuda,  // The module's CUDA kernel: every format, mode and size.
  kVic,   // The VIC through VPI 4, on Jetson: what vic_limitation() allows.
  kAuto,  // The VIC when it can do the conversion, else CUDA; chosen on the first frame.
};

// "cuda", "vic", "auto".
[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;
[[nodiscard]] std::optional<Backend> parse_backend(std::string_view name) noexcept;

// What Backend::kAuto's choice depends on.
struct PlatformFacts {
  bool vpi_compiled{};  // This build has the vic backend.
  bool has_vic{};       // VPI reports a VIC (Jetson).
};

// Why the VIC cannot convert a `source_width` x `source_height` `source` image
// into a `width` x `height` `destination` one with `params`, or nothing when
// it can. The VIC converts and rescales between NV12, YUYV, UYVY, RGBA8,
// BGRA8 and MONO8 with nearest or bilinear sampling, sides of 16 to 16384
// pixels and up to 16 times up or down; it reads and writes chroma-subsampled
// images only at even offsets and sizes; between colour models it takes
// only BT.709 and never makes MONO8 from colour, the conversions whose
// results agree on Orin and Thor.
[[nodiscard]] std::optional<std::string> vic_limitation(PixelFormat source, int source_width,
                                                        int source_height, PixelFormat destination,
                                                        int width, int height,
                                                        const ConversionParams& params);

// Backend::kAuto's choice: the VIC when this build has it, the platform has
// one and it can do the conversion; CUDA otherwise.
[[nodiscard]] constexpr Backend resolve_auto(const PlatformFacts& facts,
                                             bool vic_can_convert) noexcept {
  return facts.vpi_compiled && facts.has_vic && vic_can_convert ? Backend::kVic : Backend::kCuda;
}

}  // namespace holoscan::color_convert
