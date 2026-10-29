// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The remap backends, the pixel formats and interpolations they take, and how
// Backend::kAuto chooses one. No SDK, no CUDA.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::rectify {

// Where an image is remapped.
enum class Backend : std::uint8_t {
  kCuda,     // This module's CUDA kernel.
  kVpiCuda,  // VPI 4's remap on the GPU.
  kVpiVic,   // VPI 4's remap on the VIC of Jetson Orin and Thor.
  kVpiPva,   // VPI 4's remap on the PVA of Jetson Orin and Thor.
  kAuto,     // vpi-vic on a Jetson when the VIC takes the image, else cuda.
};

// "cuda", "vpi-cuda", "vpi-vic", "vpi-pva" or "auto".
[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;
// One of those names; nullopt for anything else.
[[nodiscard]] std::optional<Backend> parse_backend(std::string_view name) noexcept;

enum class Interpolation : std::uint8_t { kLinear, kNearest };
// "linear" or "nearest".
[[nodiscard]] std::string_view interpolation_name(Interpolation interpolation) noexcept;
[[nodiscard]] std::optional<Interpolation> parse_interpolation(std::string_view name) noexcept;

// The images the operators take: packed rows of 8-bit gray or color, or of
// 16-bit values such as depth.
enum class PixelFormat : std::uint8_t { kMono8, kMono16, kRgb8, kBgr8, kRgba8, kBgra8 };
[[nodiscard]] std::string_view pixel_format_name(PixelFormat format) noexcept;
[[nodiscard]] int channels(PixelFormat format) noexcept;
[[nodiscard]] int bytes_per_pixel(PixelFormat format) noexcept;

// What a backend can do on this machine, as Backend::kAuto and
// backend_limitation() need it. Probed by probe_platform().
struct PlatformFacts {
  bool vpi_compiled{};  // This build has the vpi-* backends.
  bool has_vic{};       // VPI reports a VIC (Jetson).
  bool has_pva{};       // VPI reports a PVA (Jetson).
  bool cuda_device{};   // A CUDA device is usable.
};

// Why `backend` cannot remap `width` x `height` `format` images with
// `interpolation` on a platform with `facts`; nullopt when it can, as far as
// is known before trying. The rules, measured with VPI 4.1.4 on Jetson Orin:
//   - MONO16 holds depth or other values that must not be blended: nearest only.
//   - vpi-*: at least 64 x 16 pixels (VPI's warp grid).
//   - vpi-vic: MONO8 or MONO16, even width and height up to 16384 (an odd size
//     corrupts VPI's memory).
//   - vpi-pva: MONO8, width a multiple of 256, even height of at least 64.
// The VIC of Jetson Thor also lacks nearest interpolation; VPI reports that
// when the backend is built, which create_remapper() turns into an error.
[[nodiscard]] std::optional<std::string> backend_limitation(Backend backend,
                                                            const PlatformFacts& facts,
                                                            PixelFormat format,
                                                            Interpolation interpolation, int width,
                                                            int height);

// What Backend::kAuto tries first: vpi-vic where backend_limitation() allows
// it, else cuda. (If the VIC then refuses, create_remapper() falls back to cuda.)
[[nodiscard]] Backend resolve_auto(const PlatformFacts& facts, PixelFormat format,
                                   Interpolation interpolation, int width, int height);

// The backends compiled into this build, in a stable order; never kAuto.
[[nodiscard]] std::vector<Backend> available_backends();
// @throws std::invalid_argument if `backend` is not compiled into this build.
void check_available(Backend backend);
// This machine's facts. A fact that cannot be established is false.
[[nodiscard]] PlatformFacts probe_platform() noexcept;

}  // namespace holoscan::rectify
