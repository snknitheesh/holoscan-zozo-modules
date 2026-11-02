// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The codec module's settings and their names, shared by the encoder, the decoder, the operators
// and the tests. No SDK, CUDA or GPU: the host tests build this on its own.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace holoscan::codec {

enum class Codec : std::uint8_t {
  kH264,  // H.264 / AVC
  kH265,  // H.265 / HEVC
};

// Which hardware path encodes and decodes.
enum class Backend : std::uint8_t {
  // NVIDIA Video Codec SDK (NVENC / NVDEC through libnvidia-encode and libnvcuvid): discrete
  // GPUs, and Jetson Thor.
  kNvcodec,
  // Jetson Multimedia V4L2 devices (/dev/v4l2-nvenc, /dev/v4l2-nvdec): Jetson Orin.
  kV4l2,
  // The first of the above that works on this machine, chosen once: see probe_backends().
  kAuto,
};

enum class RateControl : std::uint8_t {
  kCbr,      // Constant bitrate: `bitrate` bits per second.
  kVbr,      // Variable bitrate around `bitrate`, at most `max_bitrate`.
  kConstQp,  // Constant quantizer `qp`: fixed quality, the bitrate follows the content.
};

// Speed against compression; each backend maps it to its own presets.
enum class Preset : std::uint8_t {
  kFast,     // NVENC P1, V4L2 UltraFast
  kMedium,   // NVENC P4, V4L2 Medium
  kQuality,  // NVENC P7, V4L2 Slow
};

// The pixel formats the encoder takes and the decoder gives.
enum class PixelFormat : std::uint8_t {
  kRgb8,
  kBgr8,
  kRgba8,
  kBgra8,
  // Semi-planar 4:2:0: a luma plane, then a half-size plane of interleaved U, V.
  kNv12,
};

[[nodiscard]] std::string_view codec_name(Codec codec) noexcept;            // "h264", "h265"
[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;      // "nvcodec", ...
[[nodiscard]] std::string_view rate_control_name(RateControl rc) noexcept;  // "cbr", "vbr", "cqp"
[[nodiscard]] std::string_view preset_name(Preset preset) noexcept;  // "fast", "medium", "quality"
[[nodiscard]] std::string_view pixel_format_name(PixelFormat format) noexcept;  // "rgb8", ...
// Each parses what the matching *_name() returns ("hevc" and "avc" too, for codecs); nullopt for
// anything else.
[[nodiscard]] std::optional<Codec> parse_codec(std::string_view name) noexcept;
[[nodiscard]] std::optional<Backend> parse_backend(std::string_view name) noexcept;
[[nodiscard]] std::optional<RateControl> parse_rate_control(std::string_view name) noexcept;
[[nodiscard]] std::optional<Preset> parse_preset(std::string_view name) noexcept;
[[nodiscard]] std::optional<PixelFormat> parse_pixel_format(std::string_view name) noexcept;
// Bytes per pixel of the packed formats; 0 for kNv12.
[[nodiscard]] int bytes_per_pixel(PixelFormat format) noexcept;

// How a VideoEncoder compresses.
struct EncoderSettings {
  Codec codec{Codec::kH264};
  Backend backend{Backend::kAuto};
  RateControl rate_control{RateControl::kVbr};
  // Target bits per second, for kCbr and kVbr.
  std::uint32_t bitrate{8'000'000};
  // Peak bits per second for kVbr; 0 means twice `bitrate`.
  std::uint32_t max_bitrate{0};
  // Quantizer for kConstQp, 0 (best) to 51: the module's "CRF-like" quality knob. 20 to 25 looks
  // close to the source, 35 and above shows blocks.
  int qp{23};
  // Frames from one IDR keyframe to the next: frame n of a stream is a keyframe exactly when
  // n % gop == 0. Every keyframe starts with its parameter sets (VPS for H.265, SPS, PPS), so a
  // decoder can join the stream there.
  std::uint32_t gop{30};
  Preset preset{Preset::kFast};
  // One access unit out per frame in, as soon as it is encoded, and no B-frames (the encoder never
  // uses B-frames): NVENC's ultra-low-latency tuning. Off, NVENC tunes for quality instead.
  bool low_latency{true};
  // Frames per second the rate control plans for.
  double frame_rate{30.0};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_settings(const EncoderSettings& settings);

// Largest frame side the module accepts; the hardware may allow less (see the backends).
inline constexpr int kMaxFrameSide = 8192;
// Smallest frame side, which every backend takes (NVENC encodes H.264 from 145 pixels wide, NVDEC
// decodes H.265 from 144).
inline constexpr int kMinFrameSide = 160;

}  // namespace holoscan::codec
