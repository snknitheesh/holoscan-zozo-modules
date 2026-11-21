// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/codec/codec_types.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace holoscan::codec {
namespace {

template <typename Enum, std::size_t N>
std::string_view name_of(const std::array<std::pair<Enum, std::string_view>, N>& names,
                         Enum value) noexcept {
  for (const auto& [kind, name] : names) {
    if (kind == value) {
      return name;
    }
  }
  return "unknown";
}

template <typename Enum, std::size_t N>
std::optional<Enum> parse(const std::array<std::pair<Enum, std::string_view>, N>& names,
                          std::string_view text) noexcept {
  for (const auto& [kind, name] : names) {
    if (name == text) {
      return kind;
    }
  }
  return std::nullopt;
}

constexpr std::array<std::pair<Codec, std::string_view>, 2> kCodecs{{
    {Codec::kH264, "h264"},
    {Codec::kH265, "h265"},
}};
constexpr std::array<std::pair<Backend, std::string_view>, 3> kBackends{{
    {Backend::kNvcodec, "nvcodec"},
    {Backend::kV4l2, "v4l2"},
    {Backend::kAuto, "auto"},
}};
constexpr std::array<std::pair<RateControl, std::string_view>, 3> kRateControls{{
    {RateControl::kCbr, "cbr"},
    {RateControl::kVbr, "vbr"},
    {RateControl::kConstQp, "cqp"},
}};
constexpr std::array<std::pair<Preset, std::string_view>, 3> kPresets{{
    {Preset::kFast, "fast"},
    {Preset::kMedium, "medium"},
    {Preset::kQuality, "quality"},
}};
constexpr std::array<std::pair<PixelFormat, std::string_view>, 5> kPixelFormats{{
    {PixelFormat::kRgb8, "rgb8"},
    {PixelFormat::kBgr8, "bgr8"},
    {PixelFormat::kRgba8, "rgba8"},
    {PixelFormat::kBgra8, "bgra8"},
    {PixelFormat::kNv12, "nv12"},
}};

// Bitrates the module accepts, in bits per second.
constexpr std::uint32_t kMinBitrate = 10'000;
constexpr std::uint32_t kMaxBitrate = 800'000'000;
constexpr std::uint32_t kMaxGop = 100'000;

}  // namespace

std::string_view codec_name(Codec codec) noexcept { return name_of(kCodecs, codec); }
std::string_view backend_name(Backend backend) noexcept { return name_of(kBackends, backend); }
std::string_view rate_control_name(RateControl rc) noexcept { return name_of(kRateControls, rc); }
std::string_view preset_name(Preset preset) noexcept { return name_of(kPresets, preset); }
std::string_view pixel_format_name(PixelFormat format) noexcept {
  return name_of(kPixelFormats, format);
}

std::optional<Codec> parse_codec(std::string_view name) noexcept {
  if (name == "avc") return Codec::kH264;
  if (name == "hevc") return Codec::kH265;
  return parse(kCodecs, name);
}
std::optional<Backend> parse_backend(std::string_view name) noexcept {
  return parse(kBackends, name);
}
std::optional<RateControl> parse_rate_control(std::string_view name) noexcept {
  return parse(kRateControls, name);
}
std::optional<Preset> parse_preset(std::string_view name) noexcept { return parse(kPresets, name); }
std::optional<PixelFormat> parse_pixel_format(std::string_view name) noexcept {
  return parse(kPixelFormats, name);
}

int bytes_per_pixel(PixelFormat format) noexcept {
  switch (format) {
    case PixelFormat::kRgb8:
    case PixelFormat::kBgr8:
      return 3;
    case PixelFormat::kRgba8:
    case PixelFormat::kBgra8:
      return 4;
    case PixelFormat::kNv12:
      return 0;
  }
  return 0;
}

void validate_settings(const EncoderSettings& settings) {
  if (codec_name(settings.codec) == "unknown") {
    throw std::invalid_argument("codec: unknown codec");
  }
  if (backend_name(settings.backend) == "unknown") {
    throw std::invalid_argument("backend: unknown backend");
  }
  if (preset_name(settings.preset) == "unknown") {
    throw std::invalid_argument("preset: unknown preset");
  }
  switch (settings.rate_control) {
    case RateControl::kCbr:
    case RateControl::kVbr:
      if (settings.bitrate < kMinBitrate || settings.bitrate > kMaxBitrate) {
        throw std::invalid_argument("bitrate must be " + std::to_string(kMinBitrate) + " to " +
                                    std::to_string(kMaxBitrate) + " bits per second");
      }
      if (settings.rate_control == RateControl::kVbr && settings.max_bitrate != 0 &&
          settings.max_bitrate < settings.bitrate) {
        throw std::invalid_argument("max_bitrate must be 0 or at least bitrate");
      }
      break;
    case RateControl::kConstQp:
      if (settings.qp < 0 || settings.qp > 51) {
        throw std::invalid_argument("qp must be 0 to 51");
      }
      break;
    default:
      throw std::invalid_argument("rate_control: unknown rate control");
  }
  if (settings.gop < 1 || settings.gop > kMaxGop) {
    throw std::invalid_argument("gop must be 1 to " + std::to_string(kMaxGop) + " frames");
  }
  if (!(std::isfinite(settings.frame_rate) && settings.frame_rate >= 1.0 &&
        settings.frame_rate <= 1000.0)) {
    throw std::invalid_argument("frame_rate must be 1 to 1000 frames per second");
  }
}

}  // namespace holoscan::codec
