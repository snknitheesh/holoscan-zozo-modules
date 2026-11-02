// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The SDK-free parts: the Annex-B reader on hand-made H.264 and H.265 streams, the settings'
// names and validation, and the test video and PSNR.

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <holoscan/codec/bitstream.hpp>
#include <holoscan/codec/codec_types.hpp>
#include <holoscan/codec/test_video.hpp>

#include "test_support.hpp"

namespace holoscan::codec {
namespace {

using Bytes = std::vector<std::uint8_t>;

// A NAL unit with a start code of `start_code` bytes, its header byte(s) and a payload.
void append_nal(Bytes& stream, Codec codec, int type, std::size_t start_code = 4,
                std::size_t payload = 6) {
  if (start_code == 4) stream.push_back(0);
  stream.insert(stream.end(), {0, 0, 1});
  if (codec == Codec::kH264) {
    stream.push_back(static_cast<std::uint8_t>(0x60 | type));  // nal_ref_idc 3
  } else {
    stream.push_back(static_cast<std::uint8_t>(type << 1));
    stream.push_back(1);  // nuh_temporal_id_plus1
  }
  for (std::size_t i = 0; i < payload; ++i) stream.push_back(static_cast<std::uint8_t>(0x80 + i));
}

void check_h264() {
  Bytes idr;
  append_nal(idr, Codec::kH264, 7);     // SPS
  append_nal(idr, Codec::kH264, 8, 3);  // PPS, three-byte start code
  append_nal(idr, Codec::kH264, 5);     // IDR slice
  const auto units = split_nal_units(Codec::kH264, idr);
  VERIFY(units.size() == 3);
  VERIFY(units[0].type == 7 && units[1].type == 8 && units[2].type == 5);
  VERIFY(units[0].offset == 4 && units[0].size == 7);
  VERIFY(units[1].offset == 4 + 7 + 3);
  const AccessUnitInfo info = analyze_access_unit(Codec::kH264, idr);
  VERIFY(info.sps && info.pps && info.idr && info.random_access && info.slices == 1);
  VERIFY(is_keyframe(Codec::kH264, idr));
  const Bytes sets = parameter_sets(Codec::kH264, idr);
  VERIFY(split_nal_units(Codec::kH264, sets).size() == 2);
  VERIFY(sets.size() == 2 * (4 + 7));

  Bytes p_frame;
  append_nal(p_frame, Codec::kH264, 1);
  VERIFY(!is_keyframe(Codec::kH264, p_frame));
  VERIFY(analyze_access_unit(Codec::kH264, p_frame).slices == 1);
  VERIFY(parameter_sets(Codec::kH264, p_frame).empty());
  // An IDR without its parameter sets cannot start decoding.
  Bytes bare_idr;
  append_nal(bare_idr, Codec::kH264, 5);
  VERIFY(analyze_access_unit(Codec::kH264, bare_idr).idr);
  VERIFY(!is_keyframe(Codec::kH264, bare_idr));
}

void check_h265() {
  Bytes idr;
  append_nal(idr, Codec::kH265, 32);  // VPS
  append_nal(idr, Codec::kH265, 33);  // SPS
  append_nal(idr, Codec::kH265, 34);  // PPS
  append_nal(idr, Codec::kH265, 19);  // IDR_W_RADL
  const AccessUnitInfo info = analyze_access_unit(Codec::kH265, idr);
  VERIFY(info.vps && info.sps && info.pps && info.idr && info.random_access);
  VERIFY(info.nal_units == 4 && info.slices == 1);
  VERIFY(is_keyframe(Codec::kH265, idr));
  VERIFY(split_nal_units(Codec::kH265, parameter_sets(Codec::kH265, idr)).size() == 3);
  // Without the VPS, not a keyframe for H.265.
  Bytes no_vps;
  append_nal(no_vps, Codec::kH265, 33);
  append_nal(no_vps, Codec::kH265, 34);
  append_nal(no_vps, Codec::kH265, 20);
  VERIFY(!is_keyframe(Codec::kH265, no_vps));
  // A CRA is a random access point, not an IDR.
  Bytes cra;
  append_nal(cra, Codec::kH265, 21);
  VERIFY(analyze_access_unit(Codec::kH265, cra).random_access);
  VERIFY(!analyze_access_unit(Codec::kH265, cra).idr);
  Bytes trail;
  append_nal(trail, Codec::kH265, 1);
  VERIFY(!analyze_access_unit(Codec::kH265, trail).random_access);
}

void check_degenerate_streams() {
  VERIFY(split_nal_units(Codec::kH264, {}).empty());
  const Bytes no_start = {1, 2, 3, 4, 5};
  VERIFY(split_nal_units(Codec::kH264, no_start).empty());
  VERIFY(!is_keyframe(Codec::kH264, no_start));
  const Bytes only_start = {0, 0, 0, 1};
  VERIFY(split_nal_units(Codec::kH264, only_start).empty());
  // Leading garbage is skipped; trailing zeros before the next start code are not payload.
  Bytes stream = {7, 7};
  append_nal(stream, Codec::kH264, 7, 3, 2);
  stream.insert(stream.end(), {0, 0});
  append_nal(stream, Codec::kH264, 8, 3, 2);
  const auto units = split_nal_units(Codec::kH264, stream);
  VERIFY(units.size() == 2);
  VERIFY(units[0].offset == 5 && units[0].size == 3);
}

void check_settings() {
  for (const Codec codec : {Codec::kH264, Codec::kH265}) {
    VERIFY(parse_codec(codec_name(codec)) == codec);
  }
  VERIFY(parse_codec("hevc") == Codec::kH265 && parse_codec("avc") == Codec::kH264);
  VERIFY(!parse_codec("vp9"));
  for (const Backend backend : {Backend::kNvcodec, Backend::kV4l2, Backend::kAuto}) {
    VERIFY(parse_backend(backend_name(backend)) == backend);
  }
  for (const RateControl rc : {RateControl::kCbr, RateControl::kVbr, RateControl::kConstQp}) {
    VERIFY(parse_rate_control(rate_control_name(rc)) == rc);
  }
  for (const Preset preset : {Preset::kFast, Preset::kMedium, Preset::kQuality}) {
    VERIFY(parse_preset(preset_name(preset)) == preset);
  }
  for (const PixelFormat format : {PixelFormat::kRgb8, PixelFormat::kBgr8, PixelFormat::kRgba8,
                                   PixelFormat::kBgra8, PixelFormat::kNv12}) {
    VERIFY(parse_pixel_format(pixel_format_name(format)) == format);
  }
  VERIFY(bytes_per_pixel(PixelFormat::kRgba8) == 4 && bytes_per_pixel(PixelFormat::kNv12) == 0);

  validate_settings(EncoderSettings{});
  const auto rejects = [](EncoderSettings settings) {
    return test::throws<std::invalid_argument>([&] { validate_settings(settings); });
  };
  EncoderSettings s;
  s.gop = 0;
  VERIFY(rejects(s));
  s = {};
  s.bitrate = 1000;
  VERIFY(rejects(s));
  s = {};
  s.rate_control = RateControl::kVbr;
  s.max_bitrate = s.bitrate - 1;
  VERIFY(rejects(s));
  s = {};
  s.rate_control = RateControl::kConstQp;
  s.qp = -1;
  VERIFY(rejects(s));
  s.qp = 51;
  validate_settings(s);
  s.bitrate = 0;  // Unused with a constant QP.
  validate_settings(s);
  s = {};
  s.frame_rate = 0.5;
  VERIFY(rejects(s));
  s.frame_rate = std::nan("");
  VERIFY(rejects(s));
}

void check_test_video() {
  const auto a = render_test_frame(320, 240, 7);
  const auto b = render_test_frame(320, 240, 7);
  const auto next = render_test_frame(320, 240, 8);
  VERIFY(a.size() == 320U * 240U * 3U);
  VERIFY(a == b);
  VERIFY(a != next);
  VERIFY(std::isinf(psnr(a, b, 320, 240, 3)));
  const double moved = psnr(a, next, 320, 240, 3);
  VERIFY(moved > 5.0 && moved < 30.0);
  // One pixel off by one: MSE 1/(320*240*3).
  Bytes c = a;
  c[0] = static_cast<std::uint8_t>(c[0] ^ 1);
  const double expected = 10.0 * std::log10(255.0 * 255.0 * 320 * 240 * 3);
  VERIFY(std::abs(psnr(a, c, 320, 240, 3) - expected) < 1e-9);
  const auto rgba = render_test_frame(33, 17, 3, 4);
  VERIFY(rgba.size() == 33U * 17U * 4U && rgba[3] == 255);
  VERIFY(test::throws<std::invalid_argument>([] { (void)render_test_frame(0, 10, 0); }));
  VERIFY(test::throws<std::invalid_argument>([&] { (void)psnr(a, b, 321, 240, 3); }));
}

}  // namespace
}  // namespace holoscan::codec

int main() {
  using namespace holoscan::codec;  // NOLINT(build/namespaces)
  return test::run({
      {"H.264 access units", check_h264},
      {"H.265 access units", check_h265},
      {"degenerate streams", check_degenerate_streams},
      {"settings", check_settings},
      {"test video and PSNR", check_test_video},
  });
}
