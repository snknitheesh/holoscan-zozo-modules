// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// VideoEncoder and VideoDecoder on every backend this machine has, for H.264 and H.265, with the
// synthetic test video: quality (PSNR) per rate-control setting, the bitstream's parameter sets
// and keyframes at the GOP, frame count, order and timestamps through a round trip, odd sizes, a
// change of size, empty and undecodable access units, and rejected settings and images. Exits 77
// (skipped) when no backend can both encode and decode here, saying why.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/codec/bitstream.hpp>
#include <holoscan/codec/test_video.hpp>
#include <holoscan/codec/video_codec.hpp>

#include "test_support.hpp"

namespace holoscan::codec {
namespace {

// A host image in device memory, freed on destruction.
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) { VERIFY(cudaMalloc(&data_, bytes) == cudaSuccess); }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  [[nodiscard]] std::uint8_t* get() const { return static_cast<std::uint8_t*>(data_); }

 private:
  void* data_{};
};

struct Stream {
  cudaStream_t stream{};
  Stream() { VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); }
  ~Stream() { cudaStreamDestroy(stream); }
};

// Encodes `frames` RGB8 frames of the test video, starting at frame `first`.
std::vector<EncodedFrame> encode_video(VideoEncoder& encoder, int width, int height,
                                       std::uint32_t frames, std::uint32_t first = 0,
                                       std::int64_t pts_start = 0, std::int64_t pts_step = 1) {
  Stream stream;
  const std::size_t row = static_cast<std::size_t>(width) * 3;
  DeviceBuffer device(row * height);
  std::vector<EncodedFrame> encoded;
  for (std::uint32_t i = 0; i < frames; ++i) {
    const auto pixels = render_test_frame(width, height, first + i);
    // On the encoder's stream: a non-blocking stream does not wait for cudaMemcpy's.
    VERIFY(cudaMemcpyAsync(device.get(), pixels.data(), pixels.size(), cudaMemcpyHostToDevice,
                           stream.stream) == cudaSuccess);
    const DeviceImage image{.data = device.get(), .pitch = row, .width = width, .height = height};
    encoded.push_back(encoder.encode(image, pts_start + pts_step * i, stream.stream));
  }
  return encoded;
}

// A decoded frame, as RGB8 on the host, cropped to `width` x `height`.
struct HostFrame {
  std::vector<std::uint8_t> rgb;
  int width{};
  int height{};
  int coded_width{};
  int coded_height{};
  std::int64_t pts{};
};

std::vector<HostFrame> to_host(const std::vector<DecodedFrame>& frames, int width, int height,
                               cudaStream_t stream) {
  std::vector<HostFrame> result;
  for (const DecodedFrame& frame : frames) {
    const int w = std::min(width, frame.image.width);
    const int h = std::min(height, frame.image.height);
    DeviceBuffer rgb(static_cast<std::size_t>(w) * h * 3);
    convert_from_nv12(frame.image, PixelFormat::kRgb8, rgb.get(), static_cast<std::size_t>(w) * 3,
                      w, h, stream);
    HostFrame host{.rgb = std::vector<std::uint8_t>(static_cast<std::size_t>(w) * h * 3),
                   .width = w,
                   .height = h,
                   .coded_width = frame.image.width,
                   .coded_height = frame.image.height,
                   .pts = frame.pts};
    VERIFY(cudaStreamSynchronize(stream) == cudaSuccess);
    VERIFY(cudaMemcpy(host.rgb.data(), rgb.get(), host.rgb.size(), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    result.push_back(std::move(host));
  }
  return result;
}

// Decodes `units` with one decoder, then flushes it. With `per_unit`, how many frames each
// access unit's decode() gave.
std::vector<HostFrame> decode_video(Codec codec, Backend backend,
                                    const std::vector<EncodedFrame>& units, int width, int height,
                                    bool low_latency = true,
                                    std::vector<std::size_t>* per_unit = nullptr) {
  Stream stream;
  VideoDecoder decoder(codec, backend, low_latency);
  std::vector<HostFrame> frames;
  for (const EncodedFrame& unit : units) {
    auto decoded =
        to_host(decoder.decode(unit.data, unit.pts, stream.stream), width, height, stream.stream);
    if (per_unit != nullptr) per_unit->push_back(decoded.size());
    for (auto& frame : decoded) frames.push_back(std::move(frame));
  }
  auto rest = to_host(decoder.flush(stream.stream), width, height, stream.stream);
  for (auto& frame : rest) frames.push_back(std::move(frame));
  return frames;
}

double frame_psnr(const HostFrame& frame, std::uint32_t source_index) {
  const auto source = render_test_frame(frame.width, frame.height, source_index);
  return psnr(source, frame.rgb, frame.width, frame.height, 3);
}

EncoderSettings settings_for(Codec codec, Backend backend) {
  EncoderSettings settings;
  settings.codec = codec;
  settings.backend = backend;
  return settings;
}

// The quality each rate-control setting must reach on the test video at 1280x720: the mean PSNR
// over its 30 frames, and the worst frame's. Rate control starts the stream with a small first
// keyframe, so with a bitrate the worst frame is taken after the first five.
struct QualityCase {
  const char* name;
  RateControl rate_control;
  int qp;
  std::uint32_t bitrate;
  double min_mean_psnr;
  double min_frame_psnr;
};

constexpr std::uint32_t kWarmUpFrames = 5;

constexpr QualityCase kQualityCases[] = {
    {"cqp 20", RateControl::kConstQp, 20, 0, 38.0, 36.0},
    {"cqp 30", RateControl::kConstQp, 30, 0, 32.0, 30.0},
    {"cqp 40", RateControl::kConstQp, 40, 0, 26.0, 24.0},
    {"cbr 8 Mbit/s", RateControl::kCbr, 0, 8'000'000, 36.0, 32.0},
    {"vbr 2 Mbit/s", RateControl::kVbr, 0, 2'000'000, 28.0, 19.0},
};

void check_quality(Codec codec, Backend backend) {
  constexpr int kWidth = 1280;
  constexpr int kHeight = 720;
  constexpr std::uint32_t kFrames = 30;
  for (const QualityCase& c : kQualityCases) {
    EncoderSettings settings = settings_for(codec, backend);
    settings.rate_control = c.rate_control;
    settings.qp = c.qp != 0 ? c.qp : settings.qp;
    settings.bitrate = c.bitrate != 0 ? c.bitrate : settings.bitrate;
    VideoEncoder encoder(settings);
    const auto units = encode_video(encoder, kWidth, kHeight, kFrames);
    const auto frames = decode_video(codec, backend, units, kWidth, kHeight);
    VERIFY(frames.size() == kFrames);
    const bool constant_qp = c.rate_control == RateControl::kConstQp;
    double sum = 0.0;
    double worst = 1e9;
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < frames.size(); ++i) {
      const double value = frame_psnr(frames[i], static_cast<std::uint32_t>(i));
      sum += value;
      if (constant_qp || i >= kWarmUpFrames) worst = std::min(worst, value);
    }
    for (const auto& unit : units) bytes += unit.data.size();
    const double mean = frames.empty() ? 0.0 : sum / static_cast<double>(frames.size());
    const double mbps = static_cast<double>(bytes) * 8.0 * settings.frame_rate / kFrames / 1e6;
    std::cout << "  " << codec_name(codec) << " " << backend_name(backend) << " " << c.name
              << ": mean PSNR " << mean << " dB, worst " << worst << " dB, " << mbps
              << " Mbit/s at 30 frames/s\n";
    VERIFY_MSG(mean >= c.min_mean_psnr, std::string(c.name) + ": mean PSNR " +
                                            std::to_string(mean) + " below " +
                                            std::to_string(c.min_mean_psnr));
    VERIFY_MSG(worst >= c.min_frame_psnr, std::string(c.name) + ": worst PSNR " +
                                              std::to_string(worst) + " below " +
                                              std::to_string(c.min_frame_psnr));
    if (!constant_qp) {
      // Easy content may take less than the target, never much more.
      VERIFY_MSG(mbps < 1.15 * c.bitrate / 1e6,
                 std::string(c.name) + " at " + std::to_string(mbps) + " Mbit/s");
    }
  }
}

// Parameter sets on every keyframe, keyframes exactly every GOP, P-frames between.
void check_bitstream(Codec codec, Backend backend) {
  constexpr std::uint32_t kGop = 10;
  constexpr std::uint32_t kFrames = 35;
  EncoderSettings settings = settings_for(codec, backend);
  settings.gop = kGop;
  VideoEncoder encoder(settings);
  VERIFY(encoder.backend() == backend);
  const auto units = encode_video(encoder, 640, 480, kFrames);
  VERIFY(units.size() == kFrames);
  std::vector<std::uint8_t> stream;
  for (std::uint32_t i = 0; i < units.size(); ++i) {
    const EncodedFrame& unit = units[i];
    const AccessUnitInfo info = analyze_access_unit(codec, unit.data);
    const bool expect_key = i % kGop == 0;
    VERIFY_MSG(unit.keyframe == expect_key,
               "frame " + std::to_string(i) + " keyframe " + std::to_string(unit.keyframe));
    VERIFY(unit.index == i);
    VERIFY(info.idr == expect_key);
    VERIFY(info.slices >= 1);
    if (expect_key) {
      VERIFY_MSG(info.sps && info.pps && (codec == Codec::kH264 || info.vps),
                 "keyframe " + std::to_string(i) + " lacks parameter sets");
      VERIFY(unit.data.size() >= 4 && unit.data[0] == 0 && unit.data[1] == 0);
    }
    stream.insert(stream.end(), unit.data.begin(), unit.data.end());
  }
  // The concatenated stream splits back into the same pictures.
  std::size_t slices = 0;
  for (const NalUnit& nal : split_nal_units(codec, stream)) {
    slices +=
        analyze_access_unit(codec, std::span(stream).subspan(nal.offset - 3, nal.size + 3)).slices;
  }
  VERIFY(slices >= kFrames);
}

// Every frame comes back once, in order, with its pts, and looks most like its own source: in
// low-latency mode from the decode() of its own access unit, otherwise by the flush at the end.
void check_round_trip(Codec codec, Backend backend, bool low_latency) {
  constexpr std::uint32_t kFrames = 40;
  constexpr std::int64_t kStart = 1'700'000'000'000'000'000;
  constexpr std::int64_t kStep = 33'333'333;
  EncoderSettings settings = settings_for(codec, backend);
  settings.rate_control = RateControl::kConstQp;
  settings.qp = 22;
  settings.gop = 12;
  VideoEncoder encoder(settings);
  const auto units = encode_video(encoder, 640, 480, kFrames, 0, kStart, kStep);
  std::vector<std::size_t> per_unit;
  const auto frames = decode_video(codec, backend, units, 640, 480, low_latency, &per_unit);
  VERIFY_MSG(frames.size() == kFrames, std::to_string(frames.size()) + " frames decoded");
  if (low_latency) {
    // Each frame comes with its own access unit, none is left for the flush. Jetson's V4L2
    // decoder gives a stream's first frames up to two access units late, then catches up.
    const std::size_t catch_up = backend == Backend::kV4l2 ? 3 : 0;
    std::size_t total = 0;
    for (std::size_t i = 0; i < per_unit.size(); ++i) {
      total += per_unit[i];
      VERIFY_MSG(i < catch_up || per_unit[i] == 1, "access unit " + std::to_string(i) + " gave " +
                                                       std::to_string(per_unit[i]) + " frames");
    }
    VERIFY_MSG(total == kFrames, std::to_string(kFrames - total) + " frames left for the flush");
  }
  for (std::size_t i = 0; i < frames.size(); ++i) {
    VERIFY_MSG(frames[i].pts == kStart + kStep * static_cast<std::int64_t>(i),
               "frame " + std::to_string(i) + " pts " + std::to_string(frames[i].pts));
    const auto index = static_cast<std::uint32_t>(i);
    const double own = frame_psnr(frames[i], index);
    if (i > 0) VERIFY(own > frame_psnr(frames[i], index - 1) + 3.0);
    if (i + 1 < frames.size()) VERIFY(own > frame_psnr(frames[i], index + 1) + 3.0);
  }
}

// Odd sides are coded one larger and cropped back; a size change starts a new stream.
void check_sizes(Codec codec, Backend backend) {
  {
    EncoderSettings settings = settings_for(codec, backend);
    settings.rate_control = RateControl::kConstQp;
    settings.qp = 22;
    VideoEncoder encoder(settings);
    const auto units = encode_video(encoder, 641, 361, 6);
    VERIFY(units.front().width == 641 && units.front().height == 361);
    const auto frames = decode_video(codec, backend, units, 641, 361);
    VERIFY(frames.size() == 6);
    for (std::size_t i = 0; i < frames.size(); ++i) {
      VERIFY(frames[i].coded_width == 642 && frames[i].coded_height == 362);
      VERIFY(frames[i].width == 641 && frames[i].height == 361);
      VERIFY(frame_psnr(frames[i], static_cast<std::uint32_t>(i)) > 35.0);
    }
  }
  {
    EncoderSettings settings = settings_for(codec, backend);
    settings.gop = 100;
    VideoEncoder encoder(settings);
    auto units = encode_video(encoder, 640, 480, 5);
    const auto bigger = encode_video(encoder, 800, 600, 5, 5);
    VERIFY(bigger.front().keyframe && bigger.front().index == 0);
    VERIFY(bigger.back().index == 4 && !bigger.back().keyframe);
    VERIFY(bigger.front().width == 800);
    // One decoder follows the new parameter sets.
    units.insert(units.end(), bigger.begin(), bigger.end());
    Stream stream;
    VideoDecoder decoder(codec, backend);
    std::vector<std::pair<int, int>> sizes;
    for (const EncodedFrame& unit : units) {
      for (const DecodedFrame& frame : decoder.decode(unit.data, unit.pts, stream.stream)) {
        sizes.emplace_back(frame.image.width, frame.image.height);
      }
    }
    for (const DecodedFrame& frame : decoder.flush(stream.stream)) {
      sizes.emplace_back(frame.image.width, frame.image.height);
    }
    VERIFY_MSG(sizes.size() == 10, std::to_string(sizes.size()) + " frames across the change");
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      VERIFY(sizes[i] == (i < 5 ? std::pair{640, 480} : std::pair{800, 600}));
    }
  }
  {
    // The smallest side the module takes, and one below it.
    VideoEncoder encoder(settings_for(codec, backend));
    VERIFY(encode_video(encoder, kMinFrameSide, kMinFrameSide, 2).size() == 2);
    VERIFY(test::throws<std::invalid_argument>(
        [&] { (void)encode_video(encoder, kMinFrameSide - 2, 200, 1); }));
  }
}

// Empty access units and those before the first keyframe give nothing; decoding starts at the
// next keyframe.
void check_stream_edges(Codec codec, Backend backend) {
  EncoderSettings settings = settings_for(codec, backend);
  settings.gop = 5;
  VideoEncoder encoder(settings);
  const auto units = encode_video(encoder, 640, 480, 12);
  Stream stream;
  VideoDecoder decoder(codec, backend);
  VERIFY(decoder.decode({}, 0, stream.stream).empty());
  VERIFY(decoder.flush(stream.stream).empty());  // Nothing decoded yet: nothing held.
  std::vector<std::int64_t> pts;
  // Join at frame 2: frames 2 to 4 are skipped, decoding starts at the keyframe at 5.
  for (std::size_t i = 2; i < units.size(); ++i) {
    for (const DecodedFrame& frame : decoder.decode(units[i].data, units[i].pts, stream.stream)) {
      pts.push_back(frame.pts);
    }
  }
  for (const DecodedFrame& frame : decoder.flush(stream.stream)) pts.push_back(frame.pts);
  VERIFY(decoder.skipped() == 3);
  VERIFY_MSG(pts.size() == 7, std::to_string(pts.size()) + " frames after joining");
  for (std::size_t i = 0; i < pts.size(); ++i) VERIFY(pts[i] == static_cast<std::int64_t>(i + 5));
}

void check_rejections() {
  EncoderSettings bad;
  bad.qp = 52;
  bad.rate_control = RateControl::kConstQp;
  VERIFY(test::throws<std::invalid_argument>([&] { VideoEncoder encoder(bad); }));
  bad = {};
  bad.gop = 0;
  VERIFY(test::throws<std::invalid_argument>([&] { VideoEncoder encoder(bad); }));
  bad = {};
  bad.bitrate = 100;
  VERIFY(test::throws<std::invalid_argument>([&] { VideoEncoder encoder(bad); }));
}

std::string describe(const std::vector<BackendStatus>& statuses) {
  std::string text;
  for (const BackendStatus& status : statuses) {
    text += std::string(backend_name(status.backend)) + ": " +
            (status.available ? "available, " : "unavailable, ") + status.detail + "; ";
  }
  return text;
}

}  // namespace
}  // namespace holoscan::codec

int main() {
  using namespace holoscan::codec;  // NOLINT(build/namespaces)
  std::vector<test::TestCase> cases;
  cases.push_back({"rejected settings", check_rejections});
  int pairs = 0;
  for (const Codec codec : {Codec::kH264, Codec::kH265}) {
    const auto encoders = probe_encoder_backends(codec);
    const auto decoders = probe_decoder_backends(codec);
    std::cout << codec_name(codec) << " encoders: " << describe(encoders) << '\n'
              << codec_name(codec) << " decoders: " << describe(decoders) << '\n';
    for (std::size_t i = 0; i < encoders.size(); ++i) {
      if (!encoders[i].available || !decoders[i].available) {
        continue;
      }
      ++pairs;
      const Backend backend = encoders[i].backend;
      const std::string name =
          std::string(codec_name(codec)) + " on " + std::string(backend_name(backend));
      cases.push_back({name + ": quality per setting", [=] { check_quality(codec, backend); }});
      cases.push_back(
          {name + ": parameter sets and GOP", [=] { check_bitstream(codec, backend); }});
      cases.push_back({name + ": count, order and pts, low latency",
                       [=] { check_round_trip(codec, backend, true); }});
      cases.push_back({name + ": count, order and pts, reordering decoder",
                       [=] { check_round_trip(codec, backend, false); }});
      cases.push_back(
          {name + ": odd sizes and a size change", [=] { check_sizes(codec, backend); }});
      cases.push_back({name + ": empty and undecodable access units",
                       [=] { check_stream_edges(codec, backend); }});
    }
  }
  if (pairs == 0) {
    std::cout << "SKIPPED: no backend can encode and decode here (see above)\n";
    return test::kSkipped;
  }
  return test::run(cases);
}
