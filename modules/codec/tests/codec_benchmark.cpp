// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Encode and decode throughput of VideoEncoder and VideoDecoder at 1920x1080 and 3840x2160, for
// each codec, on the backend `auto` picks: frames of the test video already in device memory,
// encoded one after another as a low-latency stream, then the access units decoded, after ten
// untimed frames that build and warm up the sessions. Each figure is frames per second of the
// whole call: color conversion, the hardware, and the copies in and out. Prints a table; exits 77
// (skipped) when no backend can encode and decode here.
//
//   codec_benchmark [--frames N] [--bitrate-1080p BPS] [--bitrate-2160p BPS]

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/codec/test_video.hpp>
#include <holoscan/codec/video_codec.hpp>

namespace {

using holoscan::codec::Backend;
using holoscan::codec::Codec;

struct Size {
  int width;
  int height;
  std::uint32_t bitrate;
};

// Distinct frames cycled through, so the encoder sees motion.
constexpr std::uint32_t kDistinctFrames = 8;
// Frames before the timing starts, while the sessions are built and warm up.
constexpr std::uint32_t kWarmUpFrames = 10;

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

struct Result {
  double encode_fps{};
  double decode_fps{};
  double mbps{};
  std::string backend;
};

Result measure(Codec codec, const Size& size, std::uint32_t frames) {
  namespace hc = holoscan::codec;
  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  const std::size_t row = static_cast<std::size_t>(size.width) * 3;
  const std::size_t bytes = row * static_cast<std::size_t>(size.height);
  std::vector<std::uint8_t*> images(kDistinctFrames);
  for (std::uint32_t i = 0; i < kDistinctFrames; ++i) {
    void* memory = nullptr;
    check(cudaMalloc(&memory, bytes), "cudaMalloc");
    images[i] = static_cast<std::uint8_t*>(memory);
    const auto pixels = hc::render_test_frame(size.width, size.height, i);
    check(cudaMemcpy(images[i], pixels.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
  }
  hc::EncoderSettings settings;
  settings.codec = codec;
  settings.bitrate = size.bitrate;
  settings.gop = 30;
  hc::VideoEncoder encoder(settings);
  hc::VideoDecoder decoder(codec);
  Result result;
  result.backend = std::string(hc::backend_name(encoder.backend())) + " / " +
                   std::string(hc::backend_name(decoder.backend()));

  const auto encode_one = [&](std::uint32_t i) {
    const hc::DeviceImage image{.data = images[i % kDistinctFrames],
                                .pitch = row,
                                .width = size.width,
                                .height = size.height};
    return encoder.encode(image, i, stream);
  };
  // The first frames build the hardware session and warm it up: not timed.
  std::vector<hc::EncodedFrame> units;
  for (std::uint32_t i = 0; i < kWarmUpFrames; ++i) units.push_back(encode_one(i));
  const auto encode_start = std::chrono::steady_clock::now();
  std::size_t total = 0;
  for (std::uint32_t i = kWarmUpFrames; i < kWarmUpFrames + frames; ++i) {
    units.push_back(encode_one(i));
    total += units.back().data.size();
  }
  const std::chrono::duration<double> encode_time = std::chrono::steady_clock::now() - encode_start;
  result.encode_fps = frames / encode_time.count();
  result.mbps = static_cast<double>(total) * 8.0 * settings.frame_rate / frames / 1e6;

  for (std::uint32_t i = 0; i < kWarmUpFrames; ++i) {
    (void)decoder.decode(units[i].data, i, stream);
  }
  std::uint32_t decoded = 0;
  const auto decode_start = std::chrono::steady_clock::now();
  for (std::uint32_t i = kWarmUpFrames; i < kWarmUpFrames + frames; ++i) {
    decoded += static_cast<std::uint32_t>(decoder.decode(units[i].data, i, stream).size());
  }
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  const std::chrono::duration<double> decode_time = std::chrono::steady_clock::now() - decode_start;
  result.decode_fps = decoded / decode_time.count();

  for (std::uint8_t* image : images) cudaFree(image);
  cudaStreamDestroy(stream);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t frames = 120;
  std::vector<Size> sizes{{1920, 1080, 8'000'000}, {3840, 2160, 20'000'000}};
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string_view option{argv[i]};
    const auto value = static_cast<std::uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
    if (option == "--frames") {
      frames = value;
    } else if (option == "--bitrate-1080p") {
      sizes[0].bitrate = value;
    } else if (option == "--bitrate-2160p") {
      sizes[1].bitrate = value;
    } else {
      std::cerr << "unknown option " << option << '\n';
      return 2;
    }
  }
  if (frames == 0) {
    std::cerr << "--frames must be positive\n";
    return 2;
  }
  int measured = 0;
  int failed = 0;
  std::printf("%-5s %-10s %-18s %12s %12s %10s\n", "codec", "size", "backend (enc/dec)",
              "encode fps", "decode fps", "Mbit/s");
  for (const Codec codec : {Codec::kH264, Codec::kH265}) {
    const auto usable = [](const std::vector<holoscan::codec::BackendStatus>& statuses) {
      for (const auto& status : statuses) {
        if (status.available) return true;
      }
      return false;
    };
    const std::string name(holoscan::codec::codec_name(codec));
    if (!usable(holoscan::codec::probe_encoder_backends(codec)) ||
        !usable(holoscan::codec::probe_decoder_backends(codec))) {
      std::printf("%-5s no backend encodes and decodes it here\n", name.c_str());
      continue;
    }
    for (const Size& size : sizes) {
      const std::string label = std::to_string(size.width) + "x" + std::to_string(size.height);
      try {
        const Result r = measure(codec, size, frames);
        std::printf("%-5s %-10s %-18s %12.1f %12.1f %10.2f\n", name.c_str(), label.c_str(),
                    r.backend.c_str(), r.encode_fps, r.decode_fps, r.mbps);
        ++measured;
      } catch (const std::exception& error) {
        std::printf("%-5s %-10s failed: %s\n", name.c_str(), label.c_str(), error.what());
        ++failed;
      }
    }
  }
  if (failed != 0) {
    return 1;
  }
  if (measured == 0) {
    std::cout << "SKIPPED: no backend encodes and decodes here\n";
    return 77;
  }
  return 0;
}
