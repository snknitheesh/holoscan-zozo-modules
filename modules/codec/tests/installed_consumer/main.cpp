// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-codec package: it builds both operators, and
// encodes ten frames of the test video with VideoEncoder and decodes them with VideoDecoder,
// checking their count and quality. Exits 77 (skipped) when no backend can encode and decode.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include <holoscan/codec/test_video.hpp>
#include <holoscan/codec/video_codec.hpp>
#include <holoscan/codec/video_decoder_op.hpp>
#include <holoscan/codec/video_encoder_op.hpp>

int main() {
  namespace hc = holoscan::codec;
  const hc::VideoEncoderOp encoder_op(hc::VideoEncoderConfig{});
  const hc::VideoDecoderOp decoder_op(hc::VideoDecoderConfig{});
  const auto available = [](const std::vector<hc::BackendStatus>& statuses) {
    return std::any_of(statuses.begin(), statuses.end(),
                       [](const hc::BackendStatus& status) { return status.available; });
  };
  if (!available(hc::probe_encoder_backends(hc::Codec::kH264)) ||
      !available(hc::probe_decoder_backends(hc::Codec::kH264))) {
    std::cout << "codec_consumer: SKIPPED, no H.264 backend here\n";
    return 77;
  }
  constexpr int kWidth = 640;
  constexpr int kHeight = 480;
  constexpr std::size_t kBytes = std::size_t{kWidth} * kHeight * 3;
  cudaStream_t stream = nullptr;
  void* rgb = nullptr;
  void* out = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess || cudaMalloc(&rgb, kBytes) != cudaSuccess ||
      cudaMalloc(&out, kBytes) != cudaSuccess) {
    std::cout << "codec_consumer: FAIL, CUDA\n";
    return 1;
  }
  hc::EncoderSettings settings;
  settings.rate_control = hc::RateControl::kConstQp;
  settings.qp = 22;
  hc::VideoEncoder encoder(settings);
  hc::VideoDecoder decoder(hc::Codec::kH264);
  int decoded = 0;
  double worst = 1e9;
  for (std::uint32_t i = 0; i < 10; ++i) {
    const auto pixels = hc::render_test_frame(kWidth, kHeight, i);
    cudaMemcpyAsync(rgb, pixels.data(), kBytes, cudaMemcpyHostToDevice, stream);
    const hc::EncodedFrame frame =
        encoder.encode(hc::DeviceImage{.data = static_cast<const std::uint8_t*>(rgb),
                                       .pitch = std::size_t{kWidth} * 3,
                                       .width = kWidth,
                                       .height = kHeight},
                       i, stream);
    for (const hc::DecodedFrame& image : decoder.decode(frame.data, frame.pts, stream)) {
      hc::convert_from_nv12(image.image, hc::PixelFormat::kRgb8, static_cast<std::uint8_t*>(out),
                            std::size_t{kWidth} * 3, kWidth, kHeight, stream);
      std::vector<std::uint8_t> host(kBytes);
      cudaStreamSynchronize(stream);
      cudaMemcpy(host.data(), out, kBytes, cudaMemcpyDeviceToHost);
      const auto source =
          hc::render_test_frame(kWidth, kHeight, static_cast<std::uint32_t>(image.pts));
      worst = std::min(worst, hc::psnr(source, host, kWidth, kHeight, 3));
      ++decoded;
    }
  }
  decoded += static_cast<int>(decoder.flush(stream).size());
  cudaFree(rgb);
  cudaFree(out);
  cudaStreamDestroy(stream);
  const bool passed = decoded == 10 && worst > 35.0;
  std::cout << "codec_consumer: " << hc::backend_name(encoder.backend()) << ", " << decoded
            << " frames, worst PSNR " << worst << " dB: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
