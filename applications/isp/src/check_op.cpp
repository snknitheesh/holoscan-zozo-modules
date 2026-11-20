// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "check_op.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "frame_check.hpp"

namespace isp_demo {
namespace {

using std::chrono_literals::operator""ns;
namespace isp = holoscan::isp;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "CheckOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         std::string("CheckOp: ") + what + ": " + cudaGetErrorString(status)};
}

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

bool on_gpu(const holoscan::Tensor& tensor) {
  const DLDevice device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

// The host path: the same per-pixel arithmetic as the GPU's.
FrameCheckResult check_frame_host(const isp::SyntheticRawScene& scene,
                                  const std::vector<float>& lut,
                                  const std::vector<std::uint8_t>& rgb,
                                  const std::vector<std::uint32_t>& histogram,
                                  const float* stats) {
  FrameCheckResult result{};
  std::vector<std::uint32_t> rebuilt(kHistogramEntries, 0U);
  const int lut_size = static_cast<int>(lut.size());
  for (int y = 0; y < scene.height; ++y) {
    for (int x = 0; x < scene.width; ++x) {
      int truth[3];
      truth_codes(scene, lut.data(), lut_size, x, y, truth);
      const std::uint8_t* pixel = rgb.data() + (static_cast<std::size_t>(y) * scene.width + x) * 3U;
      for (int k = 0; k < 3; ++k) {
        const int d = static_cast<int>(pixel[k]) - truth[k];
        result.squared_error += static_cast<unsigned long long>(d * d);
      }
      ++rebuilt[isp::kHistogramLuma * isp::kHistogramBins + luma_bin(pixel[0], pixel[1], pixel[2])];
      ++rebuilt[isp::kHistogramRed * isp::kHistogramBins + pixel[0]];
      ++rebuilt[isp::kHistogramGreen * isp::kHistogramBins + pixel[1]];
      ++rebuilt[isp::kHistogramBlue * isp::kHistogramBins + pixel[2]];
    }
  }
  for (int i = 0; i < kHistogramEntries; ++i) {
    result.histogram_mismatches += rebuilt[i] != histogram[i] ? 1U : 0U;
  }
  std::memcpy(result.stats, stats, sizeof(result.stats));
  return result;
}

}  // namespace

CheckOp::CheckOp(CheckConfig config)
    : config_(std::move(config)), lut_(isp::tone_lut(config_.params)) {
  if (!config_.report) {
    throw std::invalid_argument("CheckOp needs a report");
  }
}

CheckOp::~CheckOp() { release(); }

void CheckOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots each: the frame being checked, and the next one IspOp sends
  // meanwhile; the graph runs on one worker, so no more can arrive.
  spec.input(image, "image").queue_depth(2U);
  spec.input(stats, "stats").queue_depth(2U);
  spec.input(histogram, "histogram").queue_depth(2U);
}

holoscan::Contract CheckOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{image, stats, histogram}},
      .window = 0ns});
  result.input(image, holoscan::Exact{});
  result.input(stats, holoscan::Exact{});
  result.input(histogram, holoscan::Exact{});
  return result;
}

void CheckOp::start() {
  start_error_.clear();
  try {
    check(cudaMalloc(reinterpret_cast<void**>(&device_lut_), lut_.size() * sizeof(float)),
          "cudaMalloc");
    check(cudaMalloc(reinterpret_cast<void**>(&device_bins_),
                     sizeof(std::uint32_t) * kHistogramEntries),
          "cudaMalloc");
    check(cudaMalloc(&device_result_, sizeof(FrameCheckResult)), "cudaMalloc");
    check(cudaMallocHost(&host_result_, sizeof(FrameCheckResult)), "cudaMallocHost");
    // A copy from pageable memory may return before it has landed: wait for it.
    check(cudaMemcpy(device_lut_, lut_.data(), lut_.size() * sizeof(float),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  } catch (const std::exception& error) {
    // Reported by compute(), which can fail the frame.
    start_error_ = error.what();
    release();
  }
}

void CheckOp::stop() { release(); }

void CheckOp::release() noexcept {
  // cudaFree waits for the work that may still use the buffers.
  cudaFree(device_lut_);
  cudaFree(device_bins_);
  cudaFree(device_result_);
  cudaFreeHost(host_result_);
  device_lut_ = nullptr;
  device_bins_ = nullptr;
  device_result_ = nullptr;
  host_result_ = nullptr;
}

holoscan::expected<void, holoscan::Error> CheckOp::compute(holoscan::ExecutionContext& context) {
  if (!start_error_.empty()) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "CheckOp: start: " + start_error_});
  }
  auto picture = image.receive();
  if (!picture) {
    return holoscan::make_unexpected(std::move(picture).error());
  }
  auto values = stats.receive();
  if (!values) {
    return holoscan::make_unexpected(std::move(values).error());
  }
  auto bins = histogram.receive();
  if (!bins) {
    return holoscan::make_unexpected(std::move(bins).error());
  }
  const int width = config_.camera.width;
  const int height = config_.camera.height;
  if (!picture->data.data || picture->data.encoding != holoscan::schema::ImageEncoding_RGB8 ||
      picture->data.width != width || picture->data.height != height) {
    return holoscan::make_unexpected(invalid("expected the camera's size in RGB8"));
  }
  const holoscan::Tensor& pixels = *picture->data.data;
  if (pixels.nbytes() != static_cast<std::int64_t>(width) * height * 3 ||
      bins->data.nbytes() != static_cast<std::int64_t>(sizeof(std::uint32_t)) * kHistogramEntries ||
      values->data.nbytes() < static_cast<std::int64_t>(sizeof(float)) * isp::stats::kStatsSize) {
    return holoscan::make_unexpected(invalid("unexpected image, histogram or statistics size"));
  }

  // The frame the source rendered: frame ids count from 1.
  isp::SyntheticRawConfig camera = config_.camera;
  const std::uint64_t id = picture->metadata.frame_id;
  camera.frame = static_cast<std::uint32_t>(id == 0 ? 0 : id - 1);
  const isp::SyntheticRawScene scene = isp::make_synthetic_scene(camera);

  FrameCheckResult result{};
  if (on_gpu(pixels)) {
    // The runtime orders the inputs on this stream; one wait, for the result.
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "CheckOp: the runtime gave no stream"});
    }
    if (const cudaError_t status = check_frame_cuda(
            scene, device_lut_, static_cast<int>(lut_.size()),
            static_cast<const std::uint8_t*>(pixels.data()),
            static_cast<const std::uint32_t*>(bins->data.data()),
            static_cast<const float*>(values->data.data()), device_bins_,
            static_cast<FrameCheckResult*>(device_result_), stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("check", status));
    }
    if (const cudaError_t status = cudaMemcpyAsync(host_result_, device_result_,
                                                   sizeof(FrameCheckResult),
                                                   cudaMemcpyDeviceToHost, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
    }
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
    }
    std::memcpy(&result, host_result_, sizeof(result));
  } else {
    pixels_.resize(static_cast<std::size_t>(width) * height * 3U);
    if (auto copied = pixels.copy_to_host(pixels_.data(), pixels_.size(), context.cuda_stream());
        !copied) {
      return copied;
    }
    std::array<float, isp::stats::kStatsSize> numbers{};
    if (auto copied =
            values->data.copy_to_host(numbers.data(), sizeof(numbers), context.cuda_stream());
        !copied) {
      return copied;
    }
    counts_.resize(kHistogramEntries);
    if (auto copied = bins->data.copy_to_host(counts_.data(), counts_.size() * 4U,
                                              context.cuda_stream());
        !copied) {
      return copied;
    }
    result = check_frame_host(scene, lut_, pixels_, counts_, numbers.data());
  }

  Report& report = *config_.report;
  report.frame_gaps += id != report.frames + 1U ? 1U : 0U;
  report.bad_histograms += result.histogram_mismatches != 0U ? 1U : 0U;
  const double squared = static_cast<double>(result.squared_error);
  const double psnr =
      squared == 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 * 3.0 * width * height / squared);
  report.worst_psnr = std::min(report.worst_psnr, psnr);
  report.psnr_sum += psnr;
  report.wb_gains = {result.stats[isp::stats::kWbGainR], result.stats[isp::stats::kWbGainG],
                     result.stats[isp::stats::kWbGainB]};
  report.ae_error_ev = result.stats[isp::stats::kAeErrorEv];
  report.mean_luma = result.stats[isp::stats::kMeanLuma];
  ++report.frames;
  if (report.frames == config_.expected_frames) {
    context.request_completion();
  }
  return {};
}

}  // namespace isp_demo
