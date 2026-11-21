// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "check_ops.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/stitching/remap.hpp>
#include <holoscan/stitching/stitcher.hpp>

#include "rig_source_op.hpp"

namespace stitching_app {
namespace {

namespace st = holoscan::stitching;
using std::chrono_literals::operator""ns;

holoscan::Error failure(const std::string& what) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "check: " + what};
}

// Queues the copy of an image's bytes to `host` on `stream`, after checking
// its size.
holoscan::expected<void, holoscan::Error> fetch(const holoscan::schema::ImageT& image, int width,
                                                int height, int channels, void* host,
                                                cudaStream_t stream, const char* what) {
  const std::size_t bytes = static_cast<std::size_t>(width) * height * channels;
  if (!image.data || image.width != static_cast<std::uint32_t>(width) ||
      image.height != static_cast<std::uint32_t>(height) ||
      static_cast<std::size_t>(image.data->nbytes()) != bytes) {
    return holoscan::make_unexpected(failure(std::string("the ") + what + " image is not " +
                                             std::to_string(width) + "x" + std::to_string(height) +
                                             " packed"));
  }
  return image.data->copy_to_host_async(host, bytes, stream);
}

std::size_t rgba_bytes(const st::OutputGeometry& g) {
  return static_cast<std::size_t>(g.width) * g.height * 4U;
}

void keep_worst(Worst& worst, const st::Comparison& c) {
  worst.psnr_db = std::min(worst.psnr_db, c.psnr_db);
  worst.coverage = std::min(worst.coverage, c.coverage);
  worst.max_corner_error = std::max(worst.max_corner_error, c.max_corner_error);
  worst.gain_error = std::max(worst.gain_error, std::abs(c.gain - 1.0));
  worst.corners = std::min(worst.corners, c.corners);
}

std::string describe(const char* what, const st::Comparison& c) {
  std::ostringstream out;
  out << what << ": psnr " << std::fixed << std::setprecision(2) << c.psnr_db << " dB, coverage "
      << c.coverage << ", gain " << std::setprecision(3) << c.gain << ", " << c.corners
      << " corners, worst " << c.max_corner_error << " px";
  return out.str();
}

}  // namespace

StitchCheckOp::StitchCheckOp(StitchCheckConfig config) : config_(std::move(config)) {}

StitchCheckOp::~StitchCheckOp() { release(); }

void StitchCheckOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(birds_eye, "birds_eye").queue_depth(2U);
  spec.input(panorama, "panorama").queue_depth(2U);
}

holoscan::Contract StitchCheckOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{birds_eye, panorama}},
      .window = 0ns});
  result.input(birds_eye, holoscan::Exact{});
  result.input(panorama, holoscan::Exact{});
  return result;
}

void StitchCheckOp::release() noexcept {
  for (std::size_t i = 0; i < 2; ++i) {
    if (truth_device_[i] != nullptr) cudaFree(truth_device_[i]);
    if (stitched_host_[i] != nullptr) cudaFreeHost(stitched_host_[i]);
    if (truth_host_[i] != nullptr) cudaFreeHost(truth_host_[i]);
    truth_device_[i] = stitched_host_[i] = truth_host_[i] = nullptr;
  }
}

void StitchCheckOp::start() {
  release();
  const std::array<const st::OutputGeometry*, 2> outputs{&config_.birds_eye, &config_.panorama};
  for (std::size_t i = 0; i < 2; ++i) {
    const std::size_t bytes = rgba_bytes(*outputs[i]);
    if (cudaMalloc(&truth_device_[i], bytes) != cudaSuccess ||
        cudaMallocHost(&stitched_host_[i], bytes) != cudaSuccess ||
        cudaMallocHost(&truth_host_[i], bytes) != cudaSuccess) {
      release();
      throw std::runtime_error("StitchCheckOp: out of memory for the check's buffers");
    }
  }
  // Where the cameras resolve the outputs well enough to place a corner: from
  // the tables, built once on the GPU and released.
  {
    const st::Stitcher tables(config_.rig, config_.birds_eye, st::BlendSettings{});
    birds_eye_resolution_ = st::resolution_map(tables.download_tables());
  }
  {
    const st::Stitcher tables(config_.rig, config_.panorama, st::BlendSettings{});
    panorama_resolution_ = st::resolution_map(tables.download_tables());
  }
}

void StitchCheckOp::stop() { release(); }

holoscan::expected<void, holoscan::Error> StitchCheckOp::compute(
    holoscan::ExecutionContext& context) {
  auto bev = birds_eye.receive();
  if (!bev) {
    return holoscan::make_unexpected(std::move(bev).error());
  }
  auto pano = panorama.receive();
  if (!pano) {
    return holoscan::make_unexpected(std::move(pano).error());
  }
  if (bev->metadata.frame_id == 0 || bev->metadata.frame_id != pano->metadata.frame_id) {
    return holoscan::make_unexpected(failure("the two outputs are not of one frame"));
  }
  if (truth_device_[0] == nullptr) {
    return holoscan::make_unexpected(failure("compute() before start()"));
  }
  const auto frame = static_cast<std::uint32_t>(bev->metadata.frame_id - 1U);
  const st::RigPose pose = frame_pose(frame);
  struct Output {
    const holoscan::schema::ImageT* image;
    const st::OutputGeometry* geometry;
    const char* name;
    double min_psnr;
    const std::vector<float>* resolution;
  };
  const Thresholds& t = config_.thresholds;
  const std::array<Output, 2> outputs{Output{&bev->data, &config_.birds_eye, "birds_eye",
                                             t.birds_eye_psnr_db, &birds_eye_resolution_},
                                      Output{&pano->data, &config_.panorama, "panorama",
                                             t.panorama_psnr_db, &panorama_resolution_}};
  // Both outputs and their truths to the host on this operator's stream; one
  // wait for all four copies.
  cudaStream_t stream = context.cuda_stream().get();
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const st::OutputGeometry& g = *outputs[i].geometry;
    const std::size_t pitch = static_cast<std::size_t>(g.width) * 4U;
    if (auto fetched = fetch(*outputs[i].image, g.width, g.height, 4, stitched_host_[i], stream,
                             outputs[i].name);
        !fetched) {
      return fetched;
    }
    if (st::render_truth(config_.world, g, pose, truth_device_[i], pitch, 2, stream) !=
            cudaSuccess ||
        cudaMemcpyAsync(truth_host_[i], truth_device_[i], rgba_bytes(g), cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess) {
      return holoscan::make_unexpected(failure("rendering the truth failed"));
    }
  }
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(failure(cudaGetErrorString(status)));
  }
  std::string failed;
  std::array<st::Comparison, 2> results{};
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const Output& out = outputs[i];
    const st::OutputGeometry& g = *out.geometry;
    const std::size_t pitch = static_cast<std::size_t>(g.width) * 4U;
    const st::Comparison c = st::compare_with_truth(
        static_cast<const std::uint8_t*>(stitched_host_[i]), pitch,
        static_cast<const std::uint8_t*>(truth_host_[i]), pitch, g.width, g.height,
        st::checker_corners(config_.world, g, pose), 2, *out.resolution);
    results[i] = c;
    if (c.psnr_db < out.min_psnr || c.coverage < t.coverage ||
        c.max_corner_error > t.corner_error_px || std::abs(c.gain - 1.0) > t.gain_error ||
        c.corners < t.corners) {
      failed += (failed.empty() ? "" : "; ") + describe(out.name, c);
    }
  }
  {
    const std::lock_guard lock(config_.report->mutex);
    keep_worst(config_.report->birds_eye, results[0]);
    keep_worst(config_.report->panorama, results[1]);
    if (!failed.empty()) {
      ++config_.report->failed_frames;
      if (config_.report->first_failure.empty()) {
        config_.report->first_failure = "frame " + std::to_string(frame) + ": " + failed;
      }
    }
  }
  config_.report->frames.fetch_add(1);
  return {};
}

RenderCheckOp::RenderCheckOp(RenderCheckConfig config) : config_(std::move(config)) {}

RenderCheckOp::~RenderCheckOp() { stop(); }

void RenderCheckOp::start() {
  stop();
  const std::size_t bytes = static_cast<std::size_t>(config_.width) * config_.height * 4U;
  if (cudaMallocHost(&pixels_, bytes) != cudaSuccess) {
    pixels_ = nullptr;
    throw std::runtime_error("RenderCheckOp: out of memory for a render");
  }
}

void RenderCheckOp::stop() {
  if (pixels_ != nullptr) {
    cudaFreeHost(pixels_);
    pixels_ = nullptr;
  }
}

void RenderCheckOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(rendered, "rendered").queue_depth(1U);
}

holoscan::Contract RenderCheckOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{rendered});
  return result;
}

holoscan::expected<void, holoscan::Error> RenderCheckOp::compute(
    holoscan::ExecutionContext& context) {
  auto render = rendered.receive();
  if (!render) {
    return holoscan::make_unexpected(std::move(render).error());
  }
  if (pixels_ == nullptr) {
    return holoscan::make_unexpected(failure("compute() before start()"));
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (auto fetched =
          fetch(render->data, config_.width, config_.height, 4, pixels_, stream, "render");
      !fetched) {
    return fetched;
  }
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(failure(cudaGetErrorString(status)));
  }
  const std::span<const std::uint8_t> render_bytes(
      static_cast<const std::uint8_t*>(pixels_),
      static_cast<std::size_t>(config_.width) * config_.height * 4U);
  const std::uint32_t number = config_.report->rendered.load();
  // Every tile shows a picture: its luminance varies.
  std::string failed;
  for (std::size_t i = 0; i < config_.tiles.size(); ++i) {
    const Tile& tile = config_.tiles[i];
    double sum = 0.0;
    double squares = 0.0;
    std::size_t n = 0;
    for (int y = tile.y + 2; y < tile.y + tile.height - 2; y += 2) {
      for (int x = tile.x + 2; x < tile.x + tile.width - 2; x += 2) {
        const std::uint8_t* p =
            render_bytes.data() + (static_cast<std::size_t>(y) * config_.width + x) * 4U;
        const double l = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
        sum += l;
        squares += l * l;
        ++n;
      }
    }
    const double mean = n == 0 ? 0.0 : sum / n;
    const double deviation = n == 0 ? 0.0 : std::sqrt(std::max(0.0, squares / n - mean * mean));
    if (deviation < 10.0) {
      failed += (failed.empty() ? "tile " : ", tile ") + std::to_string(i) + " is flat";
    }
  }
  if (!config_.image_dir.empty()) {
    char name[32];
    std::snprintf(name, sizeof(name), "render_%06u.ppm", number);
    std::ofstream file(config_.image_dir / name, std::ios::binary);
    file << "P6\n" << config_.width << ' ' << config_.height << "\n255\n";
    for (std::size_t i = 0; i < render_bytes.size(); i += 4) {
      file.write(reinterpret_cast<const char*>(render_bytes.data() + i), 3);
    }
    if (!file) {
      failed += (failed.empty() ? "" : "; ") + std::string("writing ") + name + " failed";
    }
  }
  if (!failed.empty()) {
    const std::lock_guard lock(config_.report->mutex);
    ++config_.report->failed_renders;
    if (config_.report->first_failure.empty()) {
      config_.report->first_failure = "render " + std::to_string(number) + ": " + failed;
    }
  }
  config_.report->rendered.fetch_add(1);
  return {};
}

}  // namespace stitching_app
