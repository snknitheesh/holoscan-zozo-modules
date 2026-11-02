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
#include <sstream>
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

// An image's bytes copied to the host, after checking its size.
holoscan::expected<std::vector<std::uint8_t>, holoscan::Error> fetch(
    const holoscan::schema::ImageT& image, int width, int height, int channels,
    holoscan::ExecutionContext& context, const char* what) {
  const std::size_t bytes = static_cast<std::size_t>(width) * height * channels;
  if (!image.data || image.width != width || image.height != height ||
      static_cast<std::size_t>(image.data->nbytes()) != bytes) {
    return holoscan::make_unexpected(failure(std::string("the ") + what + " image is not " +
                                             std::to_string(width) + "x" + std::to_string(height) +
                                             " packed"));
  }
  std::vector<std::uint8_t> host(bytes);
  if (auto copied = image.data->copy_to_host(host.data(), host.size(), context.cuda_stream());
      !copied) {
    return holoscan::make_unexpected(std::move(copied).error());
  }
  return host;
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

void StitchCheckOp::stop() {
  if (truth_ != nullptr) {
    cudaFree(truth_);
    truth_ = nullptr;
  }
}

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
  const auto frame = static_cast<std::uint32_t>(bev->metadata.frame_id - 1U);
  const st::RigPose pose = frame_pose(frame);
  const std::size_t largest =
      std::max(static_cast<std::size_t>(config_.birds_eye.width) * config_.birds_eye.height,
               static_cast<std::size_t>(config_.panorama.width) * config_.panorama.height) *
      4U;
  if (truth_ == nullptr && cudaMalloc(&truth_, largest) != cudaSuccess) {
    truth_ = nullptr;
    return holoscan::make_unexpected(failure("cudaMalloc failed"));
  }
  if (birds_eye_resolution_.empty()) {
    // Where the cameras resolve the outputs well enough to place a corner.
    const st::Stitcher bev_tables(config_.rig, config_.birds_eye, st::BlendSettings{});
    const st::Stitcher pano_tables(config_.rig, config_.panorama, st::BlendSettings{});
    birds_eye_resolution_ = st::resolution_map(bev_tables.download_tables());
    panorama_resolution_ = st::resolution_map(pano_tables.download_tables());
  }
  struct Output {
    const holoscan::schema::ImageT* image;
    const st::OutputGeometry* geometry;
    const char* name;
    double min_psnr;
    const std::vector<float>* resolution;
  };
  const Thresholds& t = config_.thresholds;
  std::string failed;
  st::Comparison results[2];
  int index = 0;
  for (const Output& out : {Output{&bev->data, &config_.birds_eye, "birds_eye", t.birds_eye_psnr_db,
                                   &birds_eye_resolution_},
                            Output{&pano->data, &config_.panorama, "panorama", t.panorama_psnr_db,
                                   &panorama_resolution_}}) {
    const st::OutputGeometry& g = *out.geometry;
    auto stitched = fetch(*out.image, g.width, g.height, 4, context, out.name);
    if (!stitched) {
      return holoscan::make_unexpected(std::move(stitched).error());
    }
    const std::size_t pitch = static_cast<std::size_t>(g.width) * 4U;
    std::vector<std::uint8_t> truth(pitch * g.height);
    if (st::render_truth(config_.world, g, pose, truth_, pitch, 2, nullptr) != cudaSuccess ||
        cudaMemcpy(truth.data(), truth_, truth.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
      return holoscan::make_unexpected(failure("rendering the truth failed"));
    }
    const st::Comparison c =
        st::compare_with_truth(stitched->data(), pitch, truth.data(), pitch, g.width, g.height,
                               st::checker_corners(config_.world, g, pose), 2, *out.resolution);
    results[index++] = c;
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
  auto pixels = fetch(render->data, config_.width, config_.height, 4, context, "render");
  if (!pixels) {
    return holoscan::make_unexpected(std::move(pixels).error());
  }
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
            pixels->data() + (static_cast<std::size_t>(y) * config_.width + x) * 4U;
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
    for (std::size_t i = 0; i < pixels->size(); i += 4) {
      file.write(reinterpret_cast<const char*>(pixels->data() + i), 3);
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
