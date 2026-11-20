// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Drawing clusters on the GPU, for display: labels as a color image, and each
// cluster's box projected through a pinhole camera. The cloud must be in that
// camera's optical frame (x right, y down, z forward), as DepthToPointCloudOp
// makes it. Device memory, ordered on the given stream.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define HOLOSCAN_CLUSTERING_RENDER_HOST_DEVICE __host__ __device__
#else
#define HOLOSCAN_CLUSTERING_RENDER_HOST_DEVICE
#endif

namespace holoscan::clustering {

// The display color of a cluster id: hues a golden angle apart, so that
// neighboring ids differ; dark gray for -1.
HOLOSCAN_CLUSTERING_RENDER_HOST_DEVICE inline void label_color(std::int32_t label,
                                                               std::uint8_t rgb[3]) {
  if (label < 0) {
    rgb[0] = 40U;
    rgb[1] = 40U;
    rgb[2] = 40U;
    return;
  }
  const float hue = 6.0F * (static_cast<float>(label) * 0.6180340F -
                            floorf(static_cast<float>(label) * 0.6180340F));
  const int sector = static_cast<int>(hue) % 6;
  const float f = hue - floorf(hue);
  constexpr float kValue = 240.0F;
  constexpr float kLow = 60.0F;
  const float rising = kLow + (kValue - kLow) * f;
  const float falling = kValue - (kValue - kLow) * f;
  const float table[6][3] = {{kValue, rising, kLow}, {falling, kValue, kLow},
                             {kLow, kValue, rising}, {kLow, falling, kValue},
                             {rising, kLow, kValue}, {kValue, kLow, falling}};
  for (int c = 0; c < 3; ++c) {
    rgb[c] = static_cast<std::uint8_t>(table[sector][c]);
  }
}

// Writes label_color() of each of `count` labels into packed RGB8.
cudaError_t colorize_labels(const std::int32_t* labels, std::uint32_t count, std::uint8_t* rgb,
                            cudaStream_t stream);

// Draws the 12 edges of each cluster's box, oriented when its summary has an
// oriented box and axis-aligned otherwise, into a packed RGB8 image of
// `width` x `height`, in the cluster's color. `summaries` are the clusters'
// ClusterSummary rows (clustering.hpp). Edges are clipped at 5 cm in front of
// the camera and at the image border.
cudaError_t draw_cluster_boxes(const float* summaries, std::uint32_t clusters, float fx, float fy,
                               float cx, float cy, int width, int height, std::uint8_t* rgb,
                               cudaStream_t stream);

}  // namespace holoscan::clustering
