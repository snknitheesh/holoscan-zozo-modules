// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include <holoscan/sam3/kernels.hpp>
#include <holoscan/sam3/pixel_math.hpp>

namespace holoscan::sam3 {
namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
// The ranking kernels run as one block; every thread ranks a strided share of
// the candidates against all of them, which for a few hundred is
// microseconds.
constexpr int kRankThreads = 256;
constexpr int kCopyThreads = 256;

dim3 grid_for(int width, int height) {
  return dim3((static_cast<unsigned>(width) + kBlockX - 1) / kBlockX,
              (static_cast<unsigned>(height) + kBlockY - 1) / kBlockY);
}

// One model input pixel per thread, written to all three planes.
__global__ void preprocess_kernel(PreprocessParams p, const std::uint8_t* image, float* input) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= p.input_width || v >= p.input_height) {
    return;
  }
  float rgb[3];
  preprocess_pixel(p, image, u, v, rgb);
  const std::size_t plane =
      static_cast<std::size_t>(p.input_width) * static_cast<std::size_t>(p.input_height);
  const std::size_t index =
      static_cast<std::size_t>(v) * static_cast<std::size_t>(p.input_width) + u;
  input[index] = rgb[0];
  input[plane + index] = rgb[1];
  input[2 * plane + index] = rgb[2];
}

__device__ bool passes(float score, float threshold) { return score > threshold; }

// Ranks the queries that pass among each other and writes the best
// `capacity` to the prompt's slots.
__global__ void select_kernel(SelectParams p, const float* scores, const float* boxes,
                              Candidates c) {
  __shared__ int passed;
  if (threadIdx.x == 0) {
    passed = 0;
  }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.x); i < p.queries; i += blockDim.x) {
    const float score = scores[i];
    if (!passes(score, p.score_threshold)) {
      continue;
    }
    atomicAdd(&passed, 1);
    int rank = 0;
    for (int j = 0; j < p.queries; ++j) {
      const float other = scores[j];
      if (passes(other, p.score_threshold) && ranks_before(other, j, score, i)) {
        ++rank;
      }
    }
    if (rank < p.capacity) {
      const std::size_t slot = static_cast<std::size_t>(p.prompt) * p.capacity + rank;
      c.scores[slot] = score;
      for (int k = 0; k < 4; ++k) {
        c.boxes[4 * slot + k] = boxes[4 * static_cast<std::size_t>(i) + k];
      }
      c.queries[slot] = i;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    c.counts[p.prompt] = passed < p.capacity ? passed : p.capacity;
    if (c.passed != nullptr) {
      c.passed[p.prompt] = passed;
    }
  }
}

// Copies the mask logits of slot blockIdx.y, if it holds an instance.
__global__ void copy_logits_kernel(SelectParams p, const float* masks, Candidates c) {
  const int j = static_cast<int>(blockIdx.y);
  if (j >= c.counts[p.prompt]) {
    return;
  }
  const std::size_t cells =
      static_cast<std::size_t>(p.mask_width) * static_cast<std::size_t>(p.mask_height);
  const std::size_t slot = static_cast<std::size_t>(p.prompt) * p.capacity + j;
  const float* from = masks + static_cast<std::size_t>(c.queries[slot]) * cells;
  float* to = c.logits + slot * cells;
  for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < cells; cell += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    to[cell] = from[cell];
  }
}

__device__ bool holds(const Candidates& c, int capacity, int slot) {
  return slot % capacity < c.counts[slot / capacity];
}

// Ranks every prompt's candidates together and writes the best `capacity`.
__global__ void rank_kernel(FinalizeParams p, Candidates c, Instances out) {
  __shared__ int held;
  if (threadIdx.x == 0) {
    held = 0;
  }
  __syncthreads();
  const int slots = p.prompts * p.capacity;
  for (int i = static_cast<int>(threadIdx.x); i < slots; i += blockDim.x) {
    if (!holds(c, p.capacity, i)) {
      continue;
    }
    atomicAdd(&held, 1);
    const float score = c.scores[i];
    int rank = 0;
    for (int j = 0; j < slots; ++j) {
      if (holds(c, p.capacity, j) && ranks_before(c.scores[j], j, score, i)) {
        ++rank;
      }
    }
    if (rank < p.capacity) {
      out.order[rank] = i;
      out.scores[rank] = score;
      box_to_pixels(c.boxes + 4 * static_cast<std::size_t>(i), p.width, p.height,
                    out.boxes + 4 * static_cast<std::size_t>(rank));
      out.prompt_ids[rank] = i / p.capacity;
    }
  }
  __syncthreads();
  const int count = held < p.capacity ? held : p.capacity;
  for (int k = count + static_cast<int>(threadIdx.x); k < p.capacity; k += blockDim.x) {
    out.order[k] = -1;
    out.scores[k] = 0.0F;
    for (int e = 0; e < 4; ++e) {
      out.boxes[4 * static_cast<std::size_t>(k) + e] = 0.0F;
    }
    out.prompt_ids[k] = -1;
  }
  if (threadIdx.x == 0) {
    out.count[0] = count;
  }
}

// One image pixel per thread: every instance's mask there, and its label.
__global__ void masks_kernel(FinalizeParams p, Candidates c, Instances out) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) {
    return;
  }
  const MaskSampling s = make_mask_sampling(p.mask_width, p.mask_height, p.width, p.height);
  const std::size_t cells =
      static_cast<std::size_t>(p.mask_width) * static_cast<std::size_t>(p.mask_height);
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + x;
  const int count = out.count[0];
  std::uint16_t label = 0;
  for (int k = 0; k < p.capacity; ++k) {
    bool on = false;
    if (k < count) {
      const float* logits = c.logits + static_cast<std::size_t>(out.order[k]) * cells;
      on = sample_mask(s, logits, x, y) > p.mask_logit_threshold;
      if (on && label == 0) {
        label = static_cast<std::uint16_t>(k + 1);
      }
    }
    if (out.masks != nullptr) {
      out.masks[static_cast<std::size_t>(k) * plane + index] = on ? 1 : 0;
    }
  }
  out.labels[index] = label;
}

__global__ void overlay_kernel(const std::uint16_t* labels, int width, int height,
                               std::uint8_t fill_alpha, std::uint8_t* rgba) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }
  overlay_pixel(labels, width, height, x, y, fill_alpha,
                rgba + 4 * (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + x));
}

}  // namespace

cudaError_t launch_preprocess(const ImageView& image, int input_width, int input_height,
                              const Normalization& normalization, float* input,
                              cudaStream_t stream) {
  const PreprocessParams p =
      make_preprocess_params(image.width, image.height, image.format, image.pitch, input_width,
                             input_height, normalization);
  preprocess_kernel<<<grid_for(input_width, input_height), dim3(kBlockX, kBlockY), 0, stream>>>(
      p, image.data, input);
  return cudaGetLastError();
}

cudaError_t launch_select(const float* scores, const float* boxes, const float* masks,
                          const SelectParams& params, const Candidates& candidates,
                          cudaStream_t stream) {
  select_kernel<<<1, kRankThreads, 0, stream>>>(params, scores, boxes, candidates);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  const std::size_t cells =
      static_cast<std::size_t>(params.mask_width) * static_cast<std::size_t>(params.mask_height);
  const auto blocks = static_cast<unsigned>((cells + kCopyThreads - 1) / kCopyThreads);
  copy_logits_kernel<<<dim3(blocks, static_cast<unsigned>(params.capacity)), kCopyThreads, 0,
                       stream>>>(params, masks, candidates);
  return cudaGetLastError();
}

cudaError_t launch_finalize(const Candidates& candidates, const FinalizeParams& params,
                            const Instances& instances, cudaStream_t stream) {
  rank_kernel<<<1, kRankThreads, 0, stream>>>(params, candidates, instances);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  masks_kernel<<<grid_for(params.width, params.height), dim3(kBlockX, kBlockY), 0, stream>>>(
      params, candidates, instances);
  return cudaGetLastError();
}

cudaError_t launch_overlay(const std::uint16_t* labels, int width, int height,
                           std::uint8_t fill_alpha, std::uint8_t* rgba, cudaStream_t stream) {
  overlay_kernel<<<grid_for(width, height), dim3(kBlockX, kBlockY), 0, stream>>>(
      labels, width, height, fill_alpha, rgba);
  return cudaGetLastError();
}

}  // namespace holoscan::sam3
