// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A remap backend: applies one RemapModel to images in CUDA device memory,
// frame after frame. The operators use one per camera; it also works without
// a Holoscan graph.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <holoscan/rectify/backend.hpp>
#include <holoscan/rectify/rectification.hpp>

namespace holoscan::rectify {

// A borrowed image in CUDA device memory: `pitch` bytes from one row to the next.
struct DeviceImage {
  void* data{};
  std::size_t pitch{};
  int width{};
  int height{};
};

class Remapper {
 public:
  virtual ~Remapper() = default;
  Remapper(const Remapper&) = delete;
  Remapper& operator=(const Remapper&) = delete;

  // "cuda", "vpi-cuda", "vpi-vic" or "vpi-pva".
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] Backend backend() const noexcept { return backend_; }

  // Starts remapping `input` (the input camera's size) into `output` (the
  // output size), both in `format`, after the work already queued on `stream`.
  // Output pixels that sample outside the input are zero. Call finish() before
  // `output` is read, `input` is reused, or the next submit().
  // @throws std::invalid_argument for images of another size.
  // @throws std::runtime_error on a CUDA or VPI failure.
  void submit(const DeviceImage& input, const DeviceImage& output, cudaStream_t stream);
  // Makes the submitted remap visible to work queued on `stream` afterwards:
  // for cuda nothing (it already runs on the stream); for VPI a wait on the
  // host, and for vpi-vic and vpi-pva the copy of the result onto `stream`.
  // @throws std::runtime_error on a VPI failure.
  virtual void finish() = 0;

 protected:
  Remapper(Backend backend, int input_width, int input_height, int output_width, int output_height)
      : backend_(backend),
        input_width_(input_width),
        input_height_(input_height),
        output_width_(output_width),
        output_height_(output_height) {}
  virtual void do_submit(const DeviceImage& input, const DeviceImage& output,
                         cudaStream_t stream) = 0;

  Backend backend_;
  int input_width_;
  int input_height_;
  int output_width_;
  int output_height_;
};

// Builds a remapper that applies `model` to `format` images: input of the
// model's input size, output `output_width` x `output_height`. For kAuto,
// tries resolve_auto()'s choice and falls back to cuda if the VIC refuses;
// `auto_reason`, when given, receives why it chose what it did.
// @throws std::invalid_argument when the backend cannot take these images
//         (backend_limitation(), or VPI refusing them).
// @throws std::runtime_error on a CUDA or VPI failure.
[[nodiscard]] std::unique_ptr<Remapper> create_remapper(Backend backend, const RemapModel& model,
                                                        int output_width, int output_height,
                                                        PixelFormat format,
                                                        Interpolation interpolation,
                                                        std::string* auto_reason = nullptr);

// Why `backend` cannot remap `width` x `height` `format` images with
// `interpolation` on this machine: backend_limitation(), or VPI refusing them
// when the backend is built (as Thor's VIC refuses nearest interpolation);
// nullopt when it can. Builds an identity remapper of that size and drops it.
[[nodiscard]] std::optional<std::string> try_backend(Backend backend, PixelFormat format,
                                                     Interpolation interpolation, int width,
                                                     int height);

}  // namespace holoscan::rectify
