// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A remap backend: applies one RemapModel to images in CUDA device memory,
// frame after frame, through a RemapQueue on a CUDA stream. The operators use
// one remapper per camera and one queue for all of them; both also work
// without a Holoscan graph.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
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

// The remaps of one or more remappers on one CUDA stream, made visible to the
// stream's later work by one finish().
//
// cuda needs nothing: its kernels run on the stream itself. vpi-vic and
// vpi-pva share one VPI stream that wraps the CUDA stream: every remapper's
// copy in (CUDA), remap (VIC or PVA) and copy out (CUDA) go to it in order,
// and VPI hands each over between the engines itself. VPI allows no other
// work on the CUDA stream until its own is done (vpiStreamCreateWrapperCUDA)
// and has no fence the GPU could wait on instead, so finish() waits on the
// host, once for everything submitted; and a cuda remap submitted while
// vpi-* work is queued waits for that work first.
//
// Not thread safe: one thread submits and finishes.
class RemapQueue {
 public:
  // Borrows `stream`, which must outlive the queue.
  // @throws std::invalid_argument for the legacy default stream (nullptr).
  explicit RemapQueue(cudaStream_t stream);
  // Waits for the VPI work still queued.
  ~RemapQueue();
  RemapQueue(const RemapQueue&) = delete;
  RemapQueue& operator=(const RemapQueue&) = delete;

  [[nodiscard]] cudaStream_t cuda_stream() const noexcept { return stream_; }

  // Makes every remap submitted since the last finish() visible to work
  // queued on the CUDA stream afterwards. Waits on the host only when vpi-vic
  // or vpi-pva work is queued (vpiStreamSync, which returns once VPI's last
  // task, a copy on the CUDA stream, is done).
  // @return Whether it waited: then all the work queued on the CUDA stream
  //         before is done, too.
  // @throws std::runtime_error on a VPI failure; the queue stays usable.
  bool finish();

  // The VPI stream of a build with VPI, made by the first vpi-* submit.
  struct Vpi;
  // How many times finish() was called, shared with the remappers that
  // submitted to the queue, which outlive it.
  struct Finishes {
    std::uint64_t count{};
  };

 private:
  friend class Remapper;
  struct VpiDelete {
    void operator()(Vpi* vpi) const noexcept;
  };

  // Defined with the vpi-* backends: the VPI stream, made on first use, and
  // the wait for it.
  Vpi& vpi();
  bool finish_vpi();

  cudaStream_t stream_;
  std::shared_ptr<Finishes> finishes_;
  std::unique_ptr<Vpi, VpiDelete> vpi_;
};

class Remapper {
 public:
  virtual ~Remapper() = default;
  Remapper(const Remapper&) = delete;
  Remapper& operator=(const Remapper&) = delete;

  // "cuda", "vpi-vic" or "vpi-pva".
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] Backend backend() const noexcept { return backend_; }

  // Queues the remap of `input` (the input camera's size) into `output` (the
  // output size), both in the remapper's format, on `queue`, after the work
  // already queued on its CUDA stream. Output pixels that sample outside the
  // input are zero. Call finish() on the queue before `output` is read,
  // `input` is reused, this remapper submits again (to any queue), or it is
  // destroyed.
  // @throws std::invalid_argument for images of another size, or whose rows
  //         are shorter than `width` pixels, or not aligned to the pixels'
  //         element size (2 bytes for MONO16).
  // @throws std::logic_error when this remapper's last submit is not
  //         finished: the queue it went to, if it still exists, has not
  //         called finish() since.
  // @throws std::runtime_error on a CUDA or VPI failure; call queue.finish()
  //         before anything else.
  void submit(const DeviceImage& input, const DeviceImage& output, RemapQueue& queue);

 protected:
  Remapper(Backend backend, PixelFormat format, int input_width, int input_height, int output_width,
           int output_height)
      : backend_(backend),
        format_(format),
        input_width_(input_width),
        input_height_(input_height),
        output_width_(output_width),
        output_height_(output_height) {}
  virtual void do_submit(const DeviceImage& input, const DeviceImage& output,
                         RemapQueue& queue) = 0;
  // For the vpi-* backends.
  [[nodiscard]] static RemapQueue::Vpi& vpi_of(RemapQueue& queue) { return queue.vpi(); }
  // For the cuda backend: waits for the vpi-* work queued on `queue`, if any,
  // after which CUDA work may go to its stream.
  static void finish_vpi_of(RemapQueue& queue) {
    if (queue.vpi_) (void)queue.finish_vpi();
  }

  Backend backend_;
  PixelFormat format_;
  int input_width_;
  int input_height_;
  int output_width_;
  int output_height_;

 private:
  // The finish() count of the queue of the last submit, and its value then.
  std::weak_ptr<const RemapQueue::Finishes> submitted_to_;
  std::uint64_t submitted_at_finish_{};
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
// @throws std::runtime_error on a CUDA or VPI failure, which is not a refusal.
[[nodiscard]] std::optional<std::string> try_backend(Backend backend, PixelFormat format,
                                                     Interpolation interpolation, int width,
                                                     int height);

// What the remappers and operators of this process have cost the calling
// threads so far, for tests and profiling.
struct RemapStatistics {
  // Waits of a thread for the GPU, the VIC or the PVA: RemapQueue::finish()
  // with vpi-* work, a host producer's readiness, a copy from pageable host
  // memory (CUDA stages it before returning), and the waits of construction
  // and of failures.
  std::uint64_t host_waits{};
  // Input images copied from host to device memory.
  std::uint64_t host_copies{};
};
[[nodiscard]] RemapStatistics remap_statistics() noexcept;

}  // namespace holoscan::rectify
