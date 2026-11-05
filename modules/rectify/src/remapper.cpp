// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/rectify/remapper.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "remapper_factories.hpp"

namespace holoscan::rectify {
namespace {

constexpr std::string_view kNoVpi = "this build has no VPI; install VPI 4 and rebuild";

std::atomic<std::uint64_t> g_host_waits{0};
std::atomic<std::uint64_t> g_host_copies{0};

}  // namespace

void count_host_wait() noexcept { g_host_waits.fetch_add(1, std::memory_order_relaxed); }

void count_host_copy() noexcept { g_host_copies.fetch_add(1, std::memory_order_relaxed); }

RemapStatistics remap_statistics() noexcept {
  return RemapStatistics{.host_waits = g_host_waits.load(std::memory_order_relaxed),
                         .host_copies = g_host_copies.load(std::memory_order_relaxed)};
}

std::vector<Backend> available_backends() {
  std::vector<Backend> backends{Backend::kCuda};
#if HOLOSCAN_RECTIFY_WITH_VPI
  backends.insert(backends.end(), {Backend::kVpiVic, Backend::kVpiPva});
#endif
  return backends;
}

void check_available(Backend backend) {
  if (backend_name(backend) == "unknown") throw std::invalid_argument("unknown backend");
  if (!HOLOSCAN_RECTIFY_WITH_VPI && (backend == Backend::kVpiVic || backend == Backend::kVpiPva)) {
    throw std::invalid_argument("backend " + std::string(backend_name(backend)) + ": " +
                                std::string(kNoVpi));
  }
}

RemapQueue::RemapQueue(cudaStream_t stream)
    : stream_(stream), finishes_(std::make_shared<Finishes>()) {
  if (stream == nullptr) {
    throw std::invalid_argument("a remap queue needs a CUDA stream, not the legacy default one");
  }
}

// Destroying the VPI stream waits for its work.
RemapQueue::~RemapQueue() = default;

bool RemapQueue::finish() {
  ++finishes_->count;
  return vpi_ && finish_vpi();
}

#if !HOLOSCAN_RECTIFY_WITH_VPI
struct RemapQueue::Vpi {};

void RemapQueue::VpiDelete::operator()(Vpi* vpi) const noexcept { delete vpi; }

bool RemapQueue::finish_vpi() { return false; }

void vpi_platform(bool& has_vic, bool& has_pva) {
  has_vic = false;
  has_pva = false;
}

std::unique_ptr<Remapper> create_vpi_remapper(Backend backend, const RemapModel&, int, int,
                                              PixelFormat, Interpolation) {
  throw std::invalid_argument("backend " + std::string(backend_name(backend)) + ": " +
                              std::string(kNoVpi));
}
#endif

PlatformFacts probe_platform() noexcept {
  PlatformFacts facts{.vpi_compiled = HOLOSCAN_RECTIFY_WITH_VPI != 0};
  try {
    vpi_platform(facts.has_vic, facts.has_pva);
  } catch (const std::exception&) {
    facts.has_vic = false;  // VPI that cannot answer cannot remap either.
    facts.has_pva = false;
  }
  int devices = 0;
  facts.cuda_device = cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
  return facts;
}

void Remapper::submit(const DeviceImage& input, const DeviceImage& output, RemapQueue& queue) {
  if (input.width != input_width_ || input.height != input_height_ ||
      output.width != output_width_ || output.height != output_height_) {
    throw std::invalid_argument(
        std::string(name()) + " remap: images of " + std::to_string(input.width) + "x" +
        std::to_string(input.height) + " into " + std::to_string(output.width) + "x" +
        std::to_string(output.height) + " do not match the map's " + std::to_string(input_width_) +
        "x" + std::to_string(input_height_) + " into " + std::to_string(output_width_) + "x" +
        std::to_string(output_height_));
  }
  if (input.data == nullptr || output.data == nullptr) {
    throw std::invalid_argument(std::string(name()) + " remap: an image has no data");
  }
  // Every backend reads and writes whole pixels of `element` bytes each.
  const std::size_t element =
      static_cast<std::size_t>(bytes_per_pixel(format_) / channels(format_));
  for (const DeviceImage* image : {&input, &output}) {
    const std::size_t row = static_cast<std::size_t>(image->width) * bytes_per_pixel(format_);
    if (image->pitch < row || image->pitch % element != 0 ||
        reinterpret_cast<std::uintptr_t>(image->data) % element != 0) {
      throw std::invalid_argument(std::string(name()) + " remap: an image's rows are " +
                                  std::to_string(image->pitch) + " bytes apart; they must hold " +
                                  std::to_string(row) + " bytes, and rows and data be " +
                                  "aligned to " + std::to_string(element) + " bytes for " +
                                  std::string(pixel_format_name(format_)));
    }
  }
  if (const auto last = submitted_to_.lock(); last && last->count == submitted_at_finish_) {
    throw std::logic_error(std::string(name()) +
                           " remap: submitted again before the queue of its last submit finished");
  }
  // Pending from here on, even if do_submit() fails halfway.
  submitted_to_ = queue.finishes_;
  submitted_at_finish_ = queue.finishes_->count;
  do_submit(input, output, queue);
}

std::unique_ptr<Remapper> create_remapper(Backend backend, const RemapModel& model,
                                          int output_width, int output_height, PixelFormat format,
                                          Interpolation interpolation, std::string* auto_reason) {
  check_available(backend);
  const messages::CameraInfo& input = model.input();
  const int input_width = static_cast<int>(input.width);
  const int input_height = static_cast<int>(input.height);
  if (output_width < 1 || output_height < 1 || output_width > kMaxImageSide ||
      output_height > kMaxImageSide) {
    throw std::invalid_argument("the output size must be 1.." + std::to_string(kMaxImageSide));
  }
  const PlatformFacts facts = probe_platform();
  // A backend must take both the input and the output size.
  const auto limitation = [&](Backend candidate) -> std::optional<std::string> {
    if (auto why = backend_limitation(candidate, facts, format, interpolation, input_width,
                                      input_height)) {
      return why;
    }
    return backend_limitation(candidate, facts, format, interpolation, output_width, output_height);
  };
  const auto build = [&](Backend chosen) -> std::unique_ptr<Remapper> {
    if (chosen == Backend::kCuda) {
      return create_cuda_remapper(model, output_width, output_height, format, interpolation);
    }
    return create_vpi_remapper(chosen, model, output_width, output_height, format, interpolation);
  };
  if (backend != Backend::kAuto) {
    if (auto why = limitation(backend)) throw std::invalid_argument(*why);
    return build(backend);
  }
  // Auto: the VIC when it takes the images, else this module's kernel.
  std::string reason;
  if (auto why = limitation(Backend::kVpiVic)) {
    reason = *why;
  } else {
    try {
      auto remapper = build(Backend::kVpiVic);
      if (auto_reason != nullptr) *auto_reason = "the VIC takes these images";
      return remapper;
    } catch (const std::invalid_argument& error) {
      reason = error.what();  // Refused by VPI, such as Thor's VIC with nearest.
    }
  }
  if (auto why = limitation(Backend::kCuda)) throw std::invalid_argument(*why);
  if (auto_reason != nullptr) *auto_reason = "not the VIC: " + reason;
  return build(Backend::kCuda);
}

std::optional<std::string> try_backend(Backend backend, PixelFormat format,
                                       Interpolation interpolation, int width, int height) {
  if (backend == Backend::kAuto) return std::nullopt;
  try {
    check_available(backend);
    if (auto why =
            backend_limitation(backend, probe_platform(), format, interpolation, width, height)) {
      return why;
    }
    messages::CameraInfo camera;
    camera.width = static_cast<std::uint32_t>(width);
    camera.height = static_cast<std::uint32_t>(height);
    camera.state = messages::CameraState::kUndistorted;
    camera.fx = camera.fy = std::max(width, height);
    camera.cx = (width - 1) / 2.0;
    camera.cy = (height - 1) / 2.0;
    const RemapModel identity(camera, messages::kIdentity3,
                              Intrinsics{camera.fx, camera.fy, camera.cx, camera.cy});
    (void)create_remapper(backend, identity, width, height, format, interpolation);
  } catch (const std::invalid_argument& error) {
    // Refused: the images, or the backend on this machine. A CUDA or VPI
    // failure (std::runtime_error) is not a refusal and goes to the caller.
    return std::string(error.what());
  }
  return std::nullopt;
}

}  // namespace holoscan::rectify
