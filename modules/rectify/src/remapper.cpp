// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/rectify/remapper.hpp"

#include <algorithm>
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

}  // namespace

std::vector<Backend> available_backends() {
  std::vector<Backend> backends{Backend::kCuda};
#if HOLOSCAN_RECTIFY_WITH_VPI
  backends.insert(backends.end(), {Backend::kVpiCuda, Backend::kVpiVic, Backend::kVpiPva});
#endif
  return backends;
}

void check_available(Backend backend) {
  if (backend_name(backend) == "unknown") throw std::invalid_argument("unknown backend");
  if (!HOLOSCAN_RECTIFY_WITH_VPI && (backend == Backend::kVpiCuda || backend == Backend::kVpiVic ||
                                     backend == Backend::kVpiPva)) {
    throw std::invalid_argument("backend " + std::string(backend_name(backend)) + ": " +
                                std::string(kNoVpi));
  }
}

#if !HOLOSCAN_RECTIFY_WITH_VPI
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

void Remapper::submit(const DeviceImage& input, const DeviceImage& output, cudaStream_t stream) {
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
  do_submit(input, output, stream);
}

std::unique_ptr<Remapper> create_remapper(Backend backend, const RemapModel& model,
                                          int output_width, int output_height, PixelFormat format,
                                          Interpolation interpolation, std::string* auto_reason) {
  check_available(backend);
  const CameraModel& input = model.input();
  if (output_width < 1 || output_height < 1 || output_width > kMaxImageSide ||
      output_height > kMaxImageSide) {
    throw std::invalid_argument("the output size must be 1.." + std::to_string(kMaxImageSide));
  }
  const PlatformFacts facts = probe_platform();
  // A backend must take both the input and the output size.
  const auto limitation = [&](Backend candidate) -> std::optional<std::string> {
    if (auto why = backend_limitation(candidate, facts, format, interpolation, input.width,
                                      input.height)) {
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
    CameraModel camera;
    camera.width = width;
    camera.height = height;
    camera.fx = camera.fy = std::max(width, height);
    camera.cx = (width - 1) / 2.0;
    camera.cy = (height - 1) / 2.0;
    const RemapModel identity(camera, Matrix3{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                              Intrinsics{camera.fx, camera.fy, camera.cx, camera.cy});
    (void)create_remapper(backend, identity, width, height, format, interpolation);
  } catch (const std::invalid_argument& error) {
    return std::string(error.what());
  }
  return std::nullopt;
}

}  // namespace holoscan::rectify
