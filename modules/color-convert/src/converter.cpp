// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/color_convert/converter.hpp"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "kernels.hpp"
#include "vic.hpp"

namespace holoscan::color_convert {
namespace {

void check_device(const ImageView& view, const char* which) {
  if (view.location != MemoryLocation::kDevice) {
    throw std::invalid_argument(std::string(which) + " image must be in CUDA device memory");
  }
}

}  // namespace

PlatformFacts probe_platform() noexcept {
  PlatformFacts facts{.vpi_compiled = detail::vpi_compiled()};
  try {
    facts.has_vic = detail::vpi_has_vic();
  } catch (const std::exception&) {
    facts.has_vic = false;  // VPI that cannot answer cannot run the VIC either.
  }
  return facts;
}

void check_available(Backend backend) {
  if (backend == Backend::kVic && !detail::vpi_compiled()) {
    throw std::invalid_argument("backend vic: this build has no VPI; install VPI 4 and rebuild");
  }
}

Converter::Converter(Backend backend) : backend_(backend) {
  check_available(backend);
  if (backend != Backend::kCuda) {
    facts_ = probe_platform();
  }
  if (backend == Backend::kVic && !facts_.has_vic) {
    throw std::invalid_argument(
        "backend vic: VPI reports no VIC on this platform; use cuda, or auto to take the VIC "
        "where there is one");
  }
}

Converter::~Converter() = default;

std::pair<Backend, std::string> Converter::choose(const ImageView& source,
                                                  const ImageView& destination,
                                                  const ConversionParams& params) const {
  if (backend_ != Backend::kAuto) {
    return {backend_, std::string()};
  }
  const auto limitation =
      vic_limitation(source.format, source.width, source.height, destination.format,
                     destination.width, destination.height, params);
  const Backend chosen = resolve_auto(facts_, !limitation.has_value());
  std::string reason;
  if (chosen == Backend::kVic) {
    reason = "VPI reports a VIC, which can do this conversion";
  } else if (!facts_.vpi_compiled) {
    reason = "this build has no vic backend";
  } else if (!facts_.has_vic) {
    reason = "VPI reports no VIC on this platform";
  } else {
    reason = *limitation;
  }
  return {chosen, std::move(reason)};
}

Backend Converter::convert(const ImageView& source, const ImageView& destination,
                           const ConversionParams& params, cudaStream_t stream) {
  check_device(source, "source");
  check_device(destination, "destination");
  const Backend chosen = choose(source, destination, params).first;
  if (chosen == Backend::kVic) {
    if (backend_ == Backend::kVic) {
      if (const auto limitation =
              vic_limitation(source.format, source.width, source.height, destination.format,
                             destination.width, destination.height, params)) {
        throw std::invalid_argument("backend vic: " + *limitation);
      }
    }
    validate_conversion(source, destination, params);
    if (!vic_) {
      vic_ = detail::make_vic_engine();
    }
    vic_->convert(source, destination, params, stream);
    return Backend::kVic;
  }
  const detail::Plan plan = detail::make_plan(source, destination, params);
  if (const cudaError_t status = detail::launch_convert(plan, false, stream);
      status != cudaSuccess) {
    throw std::runtime_error(std::string("conversion kernel: ") + cudaGetErrorString(status));
  }
  return Backend::kCuda;
}

}  // namespace holoscan::color_convert
