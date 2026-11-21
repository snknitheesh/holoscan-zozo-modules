// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The v4l2 backend of a build without it (no VPI 4 headers): never available.

#include <memory>
#include <stdexcept>

#include "backend.hpp"

namespace holoscan::codec::detail {
namespace {

constexpr const char* kNotBuilt = "this build has no v4l2 backend (it needs VPI 4's headers)";

}  // namespace

Probe probe_v4l2_encoder(Codec) { return {.available = false, .detail = kNotBuilt}; }
Probe probe_v4l2_decoder(Codec) { return {.available = false, .detail = kNotBuilt}; }

std::unique_ptr<EncoderBackend> create_v4l2_encoder(const EncoderSettings&, int, int) {
  throw std::runtime_error(kNotBuilt);
}

std::unique_ptr<DecoderBackend> create_v4l2_decoder(Codec, bool) {
  throw std::runtime_error(kNotBuilt);
}

}  // namespace holoscan::codec::detail
