// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Backend::kAuto's choice for every platform fact, and the backend names.

#include <holoscan/fiducials/types.hpp>

#include "test_support.hpp"

namespace holoscan::fiducials {
namespace {

void check_resolution() {
  static_assert(resolve_auto(PlatformFacts{.cuda_device = true}) == Backend::kCuda);
  static_assert(resolve_auto(PlatformFacts{.cuda_device = false}) == Backend::kCpu);
  VERIFY(resolve_auto(PlatformFacts{.cuda_device = true}) == Backend::kCuda);
  VERIFY(resolve_auto(PlatformFacts{}) == Backend::kCpu);
}

void check_names() {
  for (const Backend backend : {Backend::kCpu, Backend::kCuda, Backend::kAuto}) {
    VERIFY(parse_backend(backend_name(backend)) == backend);
  }
  VERIFY(!parse_backend("vpi") && !parse_backend("CUDA") && !parse_backend(""));
  const auto backends = available_backends();
  VERIFY(backends.size() == 2 && backends[0] == Backend::kCpu && backends[1] == Backend::kCuda);
}

}  // namespace
}  // namespace holoscan::fiducials

int main() {
  return holoscan::fiducials::test::run({
      {"auto backend resolution", holoscan::fiducials::check_resolution},
      {"backend names", holoscan::fiducials::check_names},
  });
}
