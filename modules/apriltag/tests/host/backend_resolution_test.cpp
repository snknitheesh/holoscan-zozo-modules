// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Backend::kAuto's choice for every combination of platform facts. Needs no SDK
// or GPU, so it runs on hosted CI.

#include <holoscan/apriltag/detection.hpp>

#include "test_support.hpp"

namespace holoscan::apriltag {
namespace {

// The PVA wins only when the vpi backend is compiled in and VPI reports one.
void check_vpi_with_pva() {
  static_assert(resolve_auto({.vpi_compiled = true, .has_pva = true, .cuda_device = true}) ==
                Backend::kVpi);
  VERIFY(resolve_auto({.vpi_compiled = true, .has_pva = true, .cuda_device = true}) ==
         Backend::kVpi);
  VERIFY(resolve_auto({.vpi_compiled = true, .has_pva = true, .cuda_device = false}) ==
         Backend::kVpi);
}

// Without a PVA, or without the vpi backend, a usable GPU gets cuda: x86_64 hosts,
// which have VPI but no PVA, and Jetson builds without VPI.
void check_cuda_without_pva() {
  VERIFY(resolve_auto({.vpi_compiled = true, .has_pva = false, .cuda_device = true}) ==
         Backend::kCuda);
  VERIFY(resolve_auto({.vpi_compiled = false, .has_pva = false, .cuda_device = true}) ==
         Backend::kCuda);
  VERIFY(resolve_auto({.vpi_compiled = false, .has_pva = true, .cuda_device = true}) ==
         Backend::kCuda);
}

// Neither a PVA nor a GPU: the CPU.
void check_cpu_otherwise() {
  VERIFY(resolve_auto({.vpi_compiled = true, .has_pva = false, .cuda_device = false}) ==
         Backend::kCpu);
  VERIFY(resolve_auto({.vpi_compiled = false, .has_pva = false, .cuda_device = false}) ==
         Backend::kCpu);
  VERIFY(resolve_auto({.vpi_compiled = false, .has_pva = true, .cuda_device = false}) ==
         Backend::kCpu);
  VERIFY(resolve_auto({}) == Backend::kCpu);
}

}  // namespace
}  // namespace holoscan::apriltag

int main() {
  return holoscan::apriltag::test::run({
      {"auto: vpi with a PVA", holoscan::apriltag::check_vpi_with_pva},
      {"auto: cuda without a PVA", holoscan::apriltag::check_cuda_without_pva},
      {"auto: cpu without a PVA or a GPU", holoscan::apriltag::check_cpu_otherwise},
  });
}
