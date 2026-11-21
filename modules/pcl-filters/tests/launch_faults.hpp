// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A kernel launch made to fail, for the filters' tests: launch_faults.cpp
// defines the CUDA runtime's kernel-launch and cudaGetLastError() functions in
// the test program, which take the place of the runtime's for the module's
// library (the program exports them).

#pragma once

#include <cuda_runtime_api.h>

namespace holoscan::pcl_filters::test {

// Every <<<>>> kernel launch, CUB's radix sorts included (CUB's scans launch
// with cudaLaunchKernelEx and are not counted). While `armed`, launches are
// counted; the one numbered `fail_at` does not run and leaves
// cudaErrorLaunchOutOfResources for cudaGetLastError(), as a failed launch
// does; the launches after it are counted too.
struct LaunchFaults {
  bool armed{};
  int fail_at{};
  int launches{};
  int after_failure{};
  cudaError_t pending{cudaSuccess};
};

LaunchFaults& launch_faults();

}  // namespace holoscan::pcl_filters::test
