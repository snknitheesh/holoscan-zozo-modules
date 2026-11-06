// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "launch_faults.hpp"

#include <dlfcn.h>

#include <cstddef>

namespace holoscan::pcl_filters::test {

LaunchFaults& launch_faults() {
  static LaunchFaults faults;
  return faults;
}

}  // namespace holoscan::pcl_filters::test

// What the compiler's kernel launches call.
extern "C" cudaError_t CUDARTAPI __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block,
                                                    void** args, std::size_t shared,
                                                    cudaStream_t stream) {
  using Real = cudaError_t (*)(cudaKernel_t, dim3, dim3, void**, std::size_t, cudaStream_t);
  static const auto real = reinterpret_cast<Real>(dlsym(RTLD_NEXT, "__cudaLaunchKernel"));
  auto& faults = holoscan::pcl_filters::test::launch_faults();
  if (faults.armed) {
    if (++faults.launches == faults.fail_at) {
      faults.pending = cudaErrorLaunchOutOfResources;
      return faults.pending;
    }
    faults.after_failure += faults.fail_at != 0 && faults.launches > faults.fail_at ? 1 : 0;
  }
  return real(kernel, grid, block, args, shared, stream);
}

extern "C" cudaError_t CUDARTAPI cudaGetLastError() {
  using Real = cudaError_t (*)();
  static const auto real = reinterpret_cast<Real>(dlsym(RTLD_NEXT, "cudaGetLastError"));
  auto& faults = holoscan::pcl_filters::test::launch_faults();
  if (faults.pending != cudaSuccess) {
    const cudaError_t pending = faults.pending;
    faults.pending = cudaSuccess;
    return pending;
  }
  return real();
}
