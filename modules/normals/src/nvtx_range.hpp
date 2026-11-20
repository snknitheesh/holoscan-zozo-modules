// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// An NVTX range around a phase of the kernels' host code, so that a profiler
// (Nsight Systems) shows which phase launched which kernels, inside the SDK's
// per-operator ranges. NVTX costs nothing when no profiler is attached.

#pragma once

#include <nvtx3/nvToolsExt.h>

namespace holoscan::normals::detail {

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) noexcept { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace holoscan::normals::detail
