// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Each backend's factory, in its own source file, and the counters behind
// remap_statistics(). create_remapper() checks the factories' arguments first.

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/rectify/remapper.hpp>

namespace holoscan::rectify {

[[nodiscard]] std::unique_ptr<Remapper> create_cuda_remapper(const RemapModel& model,
                                                             int output_width, int output_height,
                                                             PixelFormat format,
                                                             Interpolation interpolation);
// kVpiVic or kVpiPva. Throws std::invalid_argument in a build without VPI.
[[nodiscard]] std::unique_ptr<Remapper> create_vpi_remapper(Backend backend,
                                                            const RemapModel& model,
                                                            int output_width, int output_height,
                                                            PixelFormat format,
                                                            Interpolation interpolation);
// What VPI reports about this platform: whether it has a VIC and a PVA.
// False for both in a build without VPI.
// @throws std::runtime_error if VPI fails to answer.
void vpi_platform(bool& has_vic, bool& has_pva);

// Called just before a thread waits for the GPU, the VIC or the PVA, and when
// a host image is copied to device memory: see RemapStatistics.
void count_host_wait() noexcept;
void count_host_copy() noexcept;

}  // namespace holoscan::rectify
