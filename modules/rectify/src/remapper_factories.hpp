// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Each backend's factory, in its own source file. create_remapper() checks
// their arguments first.

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/rectify/remapper.hpp>

namespace holoscan::rectify {

[[nodiscard]] std::unique_ptr<Remapper> create_cuda_remapper(const RemapModel& model,
                                                             int output_width, int output_height,
                                                             PixelFormat format,
                                                             Interpolation interpolation);
// kVpiCuda, kVpiVic or kVpiPva. Throws std::invalid_argument in a build without VPI.
[[nodiscard]] std::unique_ptr<Remapper> create_vpi_remapper(Backend backend,
                                                            const RemapModel& model,
                                                            int output_width, int output_height,
                                                            PixelFormat format,
                                                            Interpolation interpolation);
// What VPI reports about this platform: whether it has a VIC and a PVA.
// False for both in a build without VPI.
// @throws std::runtime_error if VPI fails to answer.
void vpi_platform(bool& has_vic, bool& has_pva);

}  // namespace holoscan::rectify
