// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The few NVIDIA V4L2 extensions the Jetson V4L2 backend uses, beyond <linux/videodev2.h>. The
// values and the initial QP layout are those of v4l2_nv_extensions.h in NVIDIA's Jetson Multimedia
// API (BSD-3-Clause; see the module's NOTICE), which the CI image does not have.

#pragma once

#include <cstdint>

#include <linux/videodev2.h>  // NOLINT(build/include_order)

namespace holoscan::codec::v4l2_nv {

// The NVIDIA encoder's H.265 stream format ('H265'; Linux's own is 'HEVC').
inline constexpr std::uint32_t kPixFmtH265 = v4l2_fourcc('H', '2', '6', '5');

// Encoder controls, set with VIDIOC_S_EXT_CTRLS in the MPEG class.
inline constexpr std::uint32_t kCidH265Profile = V4L2_CID_MPEG_BASE + 513;
inline constexpr std::uint32_t kCidIdrInterval = V4L2_CID_MPEG_BASE + 514;
inline constexpr std::uint32_t kCidNumReferenceFrames = V4L2_CID_MPEG_BASE + 532;
inline constexpr std::uint32_t kCidNumBFrames = V4L2_CID_MPEG_BASE + 534;
inline constexpr std::uint32_t kCidInsertSpsPpsAtIdr = V4L2_CID_MPEG_BASE + 535;
inline constexpr std::uint32_t kCidHwPresetType = V4L2_CID_MPEG_BASE + 540;
inline constexpr std::uint32_t kCidMaxPerformance = V4L2_CID_MPEG_BASE + 554;
inline constexpr std::uint32_t kCidInitFrameQp = V4L2_CID_MPEG_BASE + 574;

// Decoder controls: input buffers that hold parts of frames (0: whole frames), and showing each
// frame as soon as it is decoded, for streams with one reference frame.
inline constexpr std::uint32_t kCidDisableCompleteFrameInput = V4L2_CID_MPEG_BASE + 515;
inline constexpr std::uint32_t kCidDisableDpb = V4L2_CID_MPEG_BASE + 516;

// The decoder's resolution-change event, after which the capture plane is set up.
inline constexpr std::uint32_t kEventResolutionChange = 5;

inline constexpr std::int32_t kH265ProfileMain = 0;

// Values of kCidHwPresetType.
inline constexpr std::int32_t kHwPresetUltraFast = 1;
inline constexpr std::int32_t kHwPresetMedium = 3;
inline constexpr std::int32_t kHwPresetSlow = 4;

// The value of kCidInitFrameQp, passed by pointer in v4l2_ext_control::string: with rate control
// off, the QP of every frame.
struct InitFrameQp {
  std::uint32_t i_qp;
  std::uint32_t p_qp;
  std::uint32_t b_qp;
  std::uint32_t reserved[4];
};

}  // namespace holoscan::codec::v4l2_nv
