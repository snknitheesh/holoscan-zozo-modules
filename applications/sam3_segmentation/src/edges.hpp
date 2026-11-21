// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// How the demo connects its operators, in either of its two modes:
//
// - Lossless, for synthetic frames and image files: every frame is
//   segmented and reported, in order. A full queue holds back its producer;
//   the sources keep one frame in flight, so the segmenter, which cannot
//   send a refused output again, never meets one (see image_sources.hpp).
// - Real time, for the camera (and synthetic frames with --realtime true):
//   every edge keeps only the newest sample, and a full queue drops its
//   oldest. The segmenter never waits for, and is never refused by, a slower
//   consumer, and when the camera is faster than the segmenter, it takes the
//   newest frame. When the GPU falls behind and every output slot is still
//   held, the segmenter skips the image (drop_when_busy) and counts it.
//
// The display is real time in either mode.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>

#include <holoscan/core/connection_options.hpp>

namespace sam3_app {

// Every sample, in order, at the queue depth its input declares in setup().
inline holoscan::ConnectionOptions lossless() {
  return {.queue_policy = holoscan::QueuePolicy::kBlocking};
}

// Only the newest sample: a consumer still busy finds the newest one waiting
// when it is done, and its producer is never held back.
inline holoscan::ConnectionOptions newest() {
  return {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U};
}

// The longest an image may take from its capture to its instances and still
// be paired with them. SAM 3 with three prompts takes about 410 ms on Jetson
// Orin (the module's README), the slowest platform measured; this leaves
// room.
inline constexpr std::chrono::milliseconds kMaxPairingLatency{600};

// The frames a pairing edge keeps for a source with `period` between frames:
// those captured within kMaxPairingLatency, at least 4 and at most 32. A
// source faster than 32 frames in that time pairs only images whose
// instances come within 32 of its periods; one further behind costs a
// skipped pair, not a stall.
inline std::uint32_t pairing_depth(std::chrono::nanoseconds period) {
  const std::int64_t periods =
      period > std::chrono::nanoseconds::zero() ? kMaxPairingLatency / period + 1 : 32;
  return static_cast<std::uint32_t>(std::clamp<std::int64_t>(periods, 4, 32));
}

// The newest frames, for a consumer that pairs a frame with its instances by
// capture time: the frame must still be queued when they arrive, a
// segmenter's latency later, and with one slot a faster source would already
// have replaced it.
inline holoscan::ConnectionOptions recent_frames(std::chrono::nanoseconds period) {
  return {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = pairing_depth(period)};
}

}  // namespace sam3_app
