// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// How the demo connects its operators, in either of its two modes:
//
// - Lossless, for synthetic frames and image files: every frame's depth is
//   estimated and reported, in order. A full queue holds back its producer;
//   the sources keep one frame in flight, so the estimator, which cannot
//   send a refused output again, never meets one (see image_sources.hpp).
// - Real time, for the camera (and synthetic frames with --realtime true):
//   every edge keeps only the newest sample, and a full queue drops its
//   oldest. The estimator never waits for, and is never refused by, a slower
//   consumer, and when the camera is faster than the estimator, it takes the
//   newest frame.
//
// The display is real time in either mode.

#pragma once

#include <holoscan/core/connection_options.hpp>

namespace mono_depth_app {

// Every sample, in order, at the queue depth its input declares in setup().
inline holoscan::ConnectionOptions lossless() {
  return {.queue_policy = holoscan::QueuePolicy::kBlocking};
}

// Only the newest sample: a consumer still busy finds the newest one waiting
// when it is done, and its producer is never held back.
inline holoscan::ConnectionOptions newest() {
  return {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U};
}

// The newest few frames, for a consumer that pairs a frame with its depth by
// capture time: the frame must still be queued when its depth arrives, an
// estimator's latency later, and with one slot a faster camera would already
// have replaced it. Four frames cover an estimator up to three camera periods
// behind; one further behind costs a skipped pair, not a stall.
inline holoscan::ConnectionOptions recent_frames() {
  return {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 4U};
}

}  // namespace mono_depth_app
