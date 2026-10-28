// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// How the demo connects its operators, in either of its two modes:
//
// - Lossless, for synthetic frames and image files: every frame is detected
//   and reported, in order. A full queue holds back its producer; the sources
//   keep one frame in flight, so the detector, which cannot send a refused
//   output again, never meets one (see image_sources.hpp).
// - Real time, for the camera: every edge keeps only the newest sample, and a
//   full queue drops its oldest. The detector never waits for, and is never
//   refused by, a slower consumer, and when the camera is faster than the
//   detector, it takes the newest frame.

#pragma once

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/yolo26/yolo26_detection_op.hpp>

#include "detection_report_op.hpp"

namespace yolo26_app {

// Every sample, in order, at the queue depth its input declares in setup().
inline holoscan::ConnectionOptions lossless() {
  return {.queue_policy = holoscan::QueuePolicy::kBlocking};
}

// Only the newest sample: a consumer still busy finds the newest one waiting
// when it is done, and its producer is never held back.
inline holoscan::ConnectionOptions newest() {
  return {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U};
}

// The newest few frames, for a consumer that pairs a frame with its detections
// by capture time: the frame must still be queued when its detections arrive,
// a detector's latency later, and with one slot a faster camera would already
// have replaced it. Four frames cover a detector up to three camera periods
// behind; one further behind costs a skipped pair, not a stall.
inline holoscan::ConnectionOptions recent_frames() {
  return {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 4U};
}

// The detector's four outputs to the reporter, all with `options`.
inline void connect_detections(
    holoscan::Graph& graph,
    const holoscan::OperatorHandle<holoscan::yolo26::Yolo26DetectionOp>& detector,
    const holoscan::OperatorHandle<DetectionReportOp>& reporter,
    const holoscan::ConnectionOptions& options) {
  graph.add_flow(detector->boxes, reporter->boxes, options);
  graph.add_flow(detector->scores, reporter->scores, options);
  graph.add_flow(detector->class_ids, reporter->class_ids, options);
  graph.add_flow(detector->count, reporter->count, options);
}

}  // namespace yolo26_app
