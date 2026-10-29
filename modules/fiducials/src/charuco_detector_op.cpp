// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/fiducials/charuco_detector_op.hpp"

#include <chrono>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/fiducials/fiducial_messages.hpp>
#include <holoscan/fiducials/pose.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "frame_detector.hpp"

namespace holoscan::fiducials {
namespace {

// The detector options with the board's dictionary.
DetectorOptions board_options(const CharucoDetectorConfig& config) {
  DetectorOptions options = config.options;
  options.dictionary = config.board.dictionary;
  return options;
}

}  // namespace

void validate_config(const CharucoDetectorConfig& config) {
  (void)CharucoBoard(config.board);
  validate_options(board_options(config));
  if (config.charuco.min_markers < 1 || config.charuco.min_markers > 2) {
    throw std::invalid_argument("charuco.min_markers must be 1 or 2");
  }
  if (config.intrinsics) validate_intrinsics(*config.intrinsics);
}

CharucoDetectorOp::CharucoDetectorOp(CharucoDetectorConfig config)
    : config_((validate_config(config), std::move(config))), board_(config_.board) {
  detector_ =
      std::make_unique<FrameDetector>("CharucoDetectorOp", config_.backend, board_options(config_));
}

CharucoDetectorOp::~CharucoDetectorOp() = default;

Backend CharucoDetectorOp::resolved_backend() const noexcept { return detector_->resolved(); }

void CharucoDetectorOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(image, "image").queue_depth(2U);
  spec.output(detections, "detections");
}

holoscan::Contract CharucoDetectorOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void CharucoDetectorOp::start() { detector_->start(); }

void CharucoDetectorOp::stop() { detector_->stop(); }

holoscan::expected<void, holoscan::Error> CharucoDetectorOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) return holoscan::make_unexpected(std::move(sample).error());
  auto frame = detector_->prepare(sample->data, context);
  if (!frame) return holoscan::make_unexpected(std::move(frame).error());

  FrameMarkers found;
  CharucoObservation view;
  try {
    found = frame->backend->detect(frame->gray, frame->stream);
    view = observe_charuco(
        board_, found.markers,
        [&](std::span<const SaddleSeed> seeds) {
          return frame->backend->refine_saddles(frame->gray, seeds, frame->stream);
        },
        frame->gray.width, frame->gray.height, config_.charuco, config_.intrinsics);
  } catch (const std::exception& exception) {
    return holoscan::make_unexpected(detector_->error(exception, frame->stream));
  }

  schema::FiducialDetectionsT message;
  message.frame = frame->number;
  message.device_sequence = sample->data.header ? sample->data.header->device_sequence : 0U;
  message.image_width = frame->gray.width;
  message.image_height = frame->gray.height;
  message.backend = std::string(frame->backend->name());
  message.dictionary = config_.board.dictionary;
  message.markers.reserve(view.markers.size());
  for (const Marker& marker : view.markers) message.markers.push_back(to_message(marker));
  message.dropped = static_cast<std::int32_t>(found.overflow.markers);
  message.truncated = found.overflow.truncated();
  if (found.overflow.truncated() || found.overflow.markers != 0) {
    detector_->warn_overflow(found.overflow);
  }
  auto board = std::make_unique<schema::CharucoViewT>();
  const CharucoBoardConfig& b = config_.board;
  board->squares_x = b.squares_x;
  board->squares_y = b.squares_y;
  board->square_length = static_cast<float>(b.square_length);
  board->marker_length = static_cast<float>(b.marker_length);
  board->legacy_pattern = b.legacy_pattern;
  board->first_marker_id = b.first_marker_id;
  board->corners.reserve(view.corners.size());
  for (const CharucoCorner& corner : view.corners) {
    board->corners.emplace_back(
        corner.id,
        schema::Point2f(static_cast<float>(corner.image[0]), static_cast<float>(corner.image[1])),
        schema::Point3f(static_cast<float>(corner.board[0]), static_cast<float>(corner.board[1]),
                        static_cast<float>(corner.board[2])));
  }
  board->has_pose = view.pose.has_value();
  if (view.pose) board->pose = std::make_unique<schema::Pose>(to_message(*view.pose));
  board->corners_total = view.quality.corners_total;
  board->markers_found = view.quality.markers_found;
  board->markers_total = view.quality.markers_total;
  board->coverage = static_cast<float>(view.quality.coverage);
  message.board = std::move(board);
  message.detect_time_us =
      std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - frame->start)
          .count();
  return detections.emit(std::move(message), holoscan::sensor_io::derive_from(*sample));
}

}  // namespace holoscan::fiducials
