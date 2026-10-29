// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/fiducials/fiducial_messages.hpp"

#include <array>

namespace holoscan::fiducials {

schema::Pose to_message(const Pose& pose) {
  std::array<float, 9> rotation{};
  std::array<float, 3> translation{};
  for (int i = 0; i < 9; ++i) rotation[i] = static_cast<float>(pose.rotation[i]);
  for (int i = 0; i < 3; ++i) translation[i] = static_cast<float>(pose.translation[i]);
  return schema::Pose(::flatbuffers::span<const float, 9>(rotation),
                      ::flatbuffers::span<const float, 3>(translation),
                      static_cast<float>(pose.reprojection_error));
}

Pose from_message(const schema::Pose& pose) {
  Pose result;
  for (int i = 0; i < 9; ++i) result.rotation[i] = pose.rotation()->Get(i);
  for (int i = 0; i < 3; ++i) result.translation[i] = pose.translation()->Get(i);
  result.reprojection_error = pose.reprojection_error();
  return result;
}

schema::Marker to_message(const Marker& marker) {
  std::array<schema::Point2f, 4> corners;
  for (int i = 0; i < 4; ++i) {
    corners[i] = schema::Point2f(static_cast<float>(marker.corners[i][0]),
                                 static_cast<float>(marker.corners[i][1]));
  }
  return schema::Marker(marker.id, marker.hamming,
                        ::flatbuffers::span<const schema::Point2f, 4>(corners),
                        marker.pose.has_value(), marker.pose ? to_message(*marker.pose)
                                                             : schema::Pose());
}

Marker from_message(const schema::Marker& marker) {
  Marker result{.id = marker.id(), .hamming = marker.hamming(), .corners = {}, .pose = {}};
  for (int i = 0; i < 4; ++i) {
    result.corners[i] = {marker.corners()->Get(i)->x(), marker.corners()->Get(i)->y()};
  }
  if (marker.has_pose()) result.pose = from_message(marker.pose());
  return result;
}

std::vector<Marker> markers_of(const schema::FiducialDetectionsT& message) {
  std::vector<Marker> markers;
  markers.reserve(message.markers.size());
  for (const schema::Marker& marker : message.markers) markers.push_back(from_message(marker));
  return markers;
}

std::optional<CharucoObservation> board_of(const schema::FiducialDetectionsT& message) {
  if (!message.board) return std::nullopt;
  const schema::CharucoViewT& view = *message.board;
  CharucoObservation result;
  result.markers = markers_of(message);
  result.corners.reserve(view.corners.size());
  for (const schema::CharucoCorner& corner : view.corners) {
    result.corners.push_back({.id = corner.id(),
                              .image = {corner.image().x(), corner.image().y()},
                              .board = {corner.board().x(), corner.board().y(), corner.board().z()}});
  }
  if (view.has_pose && view.pose) result.pose = from_message(*view.pose);
  result.quality = CharucoQuality{.corners_found = static_cast<int>(view.corners.size()),
                                  .corners_total = view.corners_total,
                                  .markers_found = view.markers_found,
                                  .markers_total = view.markers_total,
                                  .coverage = view.coverage};
  return result;
}

}  // namespace holoscan::fiducials
