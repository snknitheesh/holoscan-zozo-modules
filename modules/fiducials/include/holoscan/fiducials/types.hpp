// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Types shared by the detectors, the operators and the tests; no SDK, no CUDA.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::fiducials {

using Point2 = std::array<double, 2>;
using Point3 = std::array<double, 3>;

// Where a detector runs.
enum class Backend : std::uint8_t {
  kCpu,   // Every stage on the CPU.
  kCuda,  // Thresholding, components, quads, refinement and decoding on the GPU.
  kAuto,  // One of the above, chosen once at start by resolve_auto().
};

[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;
// Parses "cpu", "cuda" or "auto"; nullopt for anything else.
[[nodiscard]] std::optional<Backend> parse_backend(std::string_view name) noexcept;
// The backends a detector can run on; never kAuto, which picks one of them.
[[nodiscard]] std::vector<Backend> available_backends();

// What Backend::kAuto's choice depends on.
struct PlatformFacts {
  bool cuda_device{};  // A CUDA device is usable.
};

// Backend::kAuto's choice: cuda where there is a usable CUDA device, else cpu.
[[nodiscard]] constexpr Backend resolve_auto(const PlatformFacts& facts) noexcept {
  return facts.cuda_device ? Backend::kCuda : Backend::kCpu;
}

// Where a gray image lives.
enum class MemoryLocation : std::uint8_t { kHost, kDevice };

// A borrowed 8-bit single-channel image. `pitch` is the row stride in bytes.
struct GrayImageView {
  const std::uint8_t* data{};
  int width{};
  int height{};
  std::size_t pitch{};
  MemoryLocation location{MemoryLocation::kHost};
};

// A pinhole camera in pixels, pixel centers at integer coordinates (OpenCV, ROS `CameraInfo`),
// with OpenCV's radial-tangential distortion k1, k2, p1, p2, k3 (all zero: none).
struct CameraIntrinsics {
  double fx{};
  double fy{};
  double cx{};
  double cy{};
  std::array<double, 5> distortion{};
};

// A target's frame in the camera frame: row-major rotation, translation in meters. Camera x
// right, y down, z forward; target x right, y down as printed, z into the target (the
// convention of holoscan-apriltag).
struct Pose {
  std::array<double, 9> rotation{};
  std::array<double, 3> translation{};
  double reprojection_error{};  // Root mean square, in pixels.
};

// One marker found in an image.
struct Marker {
  int id{};
  int hamming{};  // Bits corrected while decoding.
  // Pixel coordinates of the outer corners of the black border, pixel centers at integers, in
  // the order OpenCV uses: top-left, top-right, bottom-right, bottom-left as printed.
  std::array<Point2, 4> corners{};
  std::optional<Pose> pose;
};

// Detector settings, for both backends.
struct DetectorOptions {
  // A predefined dictionary by OpenCV's name: dictionary_names() lists them.
  std::string dictionary{"DICT_4X4_50"};
  // Fraction of the bits the dictionary can correct that are corrected, 0..1 (OpenCV's
  // errorCorrectionRate).
  double error_correction_rate{0.6};
  // Side of the threshold tiles, 2..64 pixels: each pixel is compared with the middle of the
  // darkest and brightest values within about 1.5 tiles of it.
  int tile_size{8};
  // Least difference, in gray levels, between dark and bright for the thresholding and for a
  // marker's cells, 1..255.
  int min_contrast{20};
  // Shortest side a marker may have, in pixels, 4..1000.
  int min_marker_px{10};
  // Refine the corners to sub-pixel accuracy by fitting the border's edges.
  bool refine_corners{true};
};

// Rejects options no backend accepts, naming the offending field.
// @throws std::invalid_argument
void validate_options(const DetectorOptions& options);

// Most markers one frame reports; the rest are counted in Overflow::markers.
inline constexpr std::size_t kMaxMarkersPerFrame = 256;

// What a frame lost to a detector's fixed capacities; all zero unless the frame was too busy.
// Past the candidate capacity, the cpu backend keeps the first candidates in scan order and the
// cuda backend whichever its threads reach first, so which markers such a frame misses differs
// between them, and on the GPU from run to run.
struct Overflow {
  std::uint32_t candidates{};  // Dark regions past the detector's capacity, not examined.
  std::uint32_t markers{};     // Markers found past kMaxMarkersPerFrame.

  // Whether markers may be missing because candidates were dropped.
  [[nodiscard]] bool truncated() const noexcept { return candidates != 0; }
};

// One frame's markers, sorted by id and then by position, and what did not fit.
struct FrameMarkers {
  std::vector<Marker> markers;
  Overflow overflow;
};

// A chessboard corner to refine: where it is predicted and the half side of the square window
// the refinement may look in.
struct SaddleSeed {
  Point2 position{};
  int half_window{};
};

}  // namespace holoscan::fiducials
