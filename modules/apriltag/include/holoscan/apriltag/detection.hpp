// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Types shared by the detector backends, the operators and the tests; no SDK.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::apriltag {

// Where a backend runs its detection stage.
enum class Backend : std::uint8_t {
  kCpu,   // AprilTag 3 reference library on the CPU.
  kCuda,  // The whole detector on the GPU.
  kVpi,   // VPI: the PVA when the platform has one and the image fits, the CPU otherwise.
  kAuto,  // One of the above, chosen once at start by resolve_auto().
};

[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;
// Parses "cpu", "cuda", "vpi" or "auto"; nullopt for anything else.
[[nodiscard]] std::optional<Backend> parse_backend(std::string_view name) noexcept;
// Backends compiled into this build, in a stable order; never kAuto, which picks one of them.
[[nodiscard]] std::vector<Backend> available_backends();

// What Backend::kAuto's choice depends on.
struct PlatformFacts {
  bool vpi_compiled{};  // This build has the vpi backend.
  bool has_pva{};       // VPI reports a PVA (Jetson Orin and Thor).
  bool cuda_device{};   // A CUDA device is usable.
};

// Backend::kAuto's choice: vpi where it is compiled in and there is a PVA, else
// cuda where there is a usable CUDA device, else cpu.
[[nodiscard]] constexpr Backend resolve_auto(const PlatformFacts& facts) noexcept {
  if (facts.vpi_compiled && facts.has_pva) {
    return Backend::kVpi;
  }
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

// Pinhole intrinsics in pixels, pixel centers at integers (OpenCV, ROS `CameraInfo`).
struct CameraIntrinsics {
  double fx{};
  double fy{};
  double cx{};
  double cy{};
};

// Detector settings. VPI uses only `family` and `max_hamming`.
struct DetectorOptions {
  std::string family{"tag36h11"};  // An AprilTag 3 family name.
  int max_hamming{1};              // Bit errors corrected when decoding, 0..2.
  int decimate{2};                 // Quads are found at 1/decimate resolution, 1..8.
  bool refine_edges{true};         // Refine quad edges at full resolution.
  double decode_sharpening{0.25};  // Laplacian sharpening of the bit values, 0..1.
  int min_white_black_diff{5};     // Tiles with less contrast are not segmented, 1..255.
};

// Rejects options no backend accepts, naming the offending field.
// @throws std::invalid_argument
void validate_options(const DetectorOptions& options);

// The tag frame in the camera frame: row-major rotation, translation in meters.
// Camera x right, y down, z forward; tag x right, y down as printed, z into the tag.
struct TagPose {
  std::array<double, 9> rotation{};
  std::array<double, 3> translation{};
  double error{};  // Object-space error the pose solver minimized.
};

struct Detection {
  int id{};
  int hamming{};
  float decision_margin{};
  // Pixel coordinates; pixel centers are at integer coordinates.
  std::array<double, 2> center{};
  // From the tag's bottom-left corner, counter-clockwise as printed (AprilTag 3).
  std::array<std::array<double, 2>, 4> corners{};
  std::optional<TagPose> pose;
};

// Most tags one frame reports, for every backend; the rest are counted in
// Overflow::detections.
inline constexpr std::size_t kMaxDetectionsPerFrame = 64;

// What a frame lost to a backend's fixed capacities; all zero unless the frame
// was too busy for them. The CUDA backend keeps whichever points, candidates and
// quads its threads wrote first, so a frame over capacity can miss tags, and
// which ones varies from run to run.
struct Overflow {
  std::uint32_t points{};      // CUDA: boundary points past the segmenter's capacity.
  std::uint32_t candidates{};  // CUDA: quad candidates past the fitting capacity.
  std::uint32_t quads{};       // CUDA: fitted quads past the decoding capacity.
  // Tags found past kMaxDetectionsPerFrame. VPI keeps at most one more than it
  // returns, so for VPI this is a lower bound.
  std::uint32_t detections{};

  // Whether an earlier stage dropped work, so tags may be missing.
  [[nodiscard]] bool truncated() const noexcept {
    return points != 0 || candidates != 0 || quads != 0;
  }
};

// One frame's results: at most kMaxDetectionsPerFrame tags, best first where the
// backend ranks them, and what did not fit.
struct FrameDetections {
  std::vector<Detection> tags;
  Overflow overflow;
};

}  // namespace holoscan::apriltag
