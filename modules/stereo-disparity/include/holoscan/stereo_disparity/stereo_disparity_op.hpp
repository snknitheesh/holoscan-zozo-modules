// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StereoDisparityOp: a rectified left and right image in, the left image's
// disparity out, estimated by NVIDIA VPI 4's stereo disparity estimator on the
// GPU (cuda), the Optical Flow Accelerator (ofa) or the OFA with the PVA (pva);
// optionally also the depth, from the pair's geometry, and VPI's confidence.
//
// The input must already be rectified: rows of the two images must see the
// same scene rows. A RealSense's infrared pair is rectified by the camera.
//
// The left and right images, and the geometry when it is an input, pair by
// capture time: the operator runs once all of them with exactly the same
// capture time have arrived. Send every part of a stereo frame with one
// capture time; a part with another time never completes its frame, which
// gives no output.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/stereo_disparity/disparity_options.hpp>
#include <holoscan/stereo_disparity/geometry_message.hpp>
#include <holoscan/stereo_disparity/stereo_geometry.hpp>

namespace holoscan::stereo_disparity {

// What an operator has done, for its owner to read at any time.
struct StereoDisparityStats {
  // The backend that ran the latest frame; kAuto before the first.
  std::atomic<Backend> backend{Backend::kAuto};
  std::atomic<std::uint64_t> frames{};
  // Time in compute() per frame, from receiving the pair to having the outputs
  // written, without building the matcher (once per backend and size).
  std::atomic<std::uint64_t> compute_ns_total{};
  std::atomic<std::uint64_t> compute_ns_max{};
};

struct StereoDisparityConfig {
  // kAuto chooses on the first frame, when the image size is known: see
  // resolve_auto(). The default is cuda, which runs everywhere.
  Backend backend{Backend::kCuda};
  DisparityParams params;
  // The pair's geometry, for the `depth` output. Give it here, or enable
  // `geometry_input` to take it with every frame, not both.
  std::optional<StereoGeometry> geometry;
  bool geometry_input{false};
  // Declare the `depth` output; needs the geometry.
  bool depth{false};
  // Declare the `confidence` output: VPI's confidence map, on cuda and pva.
  bool confidence{false};
  // Largest image the outputs are sized for.
  int max_width{1920};
  int max_height{1080};
  // Counts frames and compute time when given.
  std::shared_ptr<StereoDisparityStats> stats;
};

// @throws std::invalid_argument naming the first unusable field, or a backend
//         this platform does not have.
void validate_config(const StereoDisparityConfig& config);

class StereoDisparityOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit StereoDisparityOp(StereoDisparityConfig config);
  ~StereoDisparityOp() override;
  StereoDisparityOp(const StereoDisparityOp&) = delete;
  StereoDisparityOp& operator=(const StereoDisparityOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The backend this operator runs: the configured one, or for kAuto the one
  // chosen on the first frame, and kAuto until then. Safe from any thread.
  [[nodiscard]] Backend resolved_backend() const noexcept;

  // MONO8, RGB8, BGR8, RGBA8 or BGRA8, [H, W] or [H, W, C], rows packed, in
  // host or CUDA device memory; both the same size, already rectified. Color
  // is matched as its BT.601 luma.
  holoscan::Input<holoscan::schema::ImageT> left;
  holoscan::Input<holoscan::schema::ImageT> right;
  // With `geometry_input`: the pair's geometry, with its capture time.
  holoscan::Input<schema::StereoGeometryT> geometry;
  // MONO32F [H, W] on the GPU: the left image's disparity in pixels (left
  // column minus right column), 0 where invalid. Carries the left image's
  // capture time, frame id and header.
  holoscan::Output<holoscan::schema::ImageT> disparity;
  // With `depth`: MONO32F [H, W] on the GPU, meters along the optical axis
  // (fx * baseline / (disparity + doffs)), 0 where invalid.
  holoscan::Output<holoscan::schema::ImageT> depth;
  // With `confidence`: MONO16 [H, W] on the GPU, VPI's confidence, 65535 most
  // confident.
  holoscan::Output<holoscan::schema::ImageT> confidence;

 private:
  // The matcher, the staging buffer, a CUDA stream and the one-time logs.
  struct State;

  StereoDisparityConfig config_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::stereo_disparity
