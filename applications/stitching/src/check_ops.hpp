// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's checks: StitchCheckOp compares each frame's bird's-eye view and
// panorama with the ground truth rendered directly from the synthetic world
// (PSNR and the checkerboards' corners); RenderCheckOp checks HoloViz's
// offscreen renders and can write them as PPM.

#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/stitching/camera_rig.hpp>
#include <holoscan/stitching/projection.hpp>
#include <holoscan/stitching/synthetic_world.hpp>

namespace stitching_app {

// The worst of each output's comparisons so far.
struct Worst {
  double psnr_db{99.0};
  double coverage{1.0};
  double max_corner_error{};
  double gain_error{};   // |gain - 1|, of the stitch's mean against the truth's.
  int corners{1 << 30};  // The fewest corners checked in a frame.
};

// What the checks found, shared with main().
struct Report {
  std::atomic<std::uint32_t> frames{0};    // Frames whose outputs were checked.
  std::atomic<std::uint32_t> rendered{0};  // Renders checked.
  std::mutex mutex;                        // Guards what follows.
  Worst birds_eye;
  Worst panorama;
  std::uint32_t failed_frames{};
  std::uint32_t failed_renders{};
  std::string first_failure;
};

// What a frame must reach to pass. The PSNR bounds catch gross errors: the
// stitch is as sharp as the cameras, and the truth sharper, where the output
// samples the cameras more finely than they see (the far ground).
struct Thresholds {
  double birds_eye_psnr_db{26.0};
  double panorama_psnr_db{28.0};
  double coverage{0.9};
  double corner_error_px{1.0};
  // With exposure compensation, the stitch's overall brightness may differ.
  double gain_error{0.02};
  int corners{50};
};

struct StitchCheckConfig {
  std::shared_ptr<Report> report;
  holoscan::stitching::CameraRig rig;
  holoscan::stitching::SyntheticWorld world;
  holoscan::stitching::OutputGeometry birds_eye;
  holoscan::stitching::OutputGeometry panorama;
  Thresholds thresholds;
};

// Receives a frame's bird's-eye view and panorama (RGBA8), paired by capture
// time, and compares each with the truth of the frame's pose.
class StitchCheckOp final : public holoscan::Operator<> {
 public:
  explicit StitchCheckOp(StitchCheckConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::ImageT> birds_eye;
  holoscan::Input<holoscan::schema::ImageT> panorama;

 private:
  StitchCheckConfig config_;
  void* truth_{};  // Device memory for the truth renders.
  // resolution_map() of each output, made at the first frame.
  std::vector<float> birds_eye_resolution_;
  std::vector<float> panorama_resolution_;
};

// A rectangle of the render, in pixels.
struct Tile {
  int x{};
  int y{};
  int width{};
  int height{};
};

struct RenderCheckConfig {
  std::shared_ptr<Report> report;
  int width{};
  int height{};
  // Where HoloViz draws each image: every tile must show a picture.
  std::vector<Tile> tiles;
  // When set, every render is also written there as PPM.
  std::filesystem::path image_dir;
};

// Receives HoloViz's offscreen renders and checks that every tile shows a
// picture (not a flat colour) and that the render is the expected size.
class RenderCheckOp final : public holoscan::Operator<> {
 public:
  explicit RenderCheckOp(RenderCheckConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::ImageT> rendered;

 private:
  RenderCheckConfig config_;
};

}  // namespace stitching_app
