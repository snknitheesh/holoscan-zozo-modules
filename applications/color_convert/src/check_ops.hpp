// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's checks, against the module's CPU reference of each frame:
// FrameCheckOp compares what ColorConvertOp and ResizeOp made of a camera
// frame; RenderCheckOp compares HoloViz's offscreen render of them.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/color_convert/pixel_format.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace color_convert_app {

// How far an output is from the reference, per stored byte.
struct Difference {
  double mean{};
  int p99{};
  int max{};
  std::uint64_t wrong{};  // Bytes that differ at all.
};

// What the camera sends, and what the two operators were told to make of it.
struct Pipeline {
  holoscan::color_convert::PixelFormat camera{holoscan::color_convert::PixelFormat::kNv12};
  int width{1920};
  int height{1080};
  holoscan::color_convert::PixelFormat converted{holoscan::color_convert::PixelFormat::kRgba8};
  holoscan::color_convert::PixelFormat resized{holoscan::color_convert::PixelFormat::kRgba8};
  int resized_width{640};
  int resized_height{640};
  holoscan::color_convert::ConversionParams resize_params;
};

// The bytes the camera sends for `frame`, and what ColorConvertOp and
// ResizeOp should make of them.
std::vector<std::uint8_t> camera_frame(const Pipeline& p, std::uint32_t frame);
std::vector<std::uint8_t> expected_converted(const Pipeline& p, std::uint32_t frame);
std::vector<std::uint8_t> expected_resized(const Pipeline& p, std::uint32_t frame);

Difference compare(const std::vector<std::uint8_t>& got, const std::vector<std::uint8_t>& want);

// What the checks found, shared with main().
struct Report {
  std::atomic<std::uint32_t> frames{0};    // Frames whose outputs were checked.
  std::atomic<std::uint32_t> rendered{0};  // Renders checked.
  // The backend each operator ran, as it reported it; -1 before the first frame.
  std::atomic<int> convert_backend{-1};
  std::atomic<int> resize_backend{-1};
  std::mutex mutex;      // Guards what follows.
  Difference converted;  // The worst seen.
  Difference resized;
  Difference render;
  std::uint32_t failed_frames{};
  std::uint32_t failed_renders{};
  std::uint32_t insets_seen{};  // Renders that show the inset's letterbox band.
  std::string first_failure;
};

// Byte-exact (the cuda backend) or within the VIC's tolerance.
struct Tolerance {
  bool exact{true};
  double mean{1.5};
  int p99{4};
};

struct FrameCheckConfig {
  std::shared_ptr<Report> report;
  Pipeline pipeline;
  Tolerance converted;
  Tolerance resized;
};

// Receives a frame's two outputs, paired by capture time, and checks each
// against the reference.
class FrameCheckOp final : public holoscan::Operator<> {
 public:
  explicit FrameCheckOp(FrameCheckConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::ImageT> converted;
  holoscan::Input<holoscan::schema::ImageT> resized;

 private:
  FrameCheckConfig config_;
};

struct RenderCheckConfig {
  std::shared_ptr<Report> report;
  Pipeline pipeline;
  // The inset's place in the render, in pixels: HoloViz draws the resized
  // image there over the converted one.
  int inset_x{};
  int inset_y{};
  int inset_width{};
  int inset_height{};
  // When set, every render is also written there as PPM.
  std::filesystem::path image_dir;
};

// Receives HoloViz's offscreen renders, finds the camera frame each shows
// from where its colour bars are, and checks that outside the inset it is
// that frame converted, and that the inset shows the letterbox's fill.
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

}  // namespace color_convert_app
