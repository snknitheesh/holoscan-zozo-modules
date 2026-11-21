// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Image sources for the application: synthetic frames of a scene with known
// depth, and a list of JPEG or PPM files. Both emit a fixed number of frames
// on a clock, numbered from 1, then idle. A frame that busy consumers push
// back (an emit refused with kBackpressured, or no free slot in the output's
// pool, kResourceExhausted) is sent again on the next tick with its original
// capture time; on one worker, as the lossless graphs run, the consumers then
// catch up between ticks. And a V4L2 webcam, for real time: connected with
// only the newest frame kept, it is never held back.

#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/mono_depth/image_fit.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "image_files.hpp"

namespace mono_depth_app {

using holoscan::mono_depth::channels;
using holoscan::mono_depth::kMaxImageSide;
using holoscan::mono_depth::PixelFormat;

struct SyntheticImageSourceConfig {
  int width{1280};
  int height{720};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  PixelFormat format{PixelFormat::kRgb8};
  // Emit frames in GPU memory rather than host memory.
  bool device_output{true};
  // Emit frame 0 of the scene every time, as a still camera of a still scene.
  bool still{false};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const SyntheticImageSourceConfig& config);

// Frame f (1, 2, ...) is make_depth_scene(width, height, f - 1), or frame 0
// when still, drawn in `format`.
class SyntheticImageSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit SyntheticImageSourceOp(SyntheticImageSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  SyntheticImageSourceConfig config_;
  std::vector<std::uint8_t> pixels_;
  std::uint32_t emitted_{};
  std::optional<holoscan::TimePoint> pending_capture_;
  bool copy_logged_{};
};

struct ImageFileSourceConfig {
  std::vector<std::filesystem::path> files;
  // Each file's kind and size, as read_image_info() gives them, read once by
  // the application: the SDK constructs the operator more than once, and
  // reading the files in each would repeat the work and its failures.
  std::vector<ImageFileInfo> infos;
  // Frames to emit; 0 emits every file once.
  std::uint32_t count{0};
  // Start over after the last file until `count` frames are out.
  bool loop{false};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Emit BGR8 rather than RGB8.
  bool bgr{false};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const ImageFileSourceConfig& config);

// Frame f (1, 2, ...) is files[(f - 1) % files.size()], decoded into GPU
// memory as RGB8 (or BGR8): JPEG files with nvJPEG, PPM files on the host.
class ImageFileSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit ImageFileSourceOp(ImageFileSourceConfig config);
  ~ImageFileSourceOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // Frames this source emits in total.
  [[nodiscard]] std::uint32_t frame_count() const noexcept { return total_; }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  ImageFileSourceConfig config_;
  std::vector<ImageFileInfo> infos_;
  std::uint64_t max_bytes_{};
  std::uint32_t total_{};
  std::uint32_t emitted_{};
  std::optional<holoscan::TimePoint> pending_capture_;
  std::unique_ptr<JpegDecoder> decoder_;
  // The JPEG bytes the decoder reads on the GPU stream, kept until the event
  // after the decode has passed: the next frame waits for it, not this one.
  std::vector<std::uint8_t> jpeg_;
  cudaEvent_t decoded_{};
  bool jpeg_logged_{};
  bool ppm_logged_{};
};

struct CameraSourceConfig {
  std::filesystem::path device{"/dev/video0"};
  int width{640};
  int height{480};
  // The camera's frames to run for; 0 runs until the application is stopped.
  std::uint32_t count{0};
};

// Checks that `config.device` offers YUYV at the configured size.
// @throws std::invalid_argument naming what is wrong, with the YUYV sizes the
// device offers when the size is the problem.
void check_camera(const CameraSourceConfig& config);

// RGB8 frames in host memory from a V4L2 camera offering YUYV at the
// requested size, stamped with their capture time and numbered by the
// camera's own frame count from 1, so that every frame the camera captured
// and the demo did not show (passed over for a newer one here, or refused by
// a busy consumer) leaves a gap the reporter counts as skipped. Only the
// camera's newest frame goes out. Connect it with drop-oldest edges
// (edges.hpp), so that a frame arriving while the estimator is busy replaces
// the one waiting; on a blocking edge, a frame the estimator has no room for
// is dropped here.
class CameraSourceOp final : public holoscan::Operator<> {
 public:
  // The device is opened in start() and closed in stop(), not here: the runtime may
  // construct an operator more than once, and a camera serves one owner at a time.
  explicit CameraSourceOp(CameraSourceConfig config);
  ~CameraSourceOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  struct Mapping {
    void* data{};
    std::size_t length{};
  };

  void unmap() noexcept;
  void release() noexcept;

  CameraSourceConfig config_;
  int fd_{-1};
  std::size_t pitch_{};
  std::vector<Mapping> buffers_;
  // The camera's sequence number of its first frame this run, and the id of
  // the newest frame sent.
  std::optional<std::uint32_t> first_sequence_;
  std::uint64_t last_id_{};
};

}  // namespace mono_depth_app
