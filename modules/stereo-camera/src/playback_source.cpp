// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The playback backend: the frames of a recording, in order, once or in a loop.

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include <holoscan/stereo_camera/recording.hpp>
#include <holoscan/stereo_camera/source.hpp>

#include "sources.hpp"

namespace holoscan::stereo_camera::detail {
namespace {

class PlaybackSource final : public StereoSource {
 public:
  explicit PlaybackSource(const SourceConfig& config)
      : reader_(config.playback.directory),
        loop_(config.playback.loop),
        rgb_(config.rgb),
        depth_(config.depth) {
    layout_.left = reader_.left_layout();
    layout_.right = reader_.right_layout();
    if (rgb_) layout_.rgb = reader_.rgb_layout();
    if (depth_) layout_.depth = reader_.depth_layout();
    layout_.live = false;
  }

  [[nodiscard]] Backend backend() const noexcept override { return Backend::kPlayback; }
  [[nodiscard]] const SourceLayout& layout() const noexcept override { return layout_; }

  void open() override {
    next_ = 0;
    opened_ = true;
  }

  [[nodiscard]] const StereoCalibration& calibration() const override {
    if (!opened_) {
      throw std::logic_error("playback: calibration() before open()");
    }
    return reader_.calibration();
  }

  [[nodiscard]] std::optional<StereoFrameView> grab() override {
    if (!opened_) {
      throw std::logic_error("playback: grab() before open()");
    }
    if (next_ >= reader_.size()) {
      if (!loop_) {
        return std::nullopt;
      }
      next_ = 0;
    }
    const std::size_t index = next_;
    try {
      reader_.read(index, left_, right_, rgb_ ? &rgb_image_ : nullptr,
                   depth_ ? &depth_image_ : nullptr);
    } catch (const std::invalid_argument& error) {
      throw std::runtime_error(error.what());
    }
    ++next_;
    const RecordedFrame& recorded = reader_.frames()[index];
    StereoFrameView frame{.left = left_.view(),
                          .right = right_.view(),
                          .index = recorded.index,
                          .timestamp_ns = recorded.timestamp_ns,
                          .device_sequence = recorded.device_sequence};
    if (rgb_) frame.rgb = rgb_image_.view();
    if (depth_) frame.depth = depth_image_.view();
    return frame;
  }

  [[nodiscard]] bool finished() const noexcept override {
    return opened_ && !loop_ && next_ >= reader_.size();
  }

  void close() noexcept override { opened_ = false; }

 private:
  RecordingReader reader_;
  bool loop_;
  bool rgb_;
  bool depth_;
  SourceLayout layout_;
  bool opened_{};
  std::size_t next_{};
  Image left_;
  Image right_;
  Image rgb_image_;
  Image depth_image_;
};

}  // namespace

std::unique_ptr<StereoSource> create_playback_source(const SourceConfig& config) {
  return std::make_unique<PlaybackSource>(config);
}

}  // namespace holoscan::stereo_camera::detail
