// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// How an operator takes its camera info: frames aligned by capture time, and
// the camera info for them, sent with every frame or at its own rate.
//
//   class MyOp : public holoscan::Operator<> {
//     holoscan::Input<holoscan::schema::ImageT> left, right;
//     holoscan::Input<holoscan::messages::schema::StereoCameraInfoT> camera_info;
//     holoscan::messages::LatestCameraInfo<holoscan::messages::schema::StereoCameraInfoT> info_;
//
//     void setup(holoscan::OperatorSpec& spec) override {
//       spec.input(left, "left");
//       spec.input(right, "right");
//       spec.input(camera_info, "camera_info");  // Mandatory: wire it.
//     }
//     holoscan::Contract contract() const override {
//       return holoscan::messages::camera_info_contract(camera_info, left, right);
//     }
//     holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
//       auto info = info_.update(camera_info);
//       if (!info) return holoscan::make_unexpected(info.error());
//       if (*info == nullptr) return {};  // No camera info yet: skip the frame.
//       if (info_.changed()) { /* recompute what depends on the calibration */ }
//       ...
//     }
//   };
//
// Why this contract: with kLatest, camera info is selected LatestWithin with
// a SafeDefault, and LatestCameraInfo keeps the newest one. In the SDK 5.0,
// LatestWithin takes each camera info once, and a Skip or ReuseLast policy
// holds back every frame that comes before the first camera info, which
// stalls the pipeline for good when a frame arrives first; so the operator
// keeps the camera info itself, and a frame before the first one is skipped.

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/conversion.hpp>

namespace holoscan::messages {

enum class CameraInfoTiming : std::uint8_t {
  // Camera info at any rate (CameraInfoSourceOp sends it periodically): each
  // frame uses the newest camera info received before it, which is the frame's
  // own when the producer sends it first. Frames before the first camera info
  // are skipped. The default.
  kLatest = 0,
  // One camera info with every frame, carrying the frame's capture time: each
  // frame waits for its own, so they always match. A frame whose camera info
  // never comes is never processed, so use this only with a producer that
  // sends camera info with every frame.
  kEveryFrame = 1,
};

struct CameraInfoInputOptions {
  CameraInfoTiming timing{CameraInfoTiming::kLatest};
  // kLatest: a camera info older than this when a frame arrives is not taken
  // for it (one already taken stays in use). Keep it above the camera info
  // producer's period: CameraInfoSourceOp sends one a second by default.
  std::chrono::nanoseconds max_age{std::chrono::seconds{10}};
};

namespace detail {
void check_options(const CameraInfoInputOptions& options);
}  // namespace detail

// The contract of an operator whose `frames` (one input, or several with one
// capture time, as a stereo pair's left and right) need `camera_info`: the
// frames aligned exactly by capture time, and the camera info as `options`
// say. Add the operator's other inputs to the result. (The inputs are typed:
// the SDK checks a contract's selectors against the ports' payload types.)
// @throws std::invalid_argument for a max_age that is not > 0.
template <typename Info, typename... Frames>
[[nodiscard]] holoscan::Contract camera_info_contract(const CameraInfoInputOptions& options,
                                                      const holoscan::Input<Info>& camera_info,
                                                      const holoscan::Input<Frames>&... frames) {
  static_assert(sizeof...(Frames) > 0, "camera_info_contract needs at least one frame input");
  using std::chrono_literals::operator""ns;
  detail::check_options(options);
  holoscan::Contract contract;
  if (options.timing == CameraInfoTiming::kEveryFrame) {
    contract.trigger(
        holoscan::OnAligned{.ports = holoscan::ports(frames..., camera_info), .window = 0ns});
    (contract.input(frames, holoscan::Exact{}), ...);
    contract.input(camera_info, holoscan::Exact{});
    return contract;
  }
  if constexpr (sizeof...(Frames) == 1) {
    // OnEach pins its frame; it takes no selector of its own.
    (contract.trigger(holoscan::OnEach{frames}), ...);
  } else {
    contract.trigger(holoscan::OnAligned{.ports = holoscan::ports(frames...), .window = 0ns});
    (contract.input(frames, holoscan::Exact{}), ...);
  }
  contract.input(camera_info, holoscan::LatestWithin{.max_age = options.max_age});
  contract.on_missing(camera_info, holoscan::SafeDefault{});
  return contract;
}

// The same with the default options: kLatest, max_age 10 s.
template <typename Info, typename... Frames>
[[nodiscard]] holoscan::Contract camera_info_contract(const holoscan::Input<Info>& camera_info,
                                                      const holoscan::Input<Frames>&... frames) {
  return camera_info_contract(CameraInfoInputOptions{}, camera_info, frames...);
}

// The edge for camera info with kLatest: the newest one replaces one the
// consumer has not taken yet, and the producer never waits for the consumer.
// (With kEveryFrame, give camera info the frames' own edge options.)
[[nodiscard]] inline holoscan::ConnectionOptions latest_camera_info_connection() noexcept {
  return holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                     .queue_depth = 1U};
}

namespace detail {
// The error for a camera info that the runtime could not select, or nothing.
[[nodiscard]] std::optional<holoscan::Error> selection_error(holoscan::SelectionReason reason);
[[nodiscard]] holoscan::Error invalid_camera_info(const std::string& what);
}  // namespace detail

// The newest valid camera info an operator received, for
// schema::CameraInfoT (Value CameraInfo) or schema::StereoCameraInfoT (Value
// StereoCameraInfo). Not thread-safe: one per operator, used from compute().
template <typename Message>
class LatestCameraInfo {
 public:
  using Value = decltype(from_message(std::declval<const Message&>()));

  // Takes this activation's camera info from `port` when there is a new one,
  // checks it with validate() and keeps it.
  // @return The newest valid camera info, or nullptr while none has arrived;
  // an error for a camera info that is not valid, or one whose capture time
  // cannot be compared with the frames' (none, or another clock domain).
  [[nodiscard]] holoscan::expected<const Value*, holoscan::Error> update(
      holoscan::Input<Message>& port) {
    changed_ = false;
    if (port.is_missing()) {
      if (auto error = detail::selection_error(port.selection().reason)) {
        return holoscan::make_unexpected(std::move(*error));
      }
      return get();
    }
    auto sample = port.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    try {
      Value value = from_message(sample->data);
      changed_ = !latest_ || !same_calibration(*latest_, value);
      latest_ = std::move(value);
    } catch (const std::invalid_argument& error) {
      return holoscan::make_unexpected(detail::invalid_camera_info(error.what()));
    }
    return get();
  }

  // The newest valid camera info, or nullptr.
  [[nodiscard]] const Value* get() const noexcept { return latest_ ? &*latest_ : nullptr; }

  // Whether the last update() took a calibration other than the one before
  // (the first one counts): headers aside, any field differs.
  [[nodiscard]] bool changed() const noexcept { return changed_; }

  // Forgets the camera info, as an operator's start() does for a new run.
  void reset() noexcept {
    latest_.reset();
    changed_ = false;
  }

 private:
  std::optional<Value> latest_;
  bool changed_{};
};

}  // namespace holoscan::messages
