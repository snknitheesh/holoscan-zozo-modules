// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// PointCloudToDepthOp on its own, through the SDK's operator test harness:
// organized cloud messages from host and device memory, which must give back
// the depth image they were made from exactly; unorganized synthetic clouds
// with colors in the body frame, from host and device memory, with camera
// info at its own rate or with every cloud and changing from cloud to cloud,
// checked against the synthetic camera; camera info at its own rate changing
// between clouds, to a usable camera and to one it refuses; images of the
// camera's size; the capture time and frame id carried to the outputs; clouds
// before the first camera info and clouds whose camera info has another
// capture time; and unusable cameras, clouds and configurations rejected.
// Through a graph: clouds from a producer's host, pinned and device pools.
// Also the image headers.
//
// Not tested: that the operator holds pinned points a producer still owns, and
// rejects misaligned points and points on another GPU. A PointCloud message's
// tensors always come from the producer's runtime pool (the SDK 5.0 refuses an
// emit of tensors of other memory), so such clouds cannot reach the operator
// on a single-GPU machine. The holding itself is tested below on its own
// (detail::hold_until_done()).

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/point_cloud/depth_truth.hpp>
#include <holoscan/point_cloud/point_cloud_to_depth_op.hpp>
#include <holoscan/point_cloud/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "host_input.hpp"
#include "tensor_support.hpp"
#include "test_support.hpp"

namespace holoscan::point_cloud {
namespace {

namespace hm = holoscan::messages;
using test::make;
using test::stall;
using test::TestMemory;
using test::wrap;

using std::chrono_literals::operator""ms;
using ImageSnapshot = holoscan::testing::Snapshot<holoscan::schema::ImageT>;

constexpr int kWidth = 96;
constexpr int kHeight = 72;
// Images a test captures at most per output; the harness gives each capture a
// pool slot, and a channel holds a bounded number of buffers.
constexpr std::size_t kCaptures = 16;
constexpr DLDataType kFloat32{kDLFloat, 32, 1};
constexpr DLDataType kUInt8{kDLUInt, 8, 1};

holoscan::Tensor points_tensor(const std::vector<float>& xyz, bool device) {
  return make({static_cast<std::int64_t>(xyz.size() / 3), 3}, kFloat32, xyz.data(),
              xyz.size() * sizeof(float), device);
}

holoscan::Tensor colors_tensor(const std::vector<std::uint8_t>& rgb, bool device) {
  return make({static_cast<std::int64_t>(rgb.size() / 3), 3}, kUInt8, rgb.data(), rgb.size(),
              device);
}

// The empty [0] tensor of a field a cloud does not carry.
holoscan::Tensor absent(DLDataType dtype, bool device) {
  return make({0}, dtype, nullptr, 0, device);
}

// A cloud message of `points` and, when given, `colors`.
hm::schema::PointCloudT cloud_message(holoscan::Tensor points,
                                      std::optional<holoscan::Tensor> colors = std::nullopt) {
  const bool device = points.device().device_type == kDLCUDA;
  hm::schema::PointCloudT cloud;
  cloud.header = std::make_shared<holoscan::schema::HeaderT>();
  cloud.header->frame_id = "body";
  cloud.organized = points.ndim() == 3;
  cloud.colors =
      std::make_shared<holoscan::Tensor>(colors ? std::move(*colors) : absent(kUInt8, device));
  cloud.points = std::make_shared<holoscan::Tensor>(std::move(points));
  cloud.normals = std::make_shared<holoscan::Tensor>(absent(kFloat32, device));
  cloud.intensity = std::make_shared<holoscan::Tensor>(absent(kFloat32, device));
  return cloud;
}

// The camera info of frame `f`'s camera, as a message.
hm::schema::CameraInfoT camera_info(const SyntheticFrame& f) {
  return hm::to_message(synthetic_camera_info(f, "camera_optical"));
}

// Frame `f`'s depth image deprojected into the camera frame, one point per
// pixel in image order: an organized cloud. Holes give points at z = 0.
std::vector<float> organized_cloud(const SyntheticFrame& f, bool millimeters) {
  std::vector<float> xyz;
  for (int v = 0; v < f.height; ++v) {
    for (int u = 0; u < f.width; ++u) {
      const float z = synthetic_scene_depth(f, millimeters, u, v);
      xyz.insert(xyz.end(), {(u - f.intrinsics.cx) * z / f.intrinsics.fx,
                             (v - f.intrinsics.cy) * z / f.intrinsics.fy, z});
    }
  }
  return xyz;
}

PointCloudToDepthConfig base_config(DepthFormat format) {
  PointCloudToDepthConfig config;
  config.max_width = kWidth;
  config.max_height = kHeight;
  config.depth_min = 0.1F;
  config.depth_max = 10.0F;
  config.depth_format = format;
  return config;
}

// The synthetic clouds' settings: body frame, and colors.
PointCloudToDepthConfig synthetic_config(DepthFormat format) {
  PointCloudToDepthConfig config = base_config(format);
  config.extrinsics = rigid_inverse(synthetic_body_from_camera());
  config.color = true;
  return config;
}

// The capture time of frame `frame` in the tests' frames.
std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

// Checks that an image carries frame `frame`'s capture time and `frame_id`.
void check_provenance(const ImageSnapshot& image, std::uint32_t frame, std::uint64_t frame_id) {
  const holoscan::SampleMetadata& metadata = image.sample_metadata();
  VERIFY_MSG(metadata.capture_timestamp_ns == capture_time(frame).count(),
             "frame " + std::to_string(frame) + " carries capture time " +
                 std::to_string(metadata.capture_timestamp_ns));
  VERIFY_MSG(
      metadata.frame_id == frame_id,
      "frame " + std::to_string(frame) + " carries frame id " + std::to_string(metadata.frame_id));
}

// The bytes of a captured image after checking its descriptor and shape.
template <typename T>
std::vector<T> image_data(const ImageSnapshot& image, holoscan::schema::ImageEncoding encoding,
                          std::int64_t channels) {
  const holoscan::schema::ImageT descriptor = image.metadata();
  VERIFY(descriptor.encoding == encoding);
  VERIFY(descriptor.width == kWidth && descriptor.height == kHeight);
  const auto& tensor = image.tensor(holoscan::schema::ImageTensorFields::data);
  const std::vector<std::int64_t> shape =
      channels == 1 ? std::vector<std::int64_t>{kHeight, kWidth}
                    : std::vector<std::int64_t>{kHeight, kWidth, channels};
  VERIFY(tensor.shape() == shape);
  std::vector<T> values(static_cast<std::size_t>(kWidth) * kHeight * channels);
  VERIFY(tensor.bytes().size() == values.size() * sizeof(T));
  if (tensor.bytes().size() == values.size() * sizeof(T)) {
    std::memcpy(values.data(), tensor.bytes().data(), tensor.bytes().size());
  }
  return values;
}

// Checks a depth image (and colors) against frame `f` of the synthetic camera.
void check_truth(const ImageSnapshot& depth, const ImageSnapshot* color, const SyntheticFrame& f,
                 const PointCloudToDepthConfig& config) {
  const bool uint16 = config.depth_format == DepthFormat::kUint16;
  const SceneTruth truth{.millimeter_depth = uint16,
                         .format = config.depth_format,
                         .depth_scale = config.depth_scale,
                         .depth_min = config.depth_min,
                         .depth_max = config.depth_max,
                         .point_size = config.point_size};
  std::vector<std::uint8_t> rgb;
  if (color != nullptr) {
    rgb = image_data<std::uint8_t>(*color, holoscan::schema::ImageEncoding_RGB8, 3);
  }
  const std::uint8_t* rgb_data = color != nullptr ? rgb.data() : nullptr;
  TruthCheck check;
  if (uint16) {
    const auto counts = image_data<std::uint16_t>(depth, holoscan::schema::ImageEncoding_MONO16, 1);
    check = check_against_scene(f, truth, counts.data(), rgb_data);
  } else {
    const auto meters = image_data<float>(depth, holoscan::schema::ImageEncoding_MONO32F, 1);
    check = check_against_scene(f, truth, meters.data(), rgb_data);
  }
  VERIFY_MSG(check.wrong_depths == 0,
             std::to_string(check.wrong_depths) + " depths differ from the truth");
  VERIFY_MSG(check.wrong_colors == 0,
             std::to_string(check.wrong_colors) + " colors differ from the truth");
}

// An organized cloud made from a depth image gives that image back exactly.
void check_organized_round_trip(bool millimeters, bool device) {
  const PointCloudToDepthConfig config =
      base_config(millimeters ? DepthFormat::kUint16 : DepthFormat::kFloat32);
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 2, false);
  const std::vector<float> xyz = organized_cloud(f, millimeters);
  h.push_frame(holoscan::testing::frame_at(
      capture_time(0), holoscan::testing::input("camera_info", camera_info(f)),
      holoscan::testing::input("point_cloud",
                               cloud_message(make({kHeight, kWidth, 3}, kFloat32, xyz.data(),
                                                  xyz.size() * 4, device)))));
  if (device) {
    h.input_source_device("point_cloud", holoscan::DeviceId{0});
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto images = result.snapshots<holoscan::schema::ImageT>("depth");
  VERIFY(images.size() == 1);
  if (images.empty()) return;
  int wrong = 0;
  int filled = 0;
  std::vector<float> meters;
  std::vector<std::uint16_t> counts;
  if (millimeters) {
    counts = image_data<std::uint16_t>(images[0], holoscan::schema::ImageEncoding_MONO16, 1);
  } else {
    meters = image_data<float>(images[0], holoscan::schema::ImageEncoding_MONO32F, 1);
  }
  for (int v = 0; v < kHeight; ++v) {
    for (int u = 0; u < kWidth; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * kWidth + u;
      const float z = synthetic_scene_depth(f, millimeters, u, v);
      const bool valid = z >= config.depth_min && z <= config.depth_max;
      filled += valid;
      if (millimeters) {
        wrong += counts[i] != (valid ? synthetic_depth_mm(f, u, v) : 0U);
      } else {
        wrong += meters[i] != (valid ? z : 0.0F);
      }
    }
  }
  VERIFY_MSG(wrong == 0, std::to_string(wrong) + " pixels differ from the image");
  VERIFY(filled > kWidth * kHeight / 2);
}

// Frame `f`'s synthetic cloud, with colors, as a message.
hm::schema::PointCloudT synthetic_cloud_message(const SyntheticFrame& f, bool millimeters,
                                                bool device) {
  std::vector<float> xyz;
  std::vector<std::uint8_t> rgb;
  synthetic_cloud(f, millimeters, xyz, rgb);
  return cloud_message(points_tensor(xyz, device), colors_tensor(rgb, device));
}

// Unorganized synthetic clouds with colors, in the body frame, with camera
// info at its own rate: sent once, before the first cloud.
void check_synthetic_clouds(DepthFormat format, int point_size, bool device) {
  PointCloudToDepthConfig config = synthetic_config(format);
  config.point_size = point_size;
  const bool millimeters = format == DepthFormat::kUint16;
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  h.push_frame(holoscan::testing::frame_at(
      capture_time(0) - 1ms,
      holoscan::testing::input("camera_info",
                               camera_info(synthetic_frame(kWidth, kHeight, 0, false)))));
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame),
        holoscan::testing::input(
            "point_cloud", synthetic_cloud_message(synthetic_frame(kWidth, kHeight, frame, false),
                                                   millimeters, device))));
  }
  if (device) {
    h.input_source_device("point_cloud", holoscan::DeviceId{0});
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.output_device("color", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth", kCaptures);
  h.capture_snapshot<holoscan::schema::ImageT>("color", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  const auto colors = result.snapshots<holoscan::schema::ImageT>("color");
  VERIFY(depths.size() == 2 && colors.size() == 2);
  for (std::size_t frame = 0; frame < depths.size() && frame < colors.size(); ++frame) {
    const auto number = static_cast<std::uint32_t>(frame);
    check_truth(depths[frame], &colors[frame], synthetic_frame(kWidth, kHeight, number, false),
                config);
    check_provenance(depths[frame], number, 0U);
    check_provenance(colors[frame], number, 0U);
  }
}

// A cloud's frame id travels to its image.
void check_frame_ids() {
  const PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, false);
    const std::vector<float> xyz = organized_cloud(f, false);
    holoscan::testing::TestSample<hm::schema::PointCloudT> cloud =
        holoscan::testing::sample_at(cloud_message(points_tensor(xyz, false)), capture_time(frame));
    cloud.metadata.frame_id = 41U + frame;
    h.push_samples("point_cloud", {std::move(cloud)});
    h.push_samples("camera_info",
                   {holoscan::testing::sample_at(camera_info(f), capture_time(frame))});
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto images = result.snapshots<holoscan::schema::ImageT>("depth");
  VERIFY(images.size() == 2);
  for (std::size_t frame = 0; frame < images.size(); ++frame) {
    check_provenance(images[frame], static_cast<std::uint32_t>(frame), 41U + frame);
  }
}

PointCloudToDepthConfig every_frame_config() {
  PointCloudToDepthConfig config = synthetic_config(DepthFormat::kUint16);
  config.camera_info.timing = hm::CameraInfoTiming::kEveryFrame;
  return config;
}

// Clouds sent with their camera's info, whose intrinsics change from cloud to
// cloud.
void check_changing_camera(bool device) {
  const PointCloudToDepthConfig config = every_frame_config();
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, true);
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame),
        holoscan::testing::input("point_cloud", synthetic_cloud_message(f, true, device)),
        holoscan::testing::input("camera_info", camera_info(f))));
  }
  if (device) {
    h.input_source_device("point_cloud", holoscan::DeviceId{0});
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.output_device("color", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth", kCaptures);
  h.capture_snapshot<holoscan::schema::ImageT>("color", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  const auto colors = result.snapshots<holoscan::schema::ImageT>("color");
  VERIFY(depths.size() == 3 && colors.size() == 3);
  for (std::size_t frame = 0; frame < depths.size() && frame < colors.size(); ++frame) {
    const auto number = static_cast<std::uint32_t>(frame);
    check_truth(depths[frame], &colors[frame], synthetic_frame(kWidth, kHeight, number, true),
                config);
    check_provenance(depths[frame], number, 0U);
  }
}

// With camera info at its own rate, a new calibration is picked up from the
// next cloud on, and the clouds between keep the one before: camera A, two
// clouds, camera B, one cloud. (Synthetic frames 0 and 5 share A's
// intrinsics; frame 4 has B's, 4 % longer focal lengths, enough to move most
// points to another pixel.)
void check_camera_change_at_own_rate() {
  const PointCloudToDepthConfig config = synthetic_config(DepthFormat::kUint16);
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  const std::array<std::uint32_t, 3> frames{0U, 5U, 4U};
  for (std::uint32_t i = 0; i < frames.size(); ++i) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frames[i], true);
    if (i != 1) {
      h.push_frame(holoscan::testing::frame_at(
          capture_time(i) - 1ms, holoscan::testing::input("camera_info", camera_info(f))));
    }
    h.push_frame(holoscan::testing::frame_at(
        capture_time(i),
        holoscan::testing::input("point_cloud", synthetic_cloud_message(f, true, false))));
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.output_device("color", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth", kCaptures);
  h.capture_snapshot<holoscan::schema::ImageT>("color", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  const auto colors = result.snapshots<holoscan::schema::ImageT>("color");
  VERIFY(depths.size() == 3 && colors.size() == 3);
  for (std::size_t i = 0; i < depths.size() && i < colors.size() && i < frames.size(); ++i) {
    check_truth(depths[i], &colors[i], synthetic_frame(kWidth, kHeight, frames[i], true), config);
  }
}

// A calibration that changes, at the camera info's own rate, to one the
// kernels cannot model fails the next cloud.
void check_camera_change_to_unusable() {
  const PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 0, false);
  const std::vector<float> xyz = organized_cloud(f, false);
  hm::CameraInfo skewed = synthetic_camera_info(f, "camera_optical");
  skewed.skew = -1.0;
  h.push_frame(holoscan::testing::frame_at(
      capture_time(0) - 1ms, holoscan::testing::input("camera_info", camera_info(f))));
  h.push_frame(holoscan::testing::frame_at(
      capture_time(0),
      holoscan::testing::input("point_cloud", cloud_message(points_tensor(xyz, false)))));
  h.push_frame(holoscan::testing::frame_at(
      capture_time(1) - 1ms, holoscan::testing::input("camera_info", hm::to_message(skewed))));
  h.push_frame(holoscan::testing::frame_at(
      capture_time(1),
      holoscan::testing::input("point_cloud", cloud_message(points_tensor(xyz, false)))));
  h.output_device("depth", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "the cloud after the skewed camera info fails");
  VERIFY_MSG(result.describe_failure().find("camera info 'camera_optical' has a skew") !=
                 std::string::npos,
             result.describe_failure());
}

// Camera info sent with every cloud pairs with it by exact capture time: a
// cloud whose camera info carries another time gives no image, while the
// clouds around it still do.
void check_unmatched_camera_info() {
  const PointCloudToDepthConfig config = every_frame_config();
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, true);
    // Frame 1's camera info is a millisecond late.
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame),
        holoscan::testing::input("point_cloud", synthetic_cloud_message(f, true, false))));
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame) + std::chrono::milliseconds(frame == 1 ? 1 : 0),
        holoscan::testing::input("camera_info", camera_info(f))));
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.output_device("color", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth", kCaptures);
  h.capture_snapshot<holoscan::schema::ImageT>("color", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  const auto colors = result.snapshots<holoscan::schema::ImageT>("color");
  VERIFY_MSG(depths.size() == 2 && colors.size() == 2,
             std::to_string(depths.size()) + " depth and " + std::to_string(colors.size()) +
                 " color images from 2 matched clouds");
  const std::array<std::uint32_t, 2> matched{0U, 2U};
  for (std::size_t i = 0; i < depths.size() && i < colors.size() && i < matched.size(); ++i) {
    check_truth(depths[i], &colors[i], synthetic_frame(kWidth, kHeight, matched[i], true), config);
    check_provenance(depths[i], matched[i], 0U);
  }
}

// With camera info at its own rate, a cloud that comes before the first one
// gives no image; the clouds after it all do.
void check_clouds_before_camera_info() {
  const PointCloudToDepthConfig config = synthetic_config(DepthFormat::kFloat32);
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, false);
    if (frame == 1) {
      h.push_frame(holoscan::testing::frame_at(
          capture_time(frame) - 1ms, holoscan::testing::input("camera_info", camera_info(f))));
    }
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame),
        holoscan::testing::input("point_cloud", synthetic_cloud_message(f, false, false))));
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.output_device("color", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth", kCaptures);
  h.capture_snapshot<holoscan::schema::ImageT>("color", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  const auto colors = result.snapshots<holoscan::schema::ImageT>("color");
  VERIFY_MSG(depths.size() == 2 && colors.size() == 2,
             std::to_string(depths.size()) + " images from 2 clouds after the camera info");
  for (std::size_t i = 0; i < depths.size() && i < colors.size(); ++i) {
    const auto number = static_cast<std::uint32_t>(i + 1);
    check_truth(depths[i], &colors[i], synthetic_frame(kWidth, kHeight, number, false), config);
    check_provenance(depths[i], number, 0U);
  }
}

// The image has the camera's size, below the configured largest.
void check_camera_sets_image_size() {
  PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  config.max_width = 2 * kWidth;
  config.max_height = 2 * kHeight;
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 0, false);
  const std::vector<float> xyz = organized_cloud(f, false);
  h.push_frame(holoscan::testing::frame_at(
      capture_time(0), holoscan::testing::input("camera_info", camera_info(f)),
      holoscan::testing::input("point_cloud", cloud_message(points_tensor(xyz, false)))));
  h.output_device("depth", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto images = result.snapshots<holoscan::schema::ImageT>("depth");
  VERIFY(images.size() == 1);
  if (images.size() != 1) return;
  // image_data() checks the descriptor's and the tensor's size.
  const std::vector<float> meters =
      image_data<float>(images[0], holoscan::schema::ImageEncoding_MONO32F, 1);
  VERIFY(std::count(meters.begin(), meters.end(), 0.0F) < kWidth * kHeight / 2);
}

// A cloud with no points gives an empty image.
void check_empty_cloud() {
  PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  config.color = true;
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  h.push_frame(holoscan::testing::frame_at(
      100ms,
      holoscan::testing::input("camera_info",
                               camera_info(synthetic_frame(kWidth, kHeight, 0, false))),
      holoscan::testing::input("point_cloud",
                               cloud_message(points_tensor({}, false), colors_tensor({}, false)))));
  h.output_device("depth", holoscan::DeviceId{0});
  h.output_device("color", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth", kCaptures);
  h.capture_snapshot<holoscan::schema::ImageT>("color", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  const auto colors = result.snapshots<holoscan::schema::ImageT>("color");
  VERIFY(depths.size() == 1 && colors.size() == 1);
  if (depths.size() != 1 || colors.size() != 1) return;
  for (const float z : image_data<float>(depths[0], holoscan::schema::ImageEncoding_MONO32F, 1)) {
    VERIFY(z == 0.0F);
  }
  for (const std::uint8_t c :
       image_data<std::uint8_t>(colors[0], holoscan::schema::ImageEncoding_RGB8, 3)) {
    VERIFY(c == 0U);
  }
}

// Host input is held until the stream that copies it is done: by its allocator
// when that can wait for a stream, without blocking; otherwise by waiting.
void check_host_input_held() {
  cudaStream_t stream{};
  VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  {
    auto memory = std::make_shared<TestMemory>(64, false, true);
    VERIFY(stall(stream));
    VERIFY(
        detail::hold_until_done(wrap(memory, 0, {64}, kUInt8, false), stream, "test").has_value());
    VERIFY_MSG(memory->streams() == 1, "the allocator is asked to wait for the stream");
    VERIFY_MSG(cudaStreamQuery(stream) == cudaErrorNotReady, "the host does not wait");
    VERIFY(cudaStreamSynchronize(stream) == cudaSuccess);
  }
  {
    auto memory = std::make_shared<TestMemory>(64, false, false);
    VERIFY(stall(stream));
    VERIFY(
        detail::hold_until_done(wrap(memory, 0, {64}, kUInt8, false), stream, "test").has_value());
    VERIFY_MSG(cudaStreamQuery(stream) == cudaSuccess,
               "the stream is done when the allocator cannot wait for it");
  }
  VERIFY(cudaStreamDestroy(stream) == cudaSuccess);
}

// A producer of clouds from a runtime pool in host, pinned or device memory,
// as the harness, which stages its inputs in memory of its own, cannot be:
// one cloud of the given points per tick, each with the synthetic camera's
// info. (A cloud message's tensors always come from a runtime pool.)
struct PoolCloudsConfig {
  std::vector<std::vector<float>> clouds;  // x, y, z per point.
  holoscan::MemoryKind memory_kind{holoscan::MemoryKind::kCudaDevice};
};

class PoolCloudsSourceOp final : public holoscan::Operator<> {
 public:
  explicit PoolCloudsSourceOp(PoolCloudsConfig config) : config_(std::move(config)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    std::uint64_t most = 1;
    for (const auto& cloud : config_.clouds) most = std::max<std::uint64_t>(most, cloud.size());
    const auto pool = [&](DLDataType dtype, std::uint64_t bytes) {
      return holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = config_.memory_kind,
                  .dtype = dtype,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly},
          .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = bytes},
          .storage = holoscan::TensorOutputStorage::kRuntimePool};
    };
    spec.output(point_cloud, "point_cloud")
        .tensor_field(hm::PointCloudFields::points, pool(kFloat32, most * sizeof(float)))
        .tensor_field(hm::PointCloudFields::colors, pool(kUInt8, 1U))
        .tensor_field(hm::PointCloudFields::normals, pool(kFloat32, 1U))
        .tensor_field(hm::PointCloudFields::intensity, pool(kFloat32, 1U));
    spec.output(camera_info, "camera_info");
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(5)});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_ >= config_.clouds.size()) {
      return {};
    }
    // A cloud keeps its capture time until both its parts are out.
    if (!capture_) {
      capture_ = context.activation_time();
    }
    const holoscan::EmitOptions options{.capture_time = *capture_, .frame_id = next_ + 1U};
    if (!sent_camera_info_) {
      hm::CameraInfo value =
          synthetic_camera_info(synthetic_frame(kWidth, kHeight, 0, false), "camera_optical");
      value.header.capture_timestamp_ns = capture_->timestamp_ns;
      if (auto sent = camera_info.emit(hm::to_message(value), options); !sent) {
        return sent.error().code == holoscan::ErrorCode::kBackpressured
                   ? holoscan::expected<void, holoscan::Error>{}
                   : sent;
      }
      sent_camera_info_ = true;
    }
    const std::vector<float>& xyz = config_.clouds[next_];
    const hm::PointCloudShape shape =
        hm::unorganized_cloud(static_cast<std::int64_t>(xyz.size() / 3));
    auto loan = hm::allocate_point_cloud(point_cloud, shape, {}, context.cuda_stream());
    if (!loan) {
      return loan.error().code == holoscan::ErrorCode::kResourceExhausted
                 ? holoscan::expected<void, holoscan::Error>{}
                 : holoscan::make_unexpected(std::move(loan).error());
    }
    const std::size_t bytes = xyz.size() * sizeof(float);
    auto written =
        config_.memory_kind == holoscan::MemoryKind::kCudaDevice
            ? loan->write_and_commit(
                  hm::PointCloudFields::points, context.cuda_stream(),
                  [&](const holoscan::TensorOutputWriteGuard& writer) {
                    return cudaMemcpyAsync(writer.data_as<void>(), xyz.data(), bytes,
                                           cudaMemcpyHostToDevice,
                                           context.cuda_stream().get()) == cudaSuccess
                               ? holoscan::expected<void, holoscan::Error>{}
                               : holoscan::make_unexpected(holoscan::Error{
                                     holoscan::ErrorCode::kFailure, "cudaMemcpyAsync failed"});
                  })
            : loan->write_host_and_commit(hm::PointCloudFields::points,
                                          [&](const holoscan::TensorOutputWriteGuard& writer) {
                                            std::memcpy(writer.data_as<void>(), xyz.data(), bytes);
                                          });
    if (!written) {
      return written;
    }
    if (auto sent = hm::emit_point_cloud(point_cloud, std::move(*loan), {.frame_id = "body"}, shape,
                                         -1, options);
        !sent) {
      return sent;
    }
    ++next_;
    capture_.reset();
    sent_camera_info_ = false;
    return {};
  }

  holoscan::Output<hm::schema::PointCloudT> point_cloud;
  holoscan::Output<hm::schema::CameraInfoT> camera_info;

 private:
  PoolCloudsConfig config_;
  std::uint64_t next_{};
  std::optional<holoscan::TimePoint> capture_;
  bool sent_camera_info_{};
};

// What the sink below saw.
struct DepthCount {
  std::uint32_t frames{};
  std::uint64_t filled_pixels{};
};

// Counts the depth images (with their color images) it receives and their
// filled pixels, and completes the run after `expected` of them.
class DepthCountOp final : public holoscan::Operator<> {
 public:
  DepthCountOp(std::shared_ptr<DepthCount> count, std::uint32_t expected, DepthFormat format,
               bool color)
      : count_(std::move(count)), expected_(expected), format_(format), color_(color) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(depth, "depth").queue_depth(2U);
    if (color_) {
      spec.input(color, "color").queue_depth(2U);
    }
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    if (!color_) {
      result.trigger(holoscan::OnEach{depth});
      return result;
    }
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::PortList{std::vector<holoscan::PortRef>{depth, color}},
        .window = std::chrono::nanoseconds{0}});
    result.input(depth, holoscan::Exact{});
    result.input(color, holoscan::Exact{});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto image = depth.receive();
    if (!image) {
      return holoscan::make_unexpected(std::move(image).error());
    }
    if (color_) {
      if (auto rgb = color.receive(); !rgb) {
        return holoscan::make_unexpected(std::move(rgb).error());
      }
    }
    if (!image->data.data) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "DepthCountOp: image has no tensor"});
    }
    const holoscan::Tensor& tensor = *image->data.data;
    const auto pixels = static_cast<std::size_t>(image->data.width) * image->data.height;
    if (format_ == DepthFormat::kUint16) {
      std::vector<std::uint16_t> counts(pixels);
      if (auto copied = tensor.copy_to_host(counts.data(), counts.size() * sizeof(std::uint16_t),
                                            context.cuda_stream());
          !copied) {
        return copied;
      }
      count_->filled_pixels += pixels - std::count(counts.begin(), counts.end(), 0U);
    } else {
      std::vector<float> meters(pixels);
      if (auto copied = tensor.copy_to_host(meters.data(), meters.size() * sizeof(float),
                                            context.cuda_stream());
          !copied) {
        return copied;
      }
      count_->filled_pixels += pixels - std::count(meters.begin(), meters.end(), 0.0F);
    }
    if (++count_->frames == expected_) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> depth;
  holoscan::Input<holoscan::schema::ImageT> color;

 private:
  std::shared_ptr<DepthCount> count_;
  std::uint32_t expected_{};
  DepthFormat format_{};
  bool color_{};
};

// Runs `clouds` through the operator into a counting sink. Returns what the
// sink saw, and in `failure` the run's diagnostics when it did not complete.
std::shared_ptr<DepthCount> run_graph(const PointCloudToDepthConfig& config,
                                      PoolCloudsConfig clouds, std::string& failure) {
  holoscan::Graph graph{"point_cloud_to_depth_test"};
  const auto count = static_cast<std::uint32_t>(clouds.clouds.size());
  const bool device_source = clouds.memory_kind == holoscan::MemoryKind::kCudaDevice;
  auto source = graph.op<PoolCloudsSourceOp>("source", std::move(clouds));
  auto project = graph.op<PointCloudToDepthOp>("project", config);
  auto report = std::make_shared<DepthCount>();
  auto sink = graph.op<DepthCountOp>("count", report, count, config.depth_format, config.color);
  graph.add_flow(source->point_cloud, project->point_cloud);
  graph.add_flow(source->camera_info, project->camera_info);
  if (config.color) {
    graph.add_flow(project->color, sink->color);
  }
  graph.add_flow(project->depth, sink->depth);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  for (const char* port : {"depth", "color"}) {
    if (std::string(port) == "depth" || config.color) {
      options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = "project", .output_port = port, .device = holoscan::DeviceId{0}});
    }
  }
  if (device_source) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "source", .output_port = "point_cloud", .device = holoscan::DeviceId{0}});
  }
  const auto plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    failure = holoscan::format_diagnostics(plan.diagnostics(), plan.graph());
    return report;
  }
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (session.is_running()) {
    session.request_stop();
  }
  session.wait();
  if (session.termination() != holoscan::RunTermination::kRequestedCompleted) {
    failure = "run did not complete: " +
              holoscan::format_diagnostics(session.diagnostics(), plan.graph());
  }
  return report;
}

// Clouds from a producer's host, pinned or device pool: two organized clouds
// of frame 0, sent as unorganized ones with the camera info of every cloud,
// each giving the image back. Host memory is staged on the GPU; the pinned
// memory is held until its copy is done.
void check_clouds_from_pool(holoscan::MemoryKind memory_kind) {
  PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  config.camera_info.timing = hm::CameraInfoTiming::kEveryFrame;
  const std::vector<float> xyz = organized_cloud(synthetic_frame(kWidth, kHeight, 0, false), false);
  std::string failure;
  const auto report = run_graph(
      config, PoolCloudsConfig{.clouds = {xyz, xyz}, .memory_kind = memory_kind}, failure);
  VERIFY_MSG(failure.empty(), failure);
  int expected = 0;
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 0, false);
  for (int v = 0; v < kHeight; ++v) {
    for (int u = 0; u < kWidth; ++u) {
      const float z = synthetic_depth(f, u, v);
      expected += z >= config.depth_min && z <= config.depth_max;
    }
  }
  VERIFY_MSG(report->frames == 2 && report->filled_pixels == 2U * expected,
             std::to_string(report->frames) + " images with " +
                 std::to_string(report->filled_pixels) + " filled pixels, expected 2 with " +
                 std::to_string(2 * expected));
}

// The images' descriptors carry the cloud's capture time and the camera info's
// frame in their header.
void check_image_headers() {
  PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  config.color = true;
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, false);
    const std::vector<float> xyz = organized_cloud(f, false);
    const std::vector<std::uint8_t> rgb(xyz.size(), 9U);
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame), holoscan::testing::input("camera_info", camera_info(f)),
        holoscan::testing::input(
            "point_cloud",
            cloud_message(
                make({kHeight, kWidth, 3}, kFloat32, xyz.data(), xyz.size() * sizeof(float), false),
                make({kHeight, kWidth, 3}, kUInt8, rgb.data(), rgb.size(), false)))));
  }
  h.output_device("depth", holoscan::DeviceId{0});
  h.output_device("color", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("depth", kCaptures);
  h.capture_snapshot<holoscan::schema::ImageT>("color", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  for (const char* port : {"depth", "color"}) {
    const auto images = result.snapshots<holoscan::schema::ImageT>(port);
    VERIFY(images.size() == 2);
    for (std::size_t frame = 0; frame < images.size(); ++frame) {
      const holoscan::schema::ImageT descriptor = images[frame].metadata();
      VERIFY_MSG(descriptor.header != nullptr, std::string(port) + " image has a header");
      if (descriptor.header == nullptr) continue;
      VERIFY_MSG(descriptor.header->capture_timestamp_ns ==
                     capture_time(static_cast<std::uint32_t>(frame)).count(),
                 std::string(port) + " header carries capture time " +
                     std::to_string(descriptor.header->capture_timestamp_ns));
      VERIFY(descriptor.header->frame_id == "camera_optical");
    }
  }
}

// Runs the operator on one cloud and checks that its failure names `what`.
void check_rejects(const PointCloudToDepthConfig& config, hm::schema::PointCloudT cloud,
                   const hm::CameraInfo& info, const std::string& what) {
  auto h = holoscan::testing::create_operator_test<PointCloudToDepthOp>(config);
  h.push_frame(holoscan::testing::frame_at(
      100ms, holoscan::testing::input("camera_info", hm::to_message(info)),
      holoscan::testing::input("point_cloud", std::move(cloud))));
  h.output_device("depth", holoscan::DeviceId{0});
  if (config.color) {
    h.output_device("color", holoscan::DeviceId{0});
  }
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

void check_rejected_inputs() {
  const PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  const hm::CameraInfo info =
      synthetic_camera_info(synthetic_frame(kWidth, kHeight, 0, false), "camera_optical");
  const std::vector<float> xyz(12, 1.0F);
  const std::vector<double> xyz64(12, 1.0);
  // Clouds the message's layout refuses.
  check_rejects(config, cloud_message(make({4, 3}, {kDLFloat, 64, 1}, xyz64.data(), 96, false)),
                info, "PointCloudToDepthOp: point_cloud.points");
  check_rejects(config, cloud_message(make({3, 4}, kFloat32, xyz.data(), 48, false)), info,
                "PointCloudToDepthOp: point_cloud.points");
  const std::vector<std::uint8_t> rgb(12, 7U);
  check_rejects(
      config, cloud_message(points_tensor(xyz, false), make({3, 3}, kUInt8, rgb.data(), 9, false)),
      info, "PointCloudToDepthOp: point_cloud.colors");
  // A cloud above the configured largest.
  PointCloudToDepthConfig small = config;
  small.max_points = 3;
  check_rejects(small, cloud_message(points_tensor(xyz, false)), info,
                "exceed the configured maximum of 3");
  // A color image asked of a cloud without colors.
  PointCloudToDepthConfig with_color = config;
  with_color.color = true;
  check_rejects(with_color, cloud_message(points_tensor(xyz, false)), info,
                "the cloud has no colors");
  // A camera larger than the outputs are sized for.
  hm::CameraInfo large = info;
  large.width = kWidth + 1;
  check_rejects(config, cloud_message(points_tensor(xyz, false)), large,
                "camera info 'camera_optical' is for 97x72 images, larger than the configured "
                "96x72");
  hm::CameraInfo tall = info;
  tall.height = kHeight + 1;
  check_rejects(config, cloud_message(points_tensor(xyz, false)), tall,
                "camera info 'camera_optical' is for 96x73 images, larger than the configured "
                "96x72");
}

// Cameras the kernels cannot model fail the frame, naming what is wrong.
void check_rejected_cameras() {
  const PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
  const std::vector<float> xyz(12, 1.0F);
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 0, false);
  hm::CameraInfo raw = synthetic_camera_info(f, "camera_optical");
  raw.state = hm::CameraState::kRaw;
  raw.distortion_model = hm::DistortionModel::kRationalPolynomial;
  raw.coefficients = {0.1, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  check_rejects(config, cloud_message(points_tensor(xyz, false)), raw,
                "camera info 'camera_optical' describes raw images, with rational_polynomial");
  hm::CameraInfo skewed = synthetic_camera_info(f, "camera_optical");
  skewed.skew = -1.0;
  check_rejects(config, cloud_message(points_tensor(xyz, false)), skewed,
                "camera info 'camera_optical' has a skew");
}

void check_rejected_config() {
  const auto rejects = [](const auto& change, const std::string& what) {
    PointCloudToDepthConfig config = base_config(DepthFormat::kFloat32);
    change(config);
    try {
      PointCloudToDepthOp op(config);
      VERIFY_MSG(false, what + " is rejected");
    } catch (const std::invalid_argument& error) {
      VERIFY_MSG(std::string(error.what()).find(what) != std::string::npos,
                 "'" + std::string(error.what()) + "' names '" + what + "'");
    }
  };
  rejects([](auto& c) { c.max_width = 0; }, "max_width and max_height must be 1..16384");
  rejects([](auto& c) { c.max_height = 16385; }, "max_width and max_height must be 1..16384");
  rejects(
      [](auto& c) {
        c.extrinsics = Extrinsics{};
        c.extrinsics->matrix[5] = std::numeric_limits<float>::infinity();
      },
      "extrinsics must be finite");
  rejects([](auto& c) { c.depth_max = c.depth_min; }, "depth range");
  rejects([](auto& c) { c.depth_min = -1.0F; }, "depth range");
  rejects([](auto& c) { c.depth_scale = 0.0F; }, "depth_scale");
  rejects([](auto& c) { c.depth_format = static_cast<DepthFormat>(7); }, "depth_format");
  rejects([](auto& c) { c.point_size = 2; }, "point_size must be 1, 3 or 5");
  rejects([](auto& c) { c.point_size = 7; }, "point_size must be 1, 3 or 5");
  rejects([](auto& c) { c.max_points = 0; }, "max_points");
  rejects([](auto& c) { c.max_points = kMaxPointsLimit + 1U; }, "max_points");
  rejects([](auto& c) { c.camera_info.max_age = std::chrono::nanoseconds::zero(); },
          "camera_info.max_age");
}

}  // namespace
}  // namespace holoscan::point_cloud

int main() {
  namespace pcd = holoscan::point_cloud;
  using pcd::DepthFormat;
  return pcd::test::run({
      {"organized float32 cloud from host memory gives its image back",
       [] { pcd::check_organized_round_trip(false, false); }},
      {"organized uint16 cloud from device memory gives its image back",
       [] { pcd::check_organized_round_trip(true, true); }},
      {"synthetic clouds with colors, float32, host",
       [] { pcd::check_synthetic_clouds(DepthFormat::kFloat32, 1, false); }},
      {"synthetic clouds with colors, uint16, device, point size 3",
       [] { pcd::check_synthetic_clouds(DepthFormat::kUint16, 3, true); }},
      {"synthetic clouds with colors, float32, device, point size 5",
       [] { pcd::check_synthetic_clouds(DepthFormat::kFloat32, 5, true); }},
      {"frame ids reach the images", pcd::check_frame_ids},
      {"a changing camera with every cloud, host", [] { pcd::check_changing_camera(false); }},
      {"a changing camera with every cloud, device", [] { pcd::check_changing_camera(true); }},
      {"camera info with another capture time gives no image", pcd::check_unmatched_camera_info},
      {"a camera that changes at its own rate is picked up", pcd::check_camera_change_at_own_rate},
      {"a camera that changes to an unusable one fails the cloud",
       pcd::check_camera_change_to_unusable},
      {"clouds before the first camera info give no image", pcd::check_clouds_before_camera_info},
      {"the image has the camera's size", pcd::check_camera_sets_image_size},
      {"a cloud without points gives an empty image", pcd::check_empty_cloud},
      {"host input is held until its copy is done", pcd::check_host_input_held},
      {"clouds from a host pool", [] { pcd::check_clouds_from_pool(holoscan::MemoryKind::kHost); }},
      {"clouds from a pinned pool",
       [] { pcd::check_clouds_from_pool(holoscan::MemoryKind::kPinnedHost); }},
      {"clouds from a device pool",
       [] { pcd::check_clouds_from_pool(holoscan::MemoryKind::kCudaDevice); }},
      {"image headers carry the capture time and the camera's frame", pcd::check_image_headers},
      {"unusable clouds are rejected", pcd::check_rejected_inputs},
      {"raw and skewed cameras are rejected", pcd::check_rejected_cameras},
      {"unusable configurations are rejected", pcd::check_rejected_config},
  });
}
