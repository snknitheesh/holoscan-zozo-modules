// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The local recordings' RGB frames and their camera (rgb.yaml, a ROS
// camera_info file) through ResizeOp on CUDA: each 640x640 letterbox, from
// host memory, equals the CPU reference byte for byte, and its camera info is
// the file's camera resized, by the worked formula, with the frame's capture
// time. Reads $HOLOSCAN_TEST_DATA/stereo-camera/<clip>/; skipped (exit 77) when
// the variable is unset, failed when it names no recording with an rgb.yaml.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/color_convert/camera_info.hpp>
#include <holoscan/color_convert/reference.hpp>
#include <holoscan/color_convert/resize_op.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "test_support.hpp"

namespace holoscan::color_convert {
namespace {

namespace fs = std::filesystem;
namespace ht = holoscan::testing;
namespace hm = holoscan::messages;

constexpr int kSkip = 77;
constexpr std::size_t kFrames = 3;
constexpr int kOut = 640;

struct Frame {
  std::int64_t time_ns{};
  int width{};
  int height{};
  std::vector<std::uint8_t> rgb;
};

// An 8-bit binary PPM (P6), as the recordings hold the RGB frames.
Frame read_ppm(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error(path.string() + ": cannot open");
  std::string magic;
  file >> magic;
  int values[3] = {};
  for (int& value : values) {
    file >> std::ws;
    while (file.peek() == '#') {
      std::string comment;
      std::getline(file, comment);
      file >> std::ws;
    }
    file >> value;
  }
  file.get();  // The single whitespace before the pixels.
  if (magic != "P6" || values[2] != 255 || values[0] < 1 || values[1] < 1) {
    throw std::runtime_error(path.string() + ": not an 8-bit binary PPM");
  }
  Frame frame{.width = values[0], .height = values[1]};
  frame.rgb.resize(static_cast<std::size_t>(frame.width) * frame.height * 3U);
  file.read(reinterpret_cast<char*>(frame.rgb.data()),
            static_cast<std::streamsize>(frame.rgb.size()));
  if (!file) throw std::runtime_error(path.string() + ": truncated");
  return frame;
}

// The first frames' RGB images and capture times, from index.csv.
std::vector<Frame> first_frames(const fs::path& clip) {
  std::ifstream index(clip / "index.csv");
  std::string line;
  std::getline(index, line);  // frame,timestamp_ns,device_sequence,left,right,rgb,depth
  std::vector<Frame> frames;
  while (frames.size() < kFrames && std::getline(index, line)) {
    std::vector<std::string> fields;
    std::stringstream row(line);
    std::string field;
    while (std::getline(row, field, ',')) fields.push_back(field);
    if (fields.size() < 6) continue;
    Frame frame = read_ppm(clip / fields[5]);
    frame.time_ns = std::stoll(fields[1]);
    frames.push_back(std::move(frame));
  }
  return frames;
}

schema::ImageT host_image(const Frame& frame) {
  const std::vector<std::int64_t> shape{frame.height, frame.width, 3};
  auto tensor = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8, 1}, std::nullopt,
                                      holoscan::MemoryStorageType::kHost,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) throw std::runtime_error(tensor.error().message);
  auto data = tensor->mutable_data();
  if (!data) throw std::runtime_error(data.error().message);
  std::copy(frame.rgb.begin(), frame.rgb.end(), static_cast<std::uint8_t*>(*data));
  schema::ImageT image;
  image.width = frame.width;
  image.height = frame.height;
  image.encoding = schema::ImageEncoding_RGB8;
  image.data = std::make_shared<holoscan::Tensor>(std::move(*tensor));
  return image;
}

template <typename T>
void add_field(ht::TestFrame& frame, std::string name, T value, std::chrono::nanoseconds time) {
  std::vector<ht::TestSample<T>> samples;
  samples.push_back(ht::sample_at(std::move(value), time));
  frame.fields.push_back({std::move(name), ht::staged_samples<T>(std::move(samples))});
}

void check_clip(const fs::path& clip) {
  const std::string name = clip.filename().string();
  const hm::CameraInfo camera = hm::read_camera_info_yaml(clip / "rgb.yaml");
  const std::vector<Frame> frames = first_frames(clip);
  VERIFY_MSG(frames.size() == kFrames, name + ": " + std::to_string(kFrames) + " frames");
  if (frames.size() != kFrames) return;
  VERIFY_MSG(!camera_info_problem(camera, frames[0].width, frames[0].height),
             name + ": rgb.yaml is for the RGB frames' size");

  const ResizeConfig config{.width = kOut,
                            .height = kOut,
                            .letterbox = true,
                            .fill_rgb = {114, 114, 114},
                            .output_format = PixelFormat::kRgb8,
                            .with_camera_info = true};
  auto h = ht::create_operator_test<ResizeOp>(config);
  for (const Frame& frame : frames) {
    const std::chrono::nanoseconds time{frame.time_ns};
    hm::CameraInfo info = camera;
    info.header.capture_timestamp_ns = frame.time_ns;
    ht::TestFrame test_frame;
    add_field(test_frame, "camera_info", hm::to_message(info), time);
    add_field(test_frame, "image", host_image(frame), time);
    h.push_frame(std::move(test_frame));
  }
  h.output_device("resized", holoscan::DeviceId{0});
  h.capture_snapshot<schema::ImageT>("resized", kFrames);
  h.capture<hm::schema::CameraInfoT>("resized_camera_info", kFrames);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), name + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto images = result.snapshots<schema::ImageT>("resized");
  const auto infos = result.samples<hm::schema::CameraInfoT>("resized_camera_info");
  VERIFY_MSG(images.size() == kFrames && infos.size() == kFrames, name + ": every frame out");

  const ConversionParams params{
      .mode = config.mode, .letterbox = true, .fill_rgb = config.fill_rgb};
  const Mapping m =
      compute_mapping(frames[0].width, frames[0].height, kOut, kOut, PixelFormat::kRgb8, params);
  const double sx = static_cast<double>(m.content.width) / frames[0].width;
  const double sy = static_cast<double>(m.content.height) / frames[0].height;
  for (std::size_t i = 0; i < images.size() && i < infos.size(); ++i) {
    const Frame& frame = frames[i];
    std::vector<std::uint8_t> want(packed_size(PixelFormat::kRgb8, kOut, kOut));
    std::vector<std::uint8_t> in = frame.rgb;
    reference_convert(
        packed_view(in.data(), PixelFormat::kRgb8, frame.width, frame.height,
                    MemoryLocation::kHost),
        packed_view(want.data(), PixelFormat::kRgb8, kOut, kOut, MemoryLocation::kHost), params);
    const auto& tensor = images[i].tensor(schema::ImageTensorFields::data);
    VERIFY_MSG(tensor.bytes().size() == want.size() &&
                   std::memcmp(tensor.bytes().data(), want.data(), want.size()) == 0,
               name + ": frame " + std::to_string(i) + " equals the reference");

    const hm::CameraInfo got = hm::from_message(infos[i].data);
    VERIFY_MSG(infos[i].metadata.capture_timestamp_ns == frame.time_ns &&
                   got.header.capture_timestamp_ns == frame.time_ns,
               name + ": the camera info carries its frame's capture time");
    VERIFY(got.width == static_cast<std::uint32_t>(kOut) &&
           got.height == static_cast<std::uint32_t>(kOut));
    VERIFY(got.distortion_model == camera.distortion_model &&
           got.coefficients == camera.coefficients && got.state == camera.state);
    const double tolerance = 1e-9;
    VERIFY_MSG(std::abs(got.fx - camera.fx * sx) < tolerance &&
                   std::abs(got.fy - camera.fy * sy) < tolerance &&
                   std::abs(got.cx - ((camera.cx + 0.5) * sx - 0.5 + m.content.x)) < tolerance &&
                   std::abs(got.cy - ((camera.cy + 0.5) * sy - 0.5 + m.content.y)) < tolerance,
               name + ": the intrinsics are in the letterbox's pixels");
  }
  std::cout << name << ": " << images.size() << " frames " << frames[0].width << "x"
            << frames[0].height << " -> " << kOut << "x" << kOut << ", fx " << camera.fx << " -> "
            << camera.fx * sx << ", content at y " << m.content.y << '\n';
}

void check_recordings(const fs::path& root) {
  std::vector<fs::path> clips;
  const fs::path folder = root / "stereo-camera";
  if (fs::is_directory(folder)) {
    for (const auto& entry : fs::directory_iterator(folder)) {
      if (fs::is_regular_file(entry.path() / "rgb.yaml")) clips.push_back(entry.path());
    }
  }
  std::sort(clips.begin(), clips.end());
  VERIFY_MSG(!clips.empty(), folder.string() + " holds no recording with an rgb.yaml");
  for (const fs::path& clip : clips) check_clip(clip);
}

}  // namespace
}  // namespace holoscan::color_convert

int main() {
  namespace cc = holoscan::color_convert;
  const char* root = std::getenv("HOLOSCAN_TEST_DATA");
  if (root == nullptr || *root == '\0') {
    std::cout << "HOLOSCAN_TEST_DATA is not set; skipping\n";
    return cc::kSkip;
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    return cc::test::no_cuda_device();
  }
  return cc::test::run({
      {"the recordings' RGB frames and camera, resized", [root] { cc::check_recordings(root); }},
  });
}
