// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-rectify package: the
// synthetic pair's calibration is written as ROS camera_info files, a source
// sends the distorted pair from host memory, StereoRectifyOp rectifies it on
// the GPU from the files, and a sink takes each rectified pair with its camera
// info, as a disparity operator would, and checks that every dot lies on the
// same row in both images.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/rectify/stereo_rectify_op.hpp>
#include <holoscan/rectify/synthetic_stereo.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using std::chrono_literals::operator""ms;
namespace rc = holoscan::rectify;
namespace hm = holoscan::messages;
namespace fs = std::filesystem;

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr std::uint32_t kFrames = 3;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

const rc::SyntheticStereo& scene() {
  static const rc::SyntheticStereo s = rc::synthetic_stereo(kWidth, kHeight);
  return s;
}

void host_image_output(holoscan::OperatorSpec& spec,
                       holoscan::Output<holoscan::schema::ImageT>& port, const char* name) {
  spec.output(port, name)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kHost,
                  .dtype = kUInt8,
                  .rank = 2U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly},
          .bounds = holoscan::tensor_bounds(std::uint64_t{kWidth} * kHeight),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>()});
}

// kFrames left and right MONO8 images of the synthetic pair, then idle.
class PairSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    host_image_output(spec, left, "left");
    host_image_output(spec, right, "right");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 20ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_ >= kFrames) return {};
    const holoscan::EmitOptions emit{.capture_time = context.activation_time(),
                                     .frame_id = emitted_ + 1U};
    for (const bool is_right : {false, true}) {
      const std::vector<std::uint8_t> pixels =
          (is_right ? right_view_ : left_view_).render(emitted_);
      auto& port = is_right ? right : left;
      auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{
          .shape = std::array<std::int64_t, 2>{kHeight, kWidth}, .dtype = kUInt8});
      if (!loan) return holoscan::make_unexpected(std::move(loan).error());
      auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
        std::memcpy(writer.data(), pixels.data(), pixels.size());
        return {};
      });
      if (!written) return holoscan::make_unexpected(std::move(written).error());
      holoscan::schema::ImageT descriptor;
      descriptor.width = kWidth;
      descriptor.height = kHeight;
      descriptor.encoding = holoscan::schema::ImageEncoding_MONO8;
      if (auto sent = port.emit_tensor(std::move(*loan), descriptor, emit); !sent) {
        return sent;
      }
    }
    ++emitted_;
    return {};
  }

  holoscan::Output<holoscan::schema::ImageT> left;
  holoscan::Output<holoscan::schema::ImageT> right;

 private:
  rc::SyntheticView left_view_ = rc::left_view(scene());
  rc::SyntheticView right_view_ = rc::right_view(scene());
  std::uint32_t emitted_{};
};

struct Tally {
  std::uint32_t frames{};
  int dots{};
  int missing{};
  double max_row_error{};
  bool camera_info_ok{true};
};

// Checks each rectified pair against the camera info it came with.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(left, "left").queue_depth(2U);
    spec.input(right, "right").queue_depth(2U);
    spec.input(camera_info, "camera_info").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    return hm::camera_info_contract(
        hm::CameraInfoInputOptions{.timing = hm::CameraInfoTiming::kEveryFrame}, camera_info, left,
        right);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto l = left.receive();
    if (!l) return holoscan::make_unexpected(std::move(l).error());
    auto r = right.receive();
    if (!r) return holoscan::make_unexpected(std::move(r).error());
    auto info = info_.update(camera_info);
    if (!info) return holoscan::make_unexpected(std::move(info).error());
    if (*info == nullptr) return {};  // Not with kEveryFrame: each pair brings its own.
    // The rectified pair: one fx, the baseline along x, the pair's capture time.
    const hm::StereoCameraInfo& pair = **info;
    const rc::StereoRectification truth = rc::compute_stereo_rectification(scene().calibration);
    tally_->camera_info_ok =
        tally_->camera_info_ok && pair.left.state == hm::CameraState::kRectified &&
        std::abs(pair.left.fx - truth.fx) < 1e-9 &&
        std::abs(pair.baseline - truth.baseline) < 1e-12 && pair.left.width == kWidth &&
        pair.header.capture_timestamp_ns == l->metadata.capture_timestamp_ns;
    std::vector<std::uint8_t> lp(static_cast<std::size_t>(kWidth) * kHeight);
    std::vector<std::uint8_t> rp(lp.size());
    if (auto copied = l->data.data->copy_to_host(lp.data(), lp.size(), context.cuda_stream());
        !copied) {
      return copied;
    }
    if (auto copied = r->data.data->copy_to_host(rp.data(), rp.size(), context.cuda_stream());
        !copied) {
      return copied;
    }
    const auto frame = static_cast<std::uint32_t>(l->metadata.frame_id - 1U);
    const rc::AlignmentReport report =
        rc::check_rectified(scene(), truth, lp.data(), rp.data(), kWidth, frame);
    tally_->dots += report.dots;
    tally_->missing += report.missing;
    tally_->max_row_error = std::max(tally_->max_row_error, report.max_row_error);
    if (++tally_->frames == kFrames) context.request_completion();
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> left;
  holoscan::Input<holoscan::schema::ImageT> right;
  holoscan::Input<hm::schema::StereoCameraInfoT> camera_info;

 private:
  std::shared_ptr<Tally> tally_;
  hm::LatestCameraInfo<hm::schema::StereoCameraInfoT> info_;
};

// A new folder for the camera_info files, removed with them at exit.
class CalibrationFolder {
 public:
  CalibrationFolder() {
    std::string pattern = (fs::temp_directory_path() / "rectify-consumer-XXXXXX").string();
    if (mkdtemp(pattern.data()) == nullptr) throw std::runtime_error("mkdtemp failed");
    path_ = pattern;
  }
  ~CalibrationFolder() {
    std::error_code ignored;
    fs::remove_all(path_, ignored);
  }
  CalibrationFolder(const CalibrationFolder&) = delete;
  CalibrationFolder& operator=(const CalibrationFolder&) = delete;
  [[nodiscard]] const fs::path& path() const noexcept { return path_; }

 private:
  fs::path path_;
};

int run() {
  const CalibrationFolder folder;
  const fs::path left_file = folder.path() / "left.yaml";
  const fs::path right_file = folder.path() / "right.yaml";
  hm::write_stereo_camera_info_yaml(scene().calibration, left_file, right_file);

  holoscan::Graph graph{"rectify_consumer"};
  auto source = graph.op<PairSourceOp>("source");
  auto rectify = graph.op<rc::StereoRectifyOp>(
      "rectify", rc::StereoRectifyConfig{.left_camera_info_path = left_file,
                                         .right_camera_info_path = right_file});
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->left, rectify->left, lossless);
  graph.add_flow(source->right, rectify->right, lossless);
  graph.add_flow(rectify->rectified_left, check->left, lossless);
  graph.add_flow(rectify->rectified_right, check->right, lossless);
  graph.add_flow(rectify->camera_info, check->camera_info, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The rectified images are in device memory, on GPU 0.
  holoscan::CompileOptions options;
  for (const char* port : {"rectified_left", "rectified_right"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "rectify", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "rectify_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed = tally->frames == kFrames && tally->dots > 100 && tally->missing == 0 &&
                      tally->max_row_error < 0.1 && tally->camera_info_ok;
  std::cout << "rectify_consumer: frames=" << tally->frames << " dots=" << tally->dots
            << " missing=" << tally->missing << " max_row_error_px=" << tally->max_row_error
            << " camera_info=" << (tally->camera_info_ok ? "ok" : "wrong") << '\n'
            << "rectify_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main() {
  try {
    return run();
  } catch (const std::exception& error) {
    std::cerr << "rectify_consumer: " << error.what() << '\n';
    return 1;
  }
}
