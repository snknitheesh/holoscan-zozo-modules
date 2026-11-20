// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-color-convert package: a
// source sends the synthetic camera's pattern as NV12 in host memory,
// ColorConvertOp turns it into RGB8 on the GPU, and ResizeOp letterboxes that
// into a 96x96 I420 image, taking the camera's info from a camera_info file
// (CameraInfoSourceOp) and sending the resized camera's. A sink checks every
// byte against the module's CPU reference, and every camera info. The graph is
// lossless, so it runs on one worker.

#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <holoscan/color_convert/camera_info.hpp>
#include <holoscan/color_convert/color_convert_op.hpp>
#include <holoscan/color_convert/reference.hpp>
#include <holoscan/color_convert/resize_op.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace cc = holoscan::color_convert;
namespace hm = holoscan::messages;

constexpr int kWidth = 160;
constexpr int kHeight = 90;
constexpr int kOut = 96;
constexpr std::uint32_t kFrames = 5;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// Frame `frame` of the pattern, as NV12.
std::vector<std::uint8_t> nv12_frame(std::uint32_t frame) {
  std::vector<std::uint8_t> rgb(cc::packed_size(cc::PixelFormat::kRgb8, kWidth, kHeight));
  cc::draw_test_pattern(cc::packed_view(rgb.data(), cc::PixelFormat::kRgb8, kWidth, kHeight,
                                        cc::MemoryLocation::kHost),
                        frame);
  std::vector<std::uint8_t> nv12(cc::packed_size(cc::PixelFormat::kNv12, kWidth, kHeight));
  cc::reference_convert(cc::packed_view(rgb.data(), cc::PixelFormat::kRgb8, kWidth, kHeight,
                                        cc::MemoryLocation::kHost),
                        cc::packed_view(nv12.data(), cc::PixelFormat::kNv12, kWidth, kHeight,
                                        cc::MemoryLocation::kHost),
                        {});
  return nv12;
}

constexpr cc::ConversionParams kLetterbox{.letterbox = true, .fill_rgb = {40, 40, 40}};

// The camera of the source's images.
hm::CameraInfo camera() {
  hm::CameraInfo info;
  info.header.frame_id = "camera";
  info.width = kWidth;
  info.height = kHeight;
  info.state = hm::CameraState::kUndistorted;
  info.fx = 120.0;
  info.fy = 120.0;
  info.cx = 79.5;
  info.cy = 44.5;
  return info;
}

// A folder of its own for the camera_info file, removed on every way out.
class TemporaryFolder {
 public:
  TemporaryFolder() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "color_convert_consumer_XXXXXX").string();
    if (::mkdtemp(pattern.data()) == nullptr) {
      throw std::system_error(errno, std::generic_category(), "mkdtemp");
    }
    path_ = pattern;
  }
  ~TemporaryFolder() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  TemporaryFolder(const TemporaryFolder&) = delete;
  TemporaryFolder& operator=(const TemporaryFolder&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

// A refusal a source sends again on its next tick: a full lossless queue
// refuses the emit (kBackpressured), and an output pool whose every slot is
// still held refuses the allocation (kResourceExhausted).
bool backpressure(const holoscan::Error& error) {
  return error.code == holoscan::ErrorCode::kBackpressured ||
         error.code == holoscan::ErrorCode::kResourceExhausted;
}

// What the sink should receive for `frame`.
std::vector<std::uint8_t> expected(std::uint32_t frame) {
  std::vector<std::uint8_t> nv12 = nv12_frame(frame);
  std::vector<std::uint8_t> rgb(cc::packed_size(cc::PixelFormat::kRgb8, kWidth, kHeight));
  const auto rgb_view = cc::packed_view(rgb.data(), cc::PixelFormat::kRgb8, kWidth, kHeight,
                                        cc::MemoryLocation::kHost);
  cc::reference_convert(cc::packed_view(nv12.data(), cc::PixelFormat::kNv12, kWidth, kHeight,
                                        cc::MemoryLocation::kHost),
                        rgb_view, cc::ConversionParams{.mode = cc::ResizeMode::kNearest});
  std::vector<std::uint8_t> i420(cc::packed_size(cc::PixelFormat::kI420, kOut, kOut));
  cc::reference_convert(
      rgb_view,
      cc::packed_view(i420.data(), cc::PixelFormat::kI420, kOut, kOut, cc::MemoryLocation::kHost),
      kLetterbox);
  return i420;
}

// kFrames NV12 frames in host memory, from its second tick on (the camera
// info goes out when the run starts), then idle.
class Nv12SourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(image, "image")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
                    .dtype = kUInt8,
                    .rank = 1U,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                },
            .bounds =
                holoscan::tensor_bounds(cc::packed_size(cc::PixelFormat::kNv12, kWidth, kHeight)),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        })
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{20}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (!started_) {
      started_ = true;
      return {};
    }
    if (emitted_ >= kFrames) {
      return {};
    }
    const std::vector<std::uint8_t> bytes = nv12_frame(emitted_);
    const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(bytes.size())};
    auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      // Every slot still held: send this frame again on the next tick.
      return backpressure(loan.error()) ? holoscan::expected<void, holoscan::Error>{}
                                        : holoscan::make_unexpected(std::move(loan).error());
    }
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      std::memcpy(writer.data(), bytes.data(), bytes.size());
      return {};
    });
    if (!written) {
      return holoscan::make_unexpected(std::move(written).error());
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = kWidth;
    descriptor.height = kHeight;
    descriptor.encoding = holoscan::schema::ImageEncoding_NV12;
    descriptor.plane_layouts = {
        {0U, kWidth, std::uint64_t{kWidth} * kHeight},
        {std::uint64_t{kWidth} * kHeight, kWidth, std::uint64_t{kWidth} * (kHeight / 2)}};
    auto emitted =
        image.emit_tensor(std::move(*loan), descriptor,
                          holoscan::EmitOptions{.capture_time = context.activation_time(),
                                                .frame_id = emitted_ + 1U});
    if (emitted) {
      ++emitted_;
      return {};
    }
    // A busy consumer refuses the frame; send it again on the next tick.
    return backpressure(emitted.error()) ? holoscan::expected<void, holoscan::Error>{} : emitted;
  }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  bool started_{};
  std::uint32_t emitted_{};
};

struct Tally {
  std::uint32_t frames{};
  std::uint64_t wrong_bytes{};
  std::uint32_t wrong_cameras{};
};

// Checks each I420 image against the reference and its camera info against
// the resized camera, then ends the run.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(image, "image").queue_depth(2U);
    spec.input(camera_info, "camera_info").queue_depth(2U);
  }

  // ResizeOp sends each image's camera info with the image's capture time.
  [[nodiscard]] holoscan::Contract contract() const override {
    return hm::camera_info_contract(
        hm::CameraInfoInputOptions{.timing = hm::CameraInfoTiming::kEveryFrame}, camera_info,
        image);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto info = camera_info.receive();
    if (!info) {
      return holoscan::make_unexpected(std::move(info).error());
    }
    auto sample = image.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    hm::CameraInfo want_camera = cc::resized_camera_info(camera(), kWidth, kHeight, kOut, kOut,
                                                         cc::PixelFormat::kI420, kLetterbox);
    want_camera.header.capture_timestamp_ns = sample->metadata.capture_timestamp_ns;
    tally_->wrong_cameras += hm::from_message(info->data) == want_camera ? 0U : 1U;
    const auto frame = static_cast<std::uint32_t>(sample->metadata.frame_id - 1U);
    const std::vector<std::uint8_t> want = expected(frame);
    std::vector<std::uint8_t> got(want.size());
    if (!sample->data.data || sample->data.encoding != holoscan::schema::ImageEncoding_I420 ||
        sample->data.plane_layouts.size() != 3U ||
        static_cast<std::size_t>(sample->data.data->nbytes()) != want.size()) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "expected a packed I420 image"});
    }
    if (auto copied =
            sample->data.data->copy_to_host(got.data(), got.size(), context.cuda_stream());
        !copied) {
      return copied;
    }
    for (std::size_t i = 0; i < want.size(); ++i) {
      tally_->wrong_bytes += got[i] != want[i];
    }
    if (++tally_->frames == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> image;
  holoscan::Input<hm::schema::CameraInfoT> camera_info;

 private:
  std::shared_ptr<Tally> tally_;
};

}  // namespace

int main() try {
  // The run constructs the operators again, reading the file then: it stays
  // until main() returns.
  const TemporaryFolder folder;
  const std::filesystem::path file = folder.path() / "camera_info.yaml";
  hm::write_camera_info_yaml(camera(), file);

  holoscan::Graph graph{"color_convert_consumer"};
  auto source = graph.op<Nv12SourceOp>("source");
  auto camera_info = graph.op<hm::CameraInfoSourceOp>(
      "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = file});
  auto convert = graph.op<cc::ColorConvertOp>(
      "convert",
      cc::ColorConvertConfig{
          .output_format = cc::PixelFormat::kRgb8, .max_width = kWidth, .max_height = kHeight});
  auto resize =
      graph.op<cc::ResizeOp>("resize", cc::ResizeConfig{.width = kOut,
                                                        .height = kOut,
                                                        .letterbox = true,
                                                        .fill_rgb = {40, 40, 40},
                                                        .output_format = cc::PixelFormat::kI420,
                                                        .with_camera_info = true});
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->image, convert->image, lossless);
  graph.add_flow(convert->converted, resize->image, lossless);
  graph.add_flow(camera_info->camera_info, resize->camera_info,
                 hm::latest_camera_info_connection());
  graph.add_flow(resize->resized, check->image, lossless);
  graph.add_flow(resize->resized_camera_info, check->camera_info, lossless);
  // Lossless, so one worker for the whole graph: across workers a producer
  // runs while its consumer's lossless queue is full, and an operator in the
  // middle of the chain fails on the refused emit (the messages module's
  // README).
  graph.partition("color_convert_consumer").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  for (const auto& [op, port] :
       {std::pair{"convert", "converted"}, std::pair{"resize", "resized"}}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  const std::uint32_t workers = plan.scheduler_plan().worker_count;
  std::cout << "color_convert_consumer: workers=" << workers << '\n';
  if (workers != 1U) {
    std::cerr << "color_convert_consumer: the lossless graph must run on one worker\n";
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "color_convert_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed =
      tally->frames == kFrames && tally->wrong_bytes == 0 && tally->wrong_cameras == 0;
  std::cout << "color_convert_consumer: frames=" << tally->frames
            << " wrong_bytes=" << tally->wrong_bytes << " wrong_cameras=" << tally->wrong_cameras
            << '\n'
            << "color_convert_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
} catch (const std::exception& error) {
  std::cerr << "color_convert_consumer: " << error.what() << '\n';
  return 1;
}
