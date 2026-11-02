// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-color-convert package: a
// source sends the synthetic camera's pattern as NV12 in host memory,
// ColorConvertOp turns it into RGB8 on the GPU, and ResizeOp letterboxes that
// into a 96x96 I420 image. A sink checks every byte against the module's CPU
// reference.

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

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
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace cc = holoscan::color_convert;

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
      cc::ConversionParams{.letterbox = true, .fill_rgb = {40, 40, 40}});
  return i420;
}

// kFrames NV12 frames in host memory, then idle.
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
    if (emitted_ >= kFrames) {
      return {};
    }
    const std::vector<std::uint8_t> bytes = nv12_frame(emitted_);
    const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(bytes.size())};
    auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
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
    return emitted.error().code == holoscan::ErrorCode::kBackpressured
               ? holoscan::expected<void, holoscan::Error>{}
               : emitted;
  }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  std::uint32_t emitted_{};
};

struct Tally {
  std::uint32_t frames{};
  std::uint64_t wrong_bytes{};
};

// Checks each I420 image against the reference, then ends the run.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(image, "image").queue_depth(2U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{image});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = image.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
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

 private:
  std::shared_ptr<Tally> tally_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"color_convert_consumer"};
  auto source = graph.op<Nv12SourceOp>("source");
  auto convert = graph.op<cc::ColorConvertOp>(
      "convert",
      cc::ColorConvertConfig{
          .output_format = cc::PixelFormat::kRgb8, .max_width = kWidth, .max_height = kHeight});
  auto resize =
      graph.op<cc::ResizeOp>("resize", cc::ResizeConfig{.width = kOut,
                                                        .height = kOut,
                                                        .letterbox = true,
                                                        .fill_rgb = {40, 40, 40},
                                                        .output_format = cc::PixelFormat::kI420});
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->image, convert->image, lossless);
  graph.add_flow(convert->converted, resize->image, lossless);
  graph.add_flow(resize->resized, check->image, lossless);
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
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "color_convert_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed = tally->frames == kFrames && tally->wrong_bytes == 0;
  std::cout << "color_convert_consumer: frames=" << tally->frames
            << " wrong_bytes=" << tally->wrong_bytes << '\n'
            << "color_convert_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
