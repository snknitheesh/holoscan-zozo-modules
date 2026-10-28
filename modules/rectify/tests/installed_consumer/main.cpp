// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-rectify package: a source
// sends a distorted synthetic stereo pair from host memory, StereoRectifyOp
// rectifies it on the GPU, and a sink checks, with the geometry message, that
// every dot lies on the same row in both rectified images.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
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
#include <holoscan/rectify/stereo_rectify_op.hpp>
#include <holoscan/rectify/synthetic_stereo.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using std::chrono_literals::operator""ms;
using std::chrono_literals::operator""ns;
namespace rc = holoscan::rectify;

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
  bool geometry_ok{true};
};

// Checks each rectified pair against the geometry it came with.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(left, "left").queue_depth(2U);
    spec.input(right, "right").queue_depth(2U);
    spec.input(geometry, "geometry").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    const std::vector<holoscan::PortRef> ports{left, right, geometry};
    result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
    result.input(left, holoscan::Exact{});
    result.input(right, holoscan::Exact{});
    result.input(geometry, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto l = left.receive();
    if (!l) return holoscan::make_unexpected(std::move(l).error());
    auto r = right.receive();
    if (!r) return holoscan::make_unexpected(std::move(r).error());
    auto g = geometry.receive();
    if (!g) return holoscan::make_unexpected(std::move(g).error());
    const rc::StereoRectification truth = rc::compute_stereo_rectification(scene().calibration);
    tally_->geometry_ok = tally_->geometry_ok && g->data.fx == truth.fx &&
                          g->data.baseline == truth.baseline && g->data.width == kWidth &&
                          g->data.backend == "cuda";
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
  holoscan::Input<rc::schema::StereoGeometryT> geometry;

 private:
  std::shared_ptr<Tally> tally_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"rectify_consumer"};
  auto source = graph.op<PairSourceOp>("source");
  auto rectify = graph.op<rc::StereoRectifyOp>(
      "rectify",
      rc::StereoRectifyConfig{.calibration = scene().calibration, .geometry_output = true});
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->left, rectify->left, lossless);
  graph.add_flow(source->right, rectify->right, lossless);
  graph.add_flow(rectify->rectified_left, check->left, lossless);
  graph.add_flow(rectify->rectified_right, check->right, lossless);
  graph.add_flow(rectify->geometry, check->geometry, lossless);
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
                      tally->max_row_error < 0.1 && tally->geometry_ok;
  std::cout << "rectify_consumer: frames=" << tally->frames << " dots=" << tally->dots
            << " missing=" << tally->missing << " max_row_error_px=" << tally->max_row_error
            << " geometry=" << (tally->geometry_ok ? "ok" : "wrong") << '\n'
            << "rectify_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
