// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// SegmentationReportOp: takes one image's instances from Sam3SegmentationOp
// and adds them to a report: their number, the time since the image's
// capture, and frames missed or skipped. It can check them: with the stub
// model against the host reference of the whole pipeline, from the image the
// segmenter saw; with the real model on the synthetic scene, against its
// true shapes (each prompt's best instance must cover its shape with an IoU
// of at least min_iou). And it can print each image file's instances, and
// write each image with its instances drawn over it as a PPM file.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sam3/sam3_segmentation_op.hpp>
#include <holoscan/sam3/stub_model.hpp>
#include <holoscan/sam3/synthetic_scene.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace sam3_app {

struct SegmentationReport {
  std::atomic<std::uint32_t> frames{};
  // Frames whose id is not the one after the previous frame's; in real time,
  // only those whose id is not above it.
  std::uint32_t frame_gaps{};
  // In real time: the frames passed over for newer ones, and the last id seen.
  std::uint64_t skipped{};
  std::uint64_t last_frame_id{};
  // Instances over the run.
  std::uint64_t instances{};
  // The stub's reference: frames checked, frames that differ from it, and the
  // most pixels a frame's masks differed by where the reference is not on a
  // mask's edge.
  std::uint32_t reference_checked{};
  std::uint32_t reference_failed{};
  std::uint64_t worst_reference_pixels{};
  // The synthetic scene's truth: frames checked, frames where a prompt found
  // nothing or too little of its shape, and the IoUs of the prompts' best
  // instances.
  std::uint32_t truth_checked{};
  std::uint32_t truth_failed{};
  double min_iou{1.0};
  double sum_iou{};
  std::uint32_t ious{};
  std::uint32_t files_written{};
  // Capture to instances on the host, per frame, in milliseconds.
  std::vector<double> latencies_ms;

  [[nodiscard]] bool passed() const noexcept {
    return frame_gaps == 0 && reference_failed == 0 && truth_failed == 0;
  }
};

enum class Verify : std::uint8_t { kNone, kReference, kTruth };

struct SegmentationReportConfig {
  // Where the reporter adds what it sees; one report per run.
  std::shared_ptr<SegmentationReport> report;
  // Frames after which the run completes; 0 never completes. In real time,
  // the run completes at the frame with this id.
  std::uint32_t expected_frames{};
  // The inputs keep only the newest sample (see edges.hpp): frames come with
  // ids that go up, but not one by one, as those that came while the reporter
  // was busy are passed over and counted as skipped.
  bool realtime{false};
  // Time spent on each frame on top of the work: a stand-in, in tests, for a
  // consumer slower than the source.
  std::chrono::nanoseconds delay{};
  Verify verify{Verify::kNone};
  // The segmenter's settings, which the checks follow.
  holoscan::sam3::Sam3SegmentationConfig segmenter;
  // kReference: the stub's shape, and how near the threshold a reference
  // logit must be for a pixel to go either way.
  holoscan::sam3::StubShape stub;
  float logit_slack{1e-3F};
  // kTruth: the synthetic scene's size, whether it is still, the shape each
  // prompt should find (none: not checked), and the smallest IoU accepted.
  int scene_width{};
  int scene_height{};
  bool still{false};
  std::vector<std::optional<holoscan::sam3::SceneShape>> expected_shapes;
  double min_iou{0.9};
  // The images of a file source, frame f being files[(f - 1) % size]: named
  // when printing.
  std::vector<std::filesystem::path> files;
  // Print each image's instances.
  bool print{false};
  // Write each image with its instances over it as
  // `image_dir`/segment_NNNNNN.ppm, numbered by frame id.
  std::filesystem::path image_dir;
};

class SegmentationReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if the report is missing or the settings
  // unusable.
  explicit SegmentationReportOp(SegmentationReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // Sam3SegmentationOp's outputs.
  holoscan::Input<holoscan::Tensor> masks;
  holoscan::Input<holoscan::Tensor> labels;
  holoscan::Input<holoscan::Tensor> scores;
  holoscan::Input<holoscan::Tensor> boxes;
  holoscan::Input<holoscan::Tensor> prompt_ids;
  holoscan::Input<holoscan::Tensor> count;
  // The image the segmenter saw; declared for kReference and `image_dir`.
  holoscan::Input<holoscan::schema::ImageT> image;

 private:
  [[nodiscard]] bool needs_image() const noexcept;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> process(
      holoscan::ExecutionContext& context);

  SegmentationReportConfig config_;
  std::uint64_t last_frame_id_{};
  std::uint32_t seen_{};
  std::vector<std::uint8_t> masks_;
  std::vector<std::uint16_t> labels_;
  std::vector<float> scores_;
  std::vector<float> boxes_;
  std::vector<std::int32_t> prompt_ids_;
  std::vector<std::uint8_t> pixels_;
};

// Connects every output of the segmenter to the reporter with `edge`.
inline void connect_instances(
    holoscan::Graph& graph,
    const holoscan::OperatorHandle<holoscan::sam3::Sam3SegmentationOp>& segmenter,
    const holoscan::OperatorHandle<SegmentationReportOp>& reporter,
    const holoscan::ConnectionOptions& edge) {
  graph.add_flow(segmenter->masks, reporter->masks, edge);
  graph.add_flow(segmenter->labels, reporter->labels, edge);
  graph.add_flow(segmenter->scores, reporter->scores, edge);
  graph.add_flow(segmenter->boxes, reporter->boxes, edge);
  graph.add_flow(segmenter->prompt_ids, reporter->prompt_ids, edge);
  graph.add_flow(segmenter->count, reporter->count, edge);
}

}  // namespace sam3_app
