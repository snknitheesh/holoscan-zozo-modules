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

#include <cuda_runtime_api.h>

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
  // In real time: the frames passed over for newer ones, and the last id seen,
  // which a source reads (image_sources.hpp).
  std::uint64_t skipped{};
  std::atomic<std::uint64_t> last_frame_id{};
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

// Page-locked host memory, grown as needed, which the GPU copies into
// without the CPU waiting for each copy.
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  ~PinnedBuffer();

  // At least `bytes`; the contents are kept only when it does not grow.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> reserve(std::size_t bytes);
  template <typename T>
  [[nodiscard]] T* as() const noexcept {
    return static_cast<T*>(data_);
  }

 private:
  void* data_{};
  std::size_t bytes_{};
};

// Checks, times, prints and writes each image's instances. Per image it
// copies to the host only what its settings read: the count always, the
// scores, boxes and prompt ids when it verifies or prints, the masks when it
// verifies, the label image and the image when it writes images, and the
// image when it checks against the reference. The copies go to pinned memory
// on the runtime's stream; it waits once for the instances, which marks the
// latency, and once more for the masks, labels and image when it reads them.
class SegmentationReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if the report is missing or the settings
  // unusable.
  explicit SegmentationReportOp(SegmentationReportConfig config);
  ~SegmentationReportOp() override;
  SegmentationReportOp(const SegmentationReportOp&) = delete;
  SegmentationReportOp& operator=(const SegmentationReportOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  // Reads the prompt table for the reference check.
  // @throws std::runtime_error if it cannot be read, or CUDA fails.
  void start() override;
  void stop() override;
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
  void release() noexcept;

  SegmentationReportConfig config_;
  std::uint64_t last_frame_id_{};
  std::uint32_t seen_{};
  // Read in start() for the reference check.
  std::optional<holoscan::sam3::PromptTable> table_;
  cudaEvent_t instances_copied_{};
  PinnedBuffer count_;
  PinnedBuffer masks_;
  PinnedBuffer labels_;
  PinnedBuffer scores_;
  PinnedBuffer boxes_;
  PinnedBuffer prompt_ids_;
  PinnedBuffer pixels_;
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
