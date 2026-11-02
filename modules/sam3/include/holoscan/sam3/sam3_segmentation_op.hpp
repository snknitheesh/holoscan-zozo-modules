// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Sam3SegmentationOp: promptable concept segmentation with SAM 3 on the GPU.
// An image goes in; every instance of every prompt's concept comes out, best
// first, as a mask at the image's size, a score, a box and the index of its
// prompt, with a label image of them all. Prompts are short noun phrases
// ("apple"), exemplar boxes, or both (prompts.hpp).
//
// The model is split in two TensorRT engines, run through the SDK's
// inference library (HoloInfer):
//
// - The encoder, SAM 3's vision backbone and its feature pyramid, the heavy
//   part, runs once per image: input float32 [1, 3, S, S] RGB normalized to
//   [-1, 1], outputs fpn0, fpn1 and fpn2, float32 [1, C, H_i, W_i].
// - The decoder, SAM 3's prompt encoder, its DETR-style detector and its mask
//   head, runs once per prompt on the encoder's features, which it reads in
//   place: inputs fpn0..fpn2, text_features [1, T, D], text_mask [1, T],
//   boxes [1, B, 4], box_labels [1, B] and box_valid [1, B]; outputs masks
//   [1, Q, M, M] (logits), boxes_xyxy [1, Q, 4] (fractions of the image) and
//   scores [1, Q] (probabilities, the presence score included).
//
// tools/prepare_model.py exports both from Meta's SAM 3 checkpoint, and the
// prompt table with the text embeddings. Given .onnx files, HoloInfer builds
// TensorRT engines for this GPU and caches them beside the files, so later
// runs start in seconds; any other file is taken as such an engine.
//
// Inside, a CUDA kernel stretches the image into the encoder's input; after
// each prompt's decoder run, a kernel keeps its instances that score above
// the threshold; a last pass ranks them all and upsamples their masks.
// Everything stays on the GPU, and the outputs are GPU tensors.
//
// Each image's outputs carry its capture time and frame id. They go out one
// after another: should one fail to go out, those before it have been
// delivered alone and compute() returns the error. Pair them by capture time
// (OnAligned), as Sam3OverlayOp does.
//
// On a blocking edge, an output a busy consumer has no room for is such a
// failure (kBackpressured): a lossless graph paces its source to what its
// consumers keep up with. For a live source, connect the input and every
// output with holoscan::ConnectionOptions{.queue_policy =
// holoscan::QueuePolicy::kDropOldest, .queue_depth = 1}: a full queue then
// drops its oldest sample, so the operator never waits for a slower consumer
// and takes the newest image.
//
// The operator segments images one at a time; video tracking, which carries
// instances from frame to frame with SAM 3's memory, is a separate operator's
// job. Its outputs are what a tracker's per-frame detections would be.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sam3/pixel_math.hpp>
#include <holoscan/sam3/prompts.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::sam3 {

// The most instances per image.
inline constexpr int kMaxInstances = 256;

struct Sam3SegmentationConfig {
  // The encoder and decoder: .onnx models, whose engines are built and cached
  // beside them, or serialized TensorRT engines built for this GPU.
  std::filesystem::path encoder;
  std::filesystem::path decoder;
  // The prompt table with the text embeddings of the prompts' texts.
  std::filesystem::path prompt_table;
  // What to segment, 1..kMaxPrompts prompts.
  std::vector<Sam3Prompt> prompts;
  // Optional: prompts that change at run time. Once its set() has succeeded,
  // the control's prompts replace these from the next image on.
  std::shared_ptr<Sam3PromptControl> prompt_control;
  // Instances need a score above this, in [0, 1). SAM 3's default.
  float score_threshold{0.5F};
  // Masks keep the pixels whose probability exceeds this, in (0, 1).
  float mask_threshold{0.5F};
  // The most instances per image, 1..kMaxInstances, across all prompts; the
  // lowest scores are dropped first, and a warning says so once per run. It
  // also sizes the outputs.
  int max_instances{16};
  // Declares the masks output. The label image holds every instance too,
  // where it is the best one; the masks also hold where instances overlap.
  bool output_masks{true};
  // The largest image, which sizes the output buffers.
  int max_width{1920};
  int max_height{1080};
  // Build the engines with FP16 kernels allowed (ignored for engine files).
  bool fp16{true};
  // Let HoloInfer replay the inference as CUDA graphs.
  bool cuda_graphs{true};
  // TensorRT's workspace limit while it builds an engine.
  std::size_t build_workspace_bytes{std::size_t{1} << 30U};
};

// @throws std::invalid_argument naming the first unusable field, prompts
// included (validate_prompt()). Whether the prompts are in the prompt table
// is checked when the table is read.
void validate_config(const Sam3SegmentationConfig& config);

// Builds and caches the TensorRT engines of an .onnx encoder and decoder
// unless they are cached already. Call it before running the graph: some
// TensorRT releases (10.13 among them) change the SIGINT and SIGTERM handlers
// while building, which the SDK's managed signal handling accepts only before
// a run starts. start() then loads the cached engines. TensorRT also starts
// threads, so create the process's holoscan::ManagedShutdownSignals before
// calling this, while main() is the only thread.
//
// It loads each model fully, as start() does, and discards it, so cached
// engines are loaded twice at startup. HoloInfer names its cached engines
// internally, so whether one exists is known only by loading it.
// @throws std::runtime_error if a model cannot be built.
void prepare_engines(const Sam3SegmentationConfig& config);

class Sam3SegmentationOp final : public holoscan::Operator<> {
 public:
  // Reads the prompt table.
  // @throws std::invalid_argument if `config` is not valid or a prompt's text
  // is not in the table; std::runtime_error if the table cannot be read.
  explicit Sam3SegmentationOp(Sam3SegmentationConfig config);
  ~Sam3SegmentationOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  // Loads the models, building their engines first if they are not cached.
  // @throws std::runtime_error if a model cannot be loaded, or the two do not
  // have the expected inputs and outputs, or do not fit each other or the
  // prompt table.
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // RGB8, BGR8, RGBA8 or BGRA8: uint8 [height, width, channels], pixels
  // packed within each row and rows any pitch apart, in GPU memory on device
  // 0 or in host memory. At most max_width x max_height.
  holoscan::Input<holoscan::schema::ImageT> image;
  // uint8 [max_instances, height, width]: 1 inside each instance, 0
  // outside; zero beyond `count`. Declared only with output_masks.
  holoscan::Output<holoscan::Tensor> masks;
  // uint16 [height, width]: 1 + the index of the best instance covering each
  // pixel, 0 for none.
  holoscan::Output<holoscan::Tensor> labels;
  // float32 [max_instances]: each instance's score, best first.
  holoscan::Output<holoscan::Tensor> scores;
  // float32 [max_instances, 4]: x1, y1, x2, y2 in image pixels, within [0,
  // width] x [0, height].
  holoscan::Output<holoscan::Tensor> boxes;
  // int32 [max_instances]: the index of each instance's prompt; -1 beyond
  // `count`.
  holoscan::Output<holoscan::Tensor> prompt_ids;
  // int32 [1]: the number of instances.
  holoscan::Output<holoscan::Tensor> count;

 private:
  struct Engines;

  [[nodiscard]] holoscan::expected<const std::uint8_t*, holoscan::Error> stage(
      const holoscan::Tensor& tensor, int width, int height, int channels, std::size_t pitch,
      cudaStream_t stream);
  // Uploads the prompts' decoder inputs, if they changed.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> load_prompts(cudaStream_t stream);
  // Prompts newer than the active ones, from the control.
  void poll_control();
  void release() noexcept;
  // Warns, once per run, if a finished image had more instances than the
  // outputs hold.
  void warn_if_dropped();

  Sam3SegmentationConfig config_;
  PromptTable table_;
  // Prompts to upload before the next image.
  std::optional<std::vector<Sam3Prompt>> pending_;
  // The prompts the uploaded decoder inputs are for, and the control's
  // generation they came from.
  std::vector<Sam3Prompt> active_;
  std::uint64_t control_generation_{};
  std::unique_ptr<Engines> engines_;
  // A GPU copy of host images, grown as needed.
  void* staging_{};
  std::size_t staging_bytes_{};
  // Recorded after each image's work. The models' buffers, the candidates and
  // `staging_` are reused by every image, but the runtime may give each
  // activation another stream, so the next image's stream waits for it before
  // touching them.
  cudaEvent_t frame_done_{};
  // Mapped host memory where the selection writes how many instances of each
  // prompt passed the threshold; read once frame_done_ has completed.
  std::int32_t* passed_{};
  int passed_prompts_{};
  bool warned_dropped_{};
};

}  // namespace holoscan::sam3
