// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Sam3OverlayOp: turns one image's instances from Sam3SegmentationOp into
// two HoloViz layers: the label image colored on the GPU, each instance
// translucent in its own color with an opaque outline, on a transparent RGBA8
// image; and a "<prompt> 0.87" label at each instance's box. Both carry the
// image's capture time, so HoloViz draws them over their own frame.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sam3/prompts.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::sam3 {

// Bytes of label text per instance; longer labels are cut.
inline constexpr std::uint32_t kOverlayLabelBytes = 48;

struct Sam3OverlayConfig {
  // The largest image, which sizes the overlay's buffers.
  int max_width{1920};
  int max_height{1080};
  // The segmenter's max_instances: the length of its outputs.
  int max_instances{16};
  // Names of the prompts by index (display_name()); others show as
  // "prompt N".
  std::vector<std::string> prompt_names;
  // Optional: the segmenter's prompt control, whose prompts, once set, name
  // the instances instead.
  std::shared_ptr<Sam3PromptControl> prompt_control;
  // The opacity of an instance's inside; its outline is opaque.
  std::uint8_t fill_alpha{110};
};

class Sam3OverlayOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for a non-positive size or capacity.
  explicit Sam3OverlayOp(Sam3OverlayConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // Sam3SegmentationOp's outputs: the label image in GPU memory, the others
  // in GPU or host memory.
  holoscan::Input<holoscan::Tensor> labels;
  holoscan::Input<holoscan::Tensor> scores;
  holoscan::Input<holoscan::Tensor> boxes;
  holoscan::Input<holoscan::Tensor> prompt_ids;
  holoscan::Input<holoscan::Tensor> count;
  // For a HoloViz text layer: per instance, a dark shadow row and then the
  // label row; unused rows are empty. Emitted first.
  holoscan::Output<holoscan::Tensor> text;
  // For a HoloViz RGBA8 image layer, in GPU memory: the instances over a
  // transparent background. Emitted last, so it can trigger the frame.
  holoscan::Output<holoscan::schema::ImageT> overlay;

 private:
  Sam3OverlayConfig config_;
  std::vector<float> scores_;
  std::vector<float> boxes_;
  std::vector<std::int32_t> prompt_ids_;
};

}  // namespace holoscan::sam3
