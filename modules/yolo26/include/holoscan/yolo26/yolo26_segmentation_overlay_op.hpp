// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Yolo26SegmentationOverlayOp: turns one image's detections and label image
// from Yolo26SegmentationOp into two Holoviz layers: an RGBA8 image with each
// instance's pixels filled semi-transparently in its class's color and the
// boxes drawn over them as Yolo26OverlayOp draws them, transparent
// elsewhere; and the same "<class> 0.87" labels. Both carry the image's
// capture time, so Holoviz draws them over their own frame.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/yolo26/yolo26_overlay_op.hpp>

namespace holoscan::yolo26 {

struct Yolo26SegmentationOverlayConfig {
  // The image's size, in pixels: the label image's.
  int width{};
  int height{};
  // The segmenter's capacity: the length of its detection outputs.
  int max_detections{32};
  // Class names by id; ids without one show as "class N".
  std::vector<std::string> labels;
  // The masks' opacity, 0 (invisible) to 255 (opaque); the boxes are opaque.
  std::uint8_t mask_alpha{110};
};

class Yolo26SegmentationOverlayOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for a non-positive size or capacity.
  explicit Yolo26SegmentationOverlayOp(Yolo26SegmentationOverlayConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // Yolo26SegmentationOp's outputs, in GPU or host memory; the label image
  // must be height x width.
  holoscan::Input<holoscan::Tensor> boxes;
  holoscan::Input<holoscan::Tensor> scores;
  holoscan::Input<holoscan::Tensor> class_ids;
  holoscan::Input<holoscan::Tensor> count;
  holoscan::Input<holoscan::Tensor> label_image;
  // For a Holoviz text layer, as Yolo26OverlayOp's. Emitted first.
  holoscan::Output<holoscan::Tensor> labels;
  // For a Holoviz RGBA8 image layer: the masks and boxes over a transparent
  // background. Emitted last, so it can trigger the frame.
  holoscan::Output<holoscan::schema::ImageT> overlay;

 private:
  Yolo26SegmentationOverlayConfig config_;
  std::vector<float> boxes_;
  std::vector<float> scores_;
  std::vector<std::int32_t> class_ids_;
  std::vector<std::int32_t> label_image_;
};

}  // namespace holoscan::yolo26
