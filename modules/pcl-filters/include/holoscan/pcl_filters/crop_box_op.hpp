// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CropBoxOp: keeps the points inside (or outside) an axis-aligned box in a
// given frame, and within a range of distance from that frame's origin. With
// one axis limited it is a pass-through filter on that axis; with only the
// distance limited, a range filter. See cloud_filter_op.hpp for the ports.

#pragma once

#include <cstdint>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/pcl_filters/cloud_filter_op.hpp>
#include <holoscan/pcl_filters/filters.hpp>

namespace holoscan::pcl_filters {

struct CropBoxConfig {
  // The region, and the transform from the cloud's frame into its frame.
  CropBoxParams box;
  // Declare the `colors` input and `filtered_colors` output.
  bool color{false};
  // Keep the input's shape (removed points NaN), or pack the kept points.
  OutputLayout layout{OutputLayout::kKeep};
  // Largest cloud accepted; see CloudSettings.
  std::uint32_t max_points{1920U * 1080U};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const CropBoxConfig& config);

class CropBoxOp final : public CloudFilterOp {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit CropBoxOp(CropBoxConfig config);

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

 private:
  CropBoxConfig config_;
};

}  // namespace holoscan::pcl_filters
