// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// OutlierRemovalOp: removes points that are far from their neighbors
// (statistical) or have too few of them (radius), using a grid on the GPU.
// See cloud_filter_op.hpp for the ports and OutlierFilter in filters.hpp for
// the method.

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/pcl_filters/cloud_filter_op.hpp>
#include <holoscan/pcl_filters/filters.hpp>

namespace holoscan::pcl_filters {

struct OutlierRemovalConfig {
  OutlierParams outliers;
  // The output's layout, size and fields.
  CloudSettings cloud;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const OutlierRemovalConfig& config);

class OutlierRemovalOp final : public CloudFilterOp {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit OutlierRemovalOp(OutlierRemovalConfig config);
  ~OutlierRemovalOp() override;

  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

 private:
  OutlierRemovalConfig config_;
  std::unique_ptr<OutlierFilter> filter_;
};

}  // namespace holoscan::pcl_filters
