// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "image_output.hpp"

#include <cstdint>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>

namespace holoscan::fiducials {

void image_output(holoscan::OperatorSpec& spec, holoscan::Output<holoscan::schema::ImageT>& image,
                  const char* name, holoscan::MemoryKind memory, int width, int height,
                  int channels) {
  constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
  const std::uint64_t bytes = std::uint64_t{static_cast<std::uint32_t>(width)} * height *
                              static_cast<std::uint32_t>(channels);
  spec.output(image, name)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = memory,
                  .dtype = kUInt8,
                  .rank = 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(bytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

}  // namespace holoscan::fiducials
