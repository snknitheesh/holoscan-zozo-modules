// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The vic backend of a build without VPI: never available.

#include <memory>
#include <stdexcept>

#include "vic.hpp"

namespace holoscan::color_convert::detail {

bool vpi_compiled() noexcept { return false; }

bool vpi_has_vic() { return false; }

std::unique_ptr<VicEngine> make_vic_engine() {
  throw std::invalid_argument("backend vic: this build has no VPI; install VPI 4 and rebuild");
}

}  // namespace holoscan::color_convert::detail
