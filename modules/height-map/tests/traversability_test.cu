// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// compute_traversability() on elevation layers of synthetic terrain made at
// test time: flat ground, ramps of known slopes (the costs rising with the
// slope), steps of known heights, a box and a pit, the unknown mask, a mapped
// noisy ramp, the CPU reference, determinism; and the color views.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <holoscan/height_map/height_mapper.hpp>
#include <holoscan/height_map/synthetic_terrain.hpp>
#include <holoscan/height_map/traversability.hpp>
#include <holoscan/height_map/views.hpp>

#include "reference.hpp"
#include "terrain_support.hpp"
#include "test_support.hpp"

namespace holoscan::height_map {
namespace {

using test::cell_x;
using test::cell_y;
using test::percentile;
using test::Scratch;

constexpr int kRows = 60;
constexpr int kCols = 80;
constexpr float kResolution = 0.05F;
constexpr double kOriginX = -2.0;
constexpr double kOriginY = -1.5;
constexpr MapGeometry kGrid{kResolution, kRows, kCols, kOriginX, kOriginY};

struct Result {
  std::vector<float> cost, slope, step, roughness;
  std::vector<std::uint8_t> unknown;
  [[nodiscard]] std::size_t at(int r, int c) const {
    return static_cast<std::size_t>(r) * kCols + c;
  }
};

Result run(const std::vector<float>& elevation, const TraversabilityParams& params,
           int rows = kRows, int cols = kCols, float resolution = kResolution) {
  const std::size_t cells = static_cast<std::size_t>(rows) * cols;
  Scratch<float> in(elevation);
  Scratch<float> cost(cells), slope(cells), step(cells), roughness(cells);
  Scratch<std::uint8_t> unknown(cells);
  const TraversabilityLayers layers{cost.data, unknown.data, slope.data, step.data, roughness.data};
  if (compute_traversability(in.data, rows, cols, resolution, params, layers, nullptr) !=
      cudaSuccess) {
    throw std::runtime_error("compute_traversability failed");
  }
  return Result{cost.host(), slope.host(), step.host(), roughness.host(), unknown.host()};
}

// The terrain's height at every cell center.
std::vector<float> exact_layer(const Terrain& terrain) {
  std::vector<float> out;
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      out.push_back(terrain_height(terrain, static_cast<float>(cell_x(kGrid, c)),
                                   static_cast<float>(cell_y(kGrid, r))));
    }
  }
  return out;
}

void check_flat() {
  const TraversabilityParams params;
  const Result out = run(exact_layer(Terrain{.base = 0.3F}), params);
  int wrong = 0;
  int edges_known = 0;
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      const std::size_t i = out.at(r, c);
      const int window = (std::min(r + 2, kRows - 1) - std::max(r - 2, 0) + 1) *
                         (std::min(c + 2, kCols - 1) - std::max(c - 2, 0) + 1);
      if (window < 13) {
        // Fewer than half of the 25 cells: near a corner.
        edges_known += out.unknown[i] == 0U ? 1 : 0;
        continue;
      }
      wrong += out.unknown[i] == 0U && out.cost[i] == 0.0F && out.slope[i] == 0.0F &&
                       out.step[i] == 0.0F && out.roughness[i] == 0.0F
                   ? 0
                   : 1;
    }
  }
  VERIFY_MSG(wrong == 0, "flat ground costs nothing, " + std::to_string(wrong) + " wrong");
  VERIFY_MSG(edges_known == 0, "windows less than half inside the grid are unknown");
}

void check_ramps() {
  const TraversabilityParams params;
  float last_cost = -1.0F;
  for (const float degrees : {2.0F, 5.0F, 10.0F, 15.0F, 20.0F, 25.0F, 29.0F, 31.0F, 40.0F}) {
    Terrain ramped;
    add_feature(ramped, ramp(-3.0F, 3.0F, -3.0F, 3.0F, degrees));
    const Result out = run(exact_layer(ramped), params);
    const double grade = std::tan(degrees * 3.14159265358979 / 180.0);
    std::vector<double> slope_errors;
    double worst_step = 0.0;
    double worst_roughness = 0.0;
    double cost_spread = 0.0;
    const float center_cost = out.cost[out.at(kRows / 2, kCols / 2)];
    for (int r = 2; r < kRows - 2; ++r) {
      for (int c = 2; c < kCols - 2; ++c) {
        const std::size_t i = out.at(r, c);
        slope_errors.push_back(std::fabs(out.slope[i] - degrees));
        worst_step = std::max(
            worst_step, std::fabs(out.step[i] - grade * kResolution * 2 * params.step_radius));
        worst_roughness = std::max(worst_roughness, static_cast<double>(out.roughness[i]));
        cost_spread =
            std::max(cost_spread, static_cast<double>(std::fabs(out.cost[i] - center_cost)));
      }
    }
    const std::string at = std::to_string(degrees) + " deg ramp: ";
    std::printf("  %4.1f deg ramp: slope error max %.4f deg, cost %.3f\n", degrees,
                percentile(slope_errors, 1.0), center_cost);
    VERIFY_MSG(percentile(slope_errors, 1.0) < 0.02,
               at + "slope error " + std::to_string(percentile(slope_errors, 1.0)));
    VERIFY_MSG(worst_step < 4e-4,
               at + "step is the window's rise, off by " + std::to_string(worst_step));
    VERIFY_MSG(worst_roughness < 2e-4, at + "roughness " + std::to_string(worst_roughness));
    VERIFY_MSG(cost_spread < 1e-3, at + "one cost over the ramp");
    if (degrees < 30.0F) {
      VERIFY_MSG(center_cost > last_cost && center_cost < 1.0F,
                 at + "the cost rises with the slope: " + std::to_string(center_cost));
    } else {
      VERIFY_MSG(center_cost == 1.0F, at + "steeper than max_slope_deg costs 1");
    }
    last_cost = center_cost;
  }
}

void check_steps() {
  // The plane fit across an edge reads a step as a slope too; with a steep
  // max_slope_deg, the step height alone decides whether it is lethal.
  TraversabilityParams params;
  params.max_step = 0.2F;
  params.max_slope_deg = 60.0F;
  for (const float height : {0.04F, 0.1F, 0.15F, 0.25F}) {
    Terrain stepped;
    // Edges at x = -0.8, -0.3, ..., 1.7 (between cells), across every row.
    add_feature(stepped, steps(-0.8F, 3.0F, -3.0F, 3.0F, height, 0.5F));
    const Result out = run(exact_layer(stepped), params);
    int wrong_edge = 0;
    int wrong_flat = 0;
    float worst_edge_cost = 0.0F;
    for (int r = 2; r < kRows - 2; ++r) {
      for (int c = 2; c < kCols - 2; ++c) {
        const std::size_t i = out.at(r, c);
        const double x = cell_x(kGrid, c);
        const double edge = terrain_edge_distance(stepped, x, cell_y(kGrid, r));
        if (edge < kResolution) {
          // Next to an edge: the step's height, exactly.
          wrong_edge += std::fabs(out.step[i] - height) < 1e-5F ? 0 : 1;
          worst_edge_cost = std::max(worst_edge_cost, out.cost[i]);
        } else if (edge > (params.step_radius + 0.5) * kResolution) {
          // Beyond the windows: flat.
          wrong_flat += out.step[i] == 0.0F && out.slope[i] < 1e-3F && out.cost[i] < 1e-3F ? 0 : 1;
        }
      }
    }
    const std::string at = std::to_string(height) + " m steps: ";
    VERIFY_MSG(wrong_edge == 0, at + "step height at the edges, " + std::to_string(wrong_edge));
    VERIFY_MSG(wrong_flat == 0, at + "flat between them, " + std::to_string(wrong_flat));
    if (height >= params.max_step) {
      VERIFY_MSG(worst_edge_cost == 1.0F, at + "too high to step on");
    } else {
      VERIFY_MSG(worst_edge_cost > 0.0F && worst_edge_cost < 1.0F, at + "a cost below 1");
    }
  }
}

void check_box_and_pit() {
  const TraversabilityParams params;
  for (const bool is_box : {true, false}) {
    Terrain terrain;
    add_feature(terrain,
                is_box ? box(-0.5F, 0.5F, -0.5F, 0.5F, 0.3F) : pit(-0.5F, 0.5F, -0.5F, 0.5F, 0.3F));
    const Result out = run(exact_layer(terrain), params);
    int wrong = 0;
    for (int r = 2; r < kRows - 2; ++r) {
      for (int c = 2; c < kCols - 2; ++c) {
        const std::size_t i = out.at(r, c);
        const double edge = terrain_edge_distance(terrain, cell_x(kGrid, c), cell_y(kGrid, r));
        if (edge < kResolution) {
          wrong += out.cost[i] == 1.0F && std::fabs(out.step[i] - 0.3F) < 1e-5F ? 0 : 1;
        } else if (edge > 1.5 * (params.step_radius + 0.5) * kResolution) {
          // Beyond the square windows, corners included.
          wrong += out.cost[i] == 0.0F ? 0 : 1;
        }
      }
    }
    VERIFY_MSG(wrong == 0, std::string(is_box ? "box" : "pit") +
                               ": lethal at its rim, free elsewhere, " + std::to_string(wrong));
  }
}

void check_unknown() {
  const TraversabilityParams params;
  std::vector<float> layer = exact_layer(Terrain{.base = 0.0F});
  const float nan = std::numeric_limits<float>::quiet_NaN();
  // A 10x10 hole, and a lone known cell in a sea of unknown.
  for (int r = 20; r < 30; ++r) {
    for (int c = 30; c < 40; ++c) {
      layer[static_cast<std::size_t>(r) * kCols + c] = nan;
    }
  }
  for (int r = 40; r < kRows; ++r) {
    for (int c = 0; c < 20; ++c) {
      layer[static_cast<std::size_t>(r) * kCols + c] = r == 50 && c == 10 ? 0.0F : nan;
    }
  }
  const Result out = run(layer, params);
  int wrong = 0;
  for (int r = 20; r < 30; ++r) {
    for (int c = 30; c < 40; ++c) {
      const std::size_t i = out.at(r, c);
      wrong += out.unknown[i] == 1U && std::isnan(out.cost[i]) && std::isnan(out.slope[i]) &&
                       std::isnan(out.step[i]) && std::isnan(out.roughness[i])
                   ? 0
                   : 1;
    }
  }
  VERIFY_MSG(wrong == 0, "unknown elevation, unknown cost");
  // Next to the hole the window still holds enough: known.
  VERIFY(out.unknown[out.at(19, 35)] == 0U && out.cost[out.at(19, 35)] == 0.0F);
  VERIFY_MSG(out.unknown[out.at(50, 10)] == 1U, "a lone cell fits no plane");
  // Collinear cells fit no plane either: a one-cell-wide strip.
  std::vector<float> strip(static_cast<std::size_t>(kRows) * kCols, nan);
  for (int c = 0; c < kCols; ++c) {
    strip[static_cast<std::size_t>(30) * kCols + c] = 0.0F;
  }
  TraversabilityParams loose;
  loose.min_valid_fraction = 0.0F;
  const Result line = run(strip, loose);
  VERIFY_MSG(
      std::all_of(line.unknown.begin(), line.unknown.end(), [](std::uint8_t v) { return v == 1U; }),
      "collinear cells are unknown");
}

// The slope of a mapped noisy ramp.
void check_mapped_ramp() {
  Terrain ramped;
  add_feature(ramped, ramp(-3.0F, 3.0F, -3.0F, 3.0F, 18.0F));
  HeightMapParams p;
  p.geometry = kGrid;
  p.elevation = ElevationMode::kMean;
  HeightMapper mapper(p);
  test::update(mapper, test::terrain_cloud(ramped, -2.0F, 2.0F, -1.5F, 1.5F, 0.01F, 0.003F, 3U),
               Transform{}, 1'000'000'000);
  const Result out = run(test::read_layers(mapper).elevation, TraversabilityParams{});
  std::vector<double> errors;
  for (int r = 2; r < kRows - 2; ++r) {
    for (int c = 2; c < kCols - 2; ++c) {
      errors.push_back(std::fabs(out.slope[out.at(r, c)] - 18.0F));
    }
  }
  std::printf("  mapped 18 deg ramp, 3 mm noise: slope error median %.3f, p95 %.3f deg\n",
              percentile(errors, 0.5), percentile(errors, 0.95));
  VERIFY_MSG(percentile(errors, 0.5) < 0.1,
             "mapped ramp: median slope error " + std::to_string(percentile(errors, 0.5)));
  VERIFY_MSG(percentile(errors, 0.95) < 0.25,
             "mapped ramp: p95 slope error " + std::to_string(percentile(errors, 0.95)));
}

void check_reference_and_determinism() {
  std::mt19937 rng(11U);
  std::normal_distribution<float> noise(0.0F, 0.01F);
  std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
  const Terrain terrain = course_terrain();
  constexpr int rows = 47;
  constexpr int cols = 53;
  constexpr float resolution = 0.07F;
  std::vector<float> layer;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const float x = 1.0F + (static_cast<float>(c) + 0.5F) * resolution;
      const float y = -1.6F + (static_cast<float>(r) + 0.5F) * resolution;
      layer.push_back(uniform(rng) < 0.15F ? std::numeric_limits<float>::quiet_NaN()
                                           : terrain_height(terrain, x, y) + noise(rng));
    }
  }
  for (const int radius : {1, 2, 4}) {
    TraversabilityParams params;
    params.window_radius = radius;
    params.step_radius = radius == 1 ? 2 : 1;
    params.min_valid_fraction = 0.4F;
    const Result got = run(layer, params, rows, cols, resolution);
    const Result again = run(layer, params, rows, cols, resolution);
    const test::ReferenceTraversability want =
        test::reference_traversability(layer, rows, cols, resolution, params);
    const std::string at = "radius " + std::to_string(radius) + ": ";
    VERIFY_MSG(got.unknown == want.unknown, at + "the unknown mask matches the reference");
    double slope_error = 0.0;
    double cost_error = 0.0;
    bool exact = true;
    for (std::size_t i = 0; i < got.cost.size(); ++i) {
      if (want.unknown[i] != 0U) {
        exact = exact && std::isnan(got.cost[i]) && std::isnan(got.slope[i]);
        continue;
      }
      slope_error =
          std::max(slope_error, static_cast<double>(std::fabs(got.slope[i] - want.slope[i])));
      cost_error = std::max(cost_error, static_cast<double>(std::fabs(got.cost[i] - want.cost[i])));
      exact = exact && got.step[i] == want.step[i] && got.roughness[i] == want.roughness[i];
    }
    VERIFY_MSG(exact, at + "step heights and roughness match exactly");
    VERIFY_MSG(slope_error < 1e-4, at + "slopes within 1e-4 deg: " + std::to_string(slope_error));
    VERIFY_MSG(cost_error < 1e-5, at + "costs within 1e-5: " + std::to_string(cost_error));
    VERIFY_MSG(test::same_floats(got.cost, again.cost) && test::same_floats(got.slope, again.slope),
               at + "deterministic");
    VERIFY(std::count(got.unknown.begin(), got.unknown.end(), 0U) > 1000);
  }
}

void check_views() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> elevation{nan, 0.0F, 0.5F, 1.0F, 2.0F, -1.0F};
  Scratch<float> in(elevation);
  Scratch<std::uint8_t> rgb(18);
  Scratch<float> range(2);
  // Fixed range 0..1: clamped below and above.
  VERIFY(colorize_elevation(in.data, 2, 3, 0.0F, 1.0F, nullptr, rgb.data, nullptr) == cudaSuccess);
  std::vector<std::uint8_t> got = rgb.host();
  VERIFY(got[0] == kUnknownGray && got[1] == kUnknownGray && got[2] == kUnknownGray);
  VERIFY_MSG(got[3] == 26 && got[4] == 38 && got[5] == 153, "low is dark blue");
  VERIFY_MSG(got[9] == 217 && got[10] == 26 && got[11] == 26, "high is red");
  VERIFY(std::equal(got.begin() + 12, got.begin() + 15, got.begin() + 9));  // Clamped high.
  VERIFY(std::equal(got.begin() + 15, got.begin() + 18, got.begin() + 3));  // Clamped low.
  VERIFY_MSG(got[6] == 51 && got[7] == 204 && got[8] == 51, "middle is green");
  // Automatic range -1..2: -1 dark blue, 2 red.
  VERIFY(colorize_elevation(in.data, 2, 3, 0.0F, 0.0F, range.data, rgb.data, nullptr) ==
         cudaSuccess);
  got = rgb.host();
  VERIFY_MSG(got[15] == 26 && got[16] == 38 && got[17] == 153, "auto: the lowest is dark blue");
  VERIFY_MSG(got[12] == 217 && got[13] == 26 && got[14] == 26, "auto: the highest is red");
  VERIFY(colorize_elevation(in.data, 2, 3, 0.0F, 0.0F, nullptr, rgb.data, nullptr) ==
         cudaErrorInvalidValue);

  const std::vector<float> cost{nan, 0.0F, 0.5F, 1.0F};
  Scratch<float> costs(cost);
  Scratch<std::uint8_t> cost_rgb(12);
  VERIFY(colorize_cost(costs.data, 1, 4, cost_rgb.data, nullptr) == cudaSuccess);
  got = cost_rgb.host();
  VERIFY(got == (std::vector<std::uint8_t>{kUnknownGray, kUnknownGray, kUnknownGray, 0, 255, 0, 255,
                                           255, 0, 255, 0, 0}));
}

void check_rejected_params() {
  const auto rejects = [](TraversabilityParams p, const std::string& what) {
    const char* message = nullptr;
    VERIFY_MSG(!valid_params(p, &message), what + " is rejected");
    VERIFY_MSG(message != nullptr && std::string(message).find(what) != std::string::npos,
               "names " + what);
    Scratch<float> one(1);
    VERIFY(compute_traversability(one.data, 1, 1, 0.1F, p, TraversabilityLayers{}, nullptr) ==
           cudaErrorInvalidValue);
  };
  TraversabilityParams p;
  p.window_radius = 0;
  rejects(p, "window_radius");
  p = TraversabilityParams{};
  p.step_radius = 11;
  rejects(p, "step_radius");
  p = TraversabilityParams{};
  p.min_valid_fraction = 1.5F;
  rejects(p, "min_valid_fraction");
  p = TraversabilityParams{};
  p.max_step = 0.0F;
  rejects(p, "max_step");
  p = TraversabilityParams{};
  p.slope_weight = p.step_weight = p.roughness_weight = 0.0F;
  rejects(p, "weights");
  Scratch<float> one(1);
  VERIFY(compute_traversability(one.data, 1, 1, 0.0F, TraversabilityParams{},
                                TraversabilityLayers{}, nullptr) == cudaErrorInvalidValue);
  VERIFY(compute_traversability(nullptr, 0, 0, 0.1F, TraversabilityParams{}, TraversabilityLayers{},
                                nullptr) == cudaSuccess);
}

}  // namespace
}  // namespace holoscan::height_map

int main() {
  using holoscan::height_map::test::TestCase;
  namespace hm = holoscan::height_map;
  return holoscan::height_map::test::run({
      TestCase{"flat ground", hm::check_flat},
      TestCase{"ramps of known slopes", hm::check_ramps},
      TestCase{"steps of known heights", hm::check_steps},
      TestCase{"a box and a pit", hm::check_box_and_pit},
      TestCase{"the unknown mask", hm::check_unknown},
      TestCase{"a mapped noisy ramp", hm::check_mapped_ramp},
      TestCase{"CPU reference and determinism", hm::check_reference_and_determinism},
      TestCase{"color views", hm::check_views},
      TestCase{"rejected settings", hm::check_rejected_params},
  });
}
