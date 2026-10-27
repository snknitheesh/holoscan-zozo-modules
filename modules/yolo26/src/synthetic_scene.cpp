// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/yolo26/stub_model.hpp>
#include <holoscan/yolo26/synthetic_scene.hpp>

namespace holoscan::yolo26 {
namespace {

constexpr std::array<std::uint8_t, 3> kBackground{40, 40, 40};

// A rectangle of fractions of the image: left and top edges travel between
// their two values over a 50-frame cycle, back and forth.
SceneRect place(int width, int height, std::uint32_t frame, float left_from, float left_to,
                float top_from, float top_to, float rect_width, float rect_height,
                std::array<std::uint8_t, 3> rgb) {
  const std::uint32_t phase = frame % 50U;
  const float t = static_cast<float>(phase < 25U ? phase : 50U - phase) / 25.0F;
  const float left = left_from + (left_to - left_from) * t;
  const float top = top_from + (top_to - top_from) * t;
  SceneRect rect;
  rect.x0 = static_cast<int>(left * static_cast<float>(width));
  rect.y0 = static_cast<int>(top * static_cast<float>(height));
  rect.x1 = std::min(width, rect.x0 + std::max(4, static_cast<int>(rect_width * width)));
  rect.y1 = std::min(height, rect.y0 + std::max(4, static_cast<int>(rect_height * height)));
  rect.rgb = rgb;
  return rect;
}

}  // namespace

SyntheticScene make_scene(int width, int height, std::uint32_t frame) {
  if (width < kMinSyntheticWidth || height < kMinSyntheticHeight || width > kMaxSyntheticSide ||
      height > kMaxSyntheticSide) {
    throw std::invalid_argument("synthetic images must be " + std::to_string(kMinSyntheticWidth) +
                                "x" + std::to_string(kMinSyntheticHeight) + " to " +
                                std::to_string(kMaxSyntheticSide) + "x" +
                                std::to_string(kMaxSyntheticSide));
  }
  SyntheticScene scene;
  scene.width = width;
  scene.height = height;
  // The red one in the left half, the blue one in the right; the distractors
  // between and below them.
  scene.rects[0] =
      place(width, height, frame, 0.04F, 0.20F, 0.08F, 0.30F, 0.26F, 0.34F, {230, 30, 30});
  scene.rects[1] =
      place(width, height, frame, 0.70F, 0.55F, 0.50F, 0.12F, 0.28F, 0.30F, {30, 30, 230});
  scene.rects[2] =
      place(width, height, frame, 0.45F, 0.40F, 0.70F, 0.60F, 0.12F, 0.20F, {30, 200, 30});
  scene.rects[3] =
      place(width, height, frame, 0.05F, 0.10F, 0.80F, 0.75F, 0.30F, 0.12F, {250, 250, 250});
  return scene;
}

std::vector<BoxDetection> scene_truth(const SyntheticScene& scene) {
  std::vector<BoxDetection> truth;
  for (int c = 0; c < kStubClasses; ++c) {
    const SceneRect& rect = scene.rects[static_cast<std::size_t>(c)];
    truth.push_back(BoxDetection{.x1 = static_cast<float>(rect.x0),
                                 .y1 = static_cast<float>(rect.y0),
                                 .x2 = static_cast<float>(rect.x1),
                                 .y2 = static_cast<float>(rect.y1),
                                 .score = kStubScores[c],
                                 .class_id = c});
  }
  return truth;
}

void render_scene(const SyntheticScene& scene, PixelFormat format, std::uint8_t* pixels,
                  std::size_t pitch) {
  const int n = channels(format);
  const bool bgr = blue_first(format);
  const auto put = [&](int x, int y, const std::array<std::uint8_t, 3>& rgb) {
    std::uint8_t* p =
        pixels + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x) * n;
    p[0] = rgb[bgr ? 2 : 0];
    p[1] = rgb[1];
    p[2] = rgb[bgr ? 0 : 2];
    if (n == 4) {
      p[3] = 255;
    }
  };
  for (int y = 0; y < scene.height; ++y) {
    for (int x = 0; x < scene.width; ++x) {
      put(x, y, kBackground);
    }
  }
  // Later rectangles cover earlier ones; the two detected ones come last so
  // that nothing covers them.
  for (const std::size_t i : {std::size_t{3}, std::size_t{2}, std::size_t{0}, std::size_t{1}}) {
    const SceneRect& rect = scene.rects[i];
    for (int y = rect.y0; y < rect.y1; ++y) {
      for (int x = rect.x0; x < rect.x1; ++x) {
        put(x, y, rect.rgb);
      }
    }
  }
}

float iou(const BoxDetection& a, const BoxDetection& b) {
  const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
  const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
  if (!(area_a > 0.0F) || !(area_b > 0.0F)) {
    return 0.0F;
  }
  const float w = std::max(0.0F, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float h = std::max(0.0F, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float overlap = w * h;
  return overlap / (area_a + area_b - overlap);
}

float edge_error(const BoxDetection& a, const BoxDetection& b) {
  return std::max({std::fabs(a.x1 - b.x1), std::fabs(a.y1 - b.y1), std::fabs(a.x2 - b.x2),
                   std::fabs(a.y2 - b.y2)});
}

TruthMatch match_truth(std::span<const BoxDetection> detections,
                       std::span<const BoxDetection> truth, float tolerance,
                       float score_tolerance) {
  TruthMatch result;
  std::vector<bool> used(detections.size(), false);
  for (const BoxDetection& t : truth) {
    float nearest = std::numeric_limits<float>::infinity();
    std::size_t match = detections.size();
    for (std::size_t i = 0; i < detections.size(); ++i) {
      const BoxDetection& d = detections[i];
      if (d.class_id != t.class_id) {
        continue;
      }
      const float error = edge_error(d, t);
      nearest = std::min(nearest, error);
      if (!used[i] && match == detections.size() && error <= tolerance &&
          std::fabs(d.score - t.score) <= score_tolerance) {
        match = i;
      }
    }
    result.max_edge_error = std::max(result.max_edge_error, nearest);
    if (match == detections.size()) {
      ++result.missed;
    } else {
      used[match] = true;
    }
  }
  result.extra = static_cast<std::uint32_t>(std::count(used.begin(), used.end(), false));
  return result;
}

}  // namespace holoscan::yolo26
