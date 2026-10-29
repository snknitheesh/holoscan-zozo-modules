// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "cpu_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace holoscan::fiducials {
namespace {

using core::Candidate;
using core::CandidateResult;

// Whether p lies inside the clockwise (image coordinates) convex quad q.
bool inside(const std::array<Point2, 4>& q, const Point2& p) {
  for (int i = 0; i < 4; ++i) {
    const Point2& a = q[i];
    const Point2& b = q[(i + 1) % 4];
    if ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0.0) return false;
  }
  return true;
}

double area(const std::array<Point2, 4>& q) {
  double twice = 0.0;
  for (int i = 0; i < 4; ++i) twice += q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1];
  return 0.5 * std::abs(twice);
}

Point2 center(const std::array<Point2, 4>& q) {
  return {0.25 * (q[0][0] + q[1][0] + q[2][0] + q[3][0]),
          0.25 * (q[0][1] + q[1][1] + q[2][1] + q[3][1])};
}

// Path-halving find; roots are the lowest index of their component.
std::int32_t find_root(std::vector<std::int32_t>& labels, std::int32_t i) {
  while (labels[i] != i) {
    labels[i] = labels[labels[i]];
    i = labels[i];
  }
  return i;
}

void unite(std::vector<std::int32_t>& labels, std::int32_t a, std::int32_t b) {
  a = find_root(labels, a);
  b = find_root(labels, b);
  if (a < b) {
    labels[b] = a;
  } else if (b < a) {
    labels[a] = b;
  }
}

}  // namespace

core::Params make_params(const DetectorOptions& options, int width, int height) {
  validate_options(options);
  if (width < 16 || height < 16 || width > core::kMaxImageSide || height > core::kMaxImageSide) {
    throw std::invalid_argument("image sides must be 16 to " +
                                std::to_string(core::kMaxImageSide) + " pixels");
  }
  const Dictionary dictionary(options.dictionary);
  core::Params p{};
  p.width = width;
  p.height = height;
  p.tile_size = options.tile_size;
  p.tiles_x = (width + options.tile_size - 1) / options.tile_size;
  p.tiles_y = (height + options.tile_size - 1) / options.tile_size;
  p.min_contrast = options.min_contrast;
  p.min_marker_px = options.min_marker_px;
  // A marker's border alone covers more than two pixels per pixel of its side.
  p.min_component_pixels = static_cast<std::uint32_t>(2 * options.min_marker_px);
  p.refine = options.refine_corners;
  p.marker_size = dictionary.marker_size();
  // As OpenCV rounds it.
  p.max_hamming =
      static_cast<int>(dictionary.max_correction_bits() * options.error_correction_rate);
  p.dictionary_count = dictionary.count();
  return p;
}

std::vector<std::uint64_t> rotated_codes(const Dictionary& dictionary) {
  std::vector<std::uint64_t> codes;
  codes.reserve(static_cast<std::size_t>(dictionary.count()) * 4);
  for (int id = 0; id < dictionary.count(); ++id) {
    std::uint64_t code = dictionary.code(id);
    for (int r = 0; r < 4; ++r) {
      codes.push_back(code);
      code = rotate_code(code, dictionary.marker_size());
    }
  }
  return codes;
}

FrameMarkers collect_markers(std::span<const CandidateResult> results,
                             std::uint32_t dropped_candidates) {
  std::vector<Marker> found;
  for (const CandidateResult& result : results) {
    if (result.marker == 0) continue;
    Marker& marker = found.emplace_back(Marker{.id = result.id, .hamming = result.hamming});
    for (int i = 0; i < 4; ++i) marker.corners[i] = {result.corners[i][0], result.corners[i][1]};
  }
  // A marker's quad is clockwise in its own corner order, so `inside` applies as is.
  std::vector<bool> keep(found.size(), true);
  for (std::size_t i = 0; i < found.size(); ++i) {
    const double area_i = area(found[i].corners);
    for (std::size_t j = 0; j < found.size(); ++j) {
      if (i == j || !keep[i] || !keep[j]) continue;
      const double area_j = area(found[j].corners);
      if ((area_i > area_j || (area_i == area_j && i < j)) &&
          inside(found[i].corners, center(found[j].corners))) {
        keep[j] = false;
      }
    }
  }
  FrameMarkers frame;
  for (std::size_t i = 0; i < found.size(); ++i) {
    if (keep[i]) frame.markers.push_back(found[i]);
  }
  std::sort(frame.markers.begin(), frame.markers.end(), [](const Marker& a, const Marker& b) {
    const Point2 ca = center(a.corners);
    const Point2 cb = center(b.corners);
    if (a.id != b.id) return a.id < b.id;
    return ca[1] != cb[1] ? ca[1] < cb[1] : ca[0] < cb[0];
  });
  frame.overflow.candidates = dropped_candidates;
  if (frame.markers.size() > kMaxMarkersPerFrame) {
    frame.overflow.markers = static_cast<std::uint32_t>(frame.markers.size() - kMaxMarkersPerFrame);
    frame.markers.resize(kMaxMarkersPerFrame);
  }
  return frame;
}

CpuDetector::CpuDetector(const DetectorOptions& options, int width, int height)
    : params_(make_params(options, width, height)),
      codes_(rotated_codes(Dictionary(options.dictionary))) {
  const auto tiles = static_cast<std::size_t>(params_.tiles_x) * params_.tiles_y;
  tile_lo_.resize(tiles);
  tile_hi_.resize(tiles);
  near_lo_.resize(tiles);
  near_hi_.resize(tiles);
  labels_.resize(static_cast<std::size_t>(width) * height);
  slots_.resize(labels_.size());
  candidates_.reserve(core::kMaxCandidates);
}

void CpuDetector::check(const GrayImageView& image) const {
  if (image.location != MemoryLocation::kHost || image.width != params_.width ||
      image.height != params_.height || image.data == nullptr) {
    throw std::invalid_argument("cpu backend: the image is not of the configured size and location");
  }
}

FrameMarkers CpuDetector::detect(const GrayImageView& view) {
  check(view);
  const core::Image image{view.data, view.width, view.height, view.pitch};
  const core::Params& p = params_;
  const int width = p.width;
  const int height = p.height;

  // 1. Threshold, labeling every dark pixel as its own component.
  for (int ty = 0; ty < p.tiles_y; ++ty) {
    for (int tx = 0; tx < p.tiles_x; ++tx) {
      const int t = ty * p.tiles_x + tx;
      core::tile_range(image, p.tile_size, tx, ty, tile_lo_[t], tile_hi_[t]);
    }
  }
  for (int ty = 0; ty < p.tiles_y; ++ty) {
    for (int tx = 0; tx < p.tiles_x; ++tx) {
      const int t = ty * p.tiles_x + tx;
      core::neighborhood_range(tile_lo_.data(), tile_hi_.data(), p.tiles_x, p.tiles_y, tx, ty,
                               near_lo_[t], near_hi_[t]);
    }
  }
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* row = view.data + static_cast<std::size_t>(y) * view.pitch;
    const int tile_row = (y / p.tile_size) * p.tiles_x;
    for (int x = 0; x < width; ++x) {
      const int t = tile_row + x / p.tile_size;
      const std::int32_t i = y * width + x;
      labels_[i] = core::is_dark(row[x], near_lo_[t], near_hi_[t], p.min_contrast) ? i : -1;
    }
  }

  // 2. 8-connected components; each root is its component's lowest index, as on the GPU.
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::int32_t i = y * width + x;
      if (labels_[i] < 0) continue;
      if (x > 0 && labels_[i - 1] >= 0) unite(labels_, i, i - 1);
      if (y == 0) continue;
      const std::int32_t up = i - width;
      if (x > 0 && labels_[up - 1] >= 0) unite(labels_, i, up - 1);
      if (labels_[up] >= 0) unite(labels_, i, up);
      if (x + 1 < width && labels_[up + 1] >= 0) unite(labels_, i, up + 1);
    }
  }
  std::fill(slots_.begin(), slots_.end(), 0);
  for (std::int32_t i = 0; i < width * height; ++i) {
    if (labels_[i] < 0) continue;
    labels_[i] = find_root(labels_, i);
    ++slots_[labels_[i]];
  }

  // 3. Candidates: components with enough pixels, then the passes over their pixels.
  candidates_.clear();
  std::uint32_t dropped = 0;
  for (std::int32_t i = 0; i < width * height; ++i) {
    if (labels_[i] != i) continue;
    if (static_cast<std::uint32_t>(slots_[i]) < p.min_component_pixels) {
      slots_[i] = -1;
    } else if (candidates_.size() == core::kMaxCandidates) {
      slots_[i] = -1;
      ++dropped;
    } else {
      slots_[i] = static_cast<std::int32_t>(candidates_.size());
      core::init_candidate(candidates_.emplace_back(), i);
    }
  }
  // Runs `visit(candidate, x, y, index)` for each pixel of a candidate.
  const auto each_pixel = [&](const auto& visit) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const std::int32_t i = y * width + x;
        if (labels_[i] < 0 || slots_[labels_[i]] < 0) continue;
        visit(candidates_[slots_[labels_[i]]], x, y, static_cast<std::uint32_t>(i));
      }
    }
  };
  each_pixel([](Candidate& c, int x, int y, std::uint32_t) {
    ++c.pixels;
    c.sum_x += static_cast<unsigned>(x);
    c.sum_y += static_cast<unsigned>(y);
    c.min_x = std::min(c.min_x, x);
    c.min_y = std::min(c.min_y, y);
    c.max_x = std::max(c.max_x, x);
    c.max_y = std::max(c.max_y, y);
  });
  each_pixel([](Candidate& c, int x, int y, std::uint32_t i) {
    c.far0 = std::max(c.far0, core::far0_key(c, x, y, i));
  });
  each_pixel([width](Candidate& c, int x, int y, std::uint32_t i) {
    c.far2 = std::max(c.far2, core::far2_key(c, width, x, y, i));
  });
  each_pixel([width](Candidate& c, int x, int y, std::uint32_t i) {
    const std::int64_t side = core::side_of(c, width, x, y);
    if (side > 0) {
      c.side_pos = std::max(c.side_pos, core::make_key(static_cast<std::uint32_t>(side), i));
    } else if (side < 0) {
      c.side_neg = std::max(c.side_neg, core::make_key(static_cast<std::uint32_t>(-side), i));
    }
  });

  // 4 and 5. Each candidate's quad, refined and decoded.
  std::vector<CandidateResult> results;
  results.reserve(candidates_.size());
  double scratch[core::kScratch];
  for (const Candidate& c : candidates_) {
    results.push_back(core::examine(core::SerialLanes{}, c, image, p, codes_.data(), scratch));
  }
  return collect_markers(results, dropped);
}

std::vector<std::optional<Point2>> CpuDetector::refine_saddles(
    const GrayImageView& view, std::span<const SaddleSeed> seeds) const {
  check(view);
  const core::Image image{view.data, view.width, view.height, view.pitch};
  std::vector<std::optional<Point2>> refined;
  refined.reserve(seeds.size());
  double scratch[core::kScratch];
  for (const SaddleSeed& seed : seeds) {
    double x = seed.position[0];
    double y = seed.position[1];
    if (core::refine_saddle(core::SerialLanes{}, image, seed.half_window, params_.min_contrast,
                            scratch, x, y)) {
      refined.emplace_back(Point2{x, y});
    } else {
      refined.emplace_back(std::nullopt);
    }
  }
  return refined;
}

}  // namespace holoscan::fiducials
