// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Writing a triangle mesh as a binary little-endian PLY file.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace holoscan::tsdf::detail {

struct PlyMesh {
  std::vector<float> vertices;           // x, y, z per vertex.
  std::vector<float> normals;            // nx, ny, nz per vertex.
  std::vector<std::uint8_t> colors;      // red, green, blue per vertex, or empty.
  std::vector<std::uint32_t> triangles;  // Three vertex indices per triangle.
};

// @throws std::runtime_error naming the path when it cannot be written.
void write_ply(const std::string& path, const PlyMesh& mesh);

}  // namespace holoscan::tsdf::detail
