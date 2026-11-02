// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "ply.hpp"

#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace holoscan::tsdf::detail {

static_assert(std::endian::native == std::endian::little,
              "the PLY writer writes the host's bytes as little-endian");

void write_ply(const std::string& path, const PlyMesh& mesh) {
  const std::size_t vertices = mesh.vertices.size() / 3U;
  const std::size_t triangles = mesh.triangles.size() / 3U;
  const bool color = !mesh.colors.empty();
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    throw std::runtime_error("cannot open " + path + " for writing: " + std::strerror(errno));
  }
  file << "ply\nformat binary_little_endian 1.0\ncomment holoscan-tsdf mesh\n"
       << "element vertex " << vertices << '\n'
       << "property float x\nproperty float y\nproperty float z\n"
       << "property float nx\nproperty float ny\nproperty float nz\n";
  if (color) {
    file << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  }
  file << "element face " << triangles << '\n'
       << "property list uchar uint vertex_indices\nend_header\n";
  for (std::size_t i = 0; i < vertices; ++i) {
    file.write(reinterpret_cast<const char*>(&mesh.vertices[3 * i]), 12);
    file.write(reinterpret_cast<const char*>(&mesh.normals[3 * i]), 12);
    if (color) {
      file.write(reinterpret_cast<const char*>(&mesh.colors[3 * i]), 3);
    }
  }
  const char three = 3;
  for (std::size_t i = 0; i < triangles; ++i) {
    file.write(&three, 1);
    file.write(reinterpret_cast<const char*>(&mesh.triangles[3 * i]), 12);
  }
  file.close();
  if (!file) {
    throw std::runtime_error("writing " + path + " failed");
  }
}

}  // namespace holoscan::tsdf::detail
