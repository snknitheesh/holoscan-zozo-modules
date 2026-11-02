# holoscan-tsdf

A Holoscan SDK 5 Module for dense 3D reconstruction on the GPU: depth images from a camera whose
pose is known (from a robot's kinematics, a tracker or odometry) are fused into a truncated signed
distance (TSDF) volume, which gives back a clean triangle mesh of the scene and depth and normal
images from any viewpoint.

- `TsdfIntegrateOp` integrates each depth image (uint16 millimeters or float32 meters), with its
  aligned color when configured, into the volume from the camera's pose.
- `TsdfMeshOp` ray casts the volume into depth, normal and color images from each pose it
  receives, and every N poses extracts the volume's mesh by marching cubes: vertices, normals,
  colors and triangle indices, with their counts, into fixed-size outputs.
- `TsdfVolume`, shared by the two operators, needs no graph: it integrates, ray casts, resets and
  saves its mesh as a PLY file from plain C++. `MeshExtractor` extracts the mesh into your own
  device buffers.

The [TSDF Reconstruction demo](../../applications/tsdf_reconstruction/) runs both end to end on a
synthetic RGB-D camera circling a known scene, checks every result against the scene's exact
surfaces, and shows the input depth beside the shaded ray cast in HoloViz.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-tsdf)` |
| Targets | `holoscan::tsdf` |
| Headers | `<holoscan/tsdf/...>` |
| C++ namespace | `holoscan::tsdf` |
| Debian package | `holoscan-tsdf` |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

## The volume

A dense grid of `dims[0] x dims[1] x dims[2]` voxels (each 2 to 1024, at most 2^30 in all),
`voxel_size` meters apart, voxel (0, 0, 0) at `origin` in the world. Each voxel holds the signed
distance to the surface along the camera's z axis, divided by the `truncation` distance and clamped
to [-1, 1] (positive in front of the surface), its weight, and with `color` an RGB color. Each depth
image adds weight 1 to the voxels it sees, as a running average, up to `weight_cap`; past the cap
the average keeps following new images with a fixed weight, so a surface that moves is followed.
Voxels more than the truncation distance behind every surface seen are never written.

| Volume | Voxel (over 1 m) | GPU memory | With color |
| --- | --- | --- | --- |
| 128^3 | 7.9 mm | 16 MiB | 24 MiB |
| 256^3 | 3.9 mm | 128 MiB | 192 MiB |
| 512^3 | 2.0 mm | 1 GiB | 1.5 GiB |

The volume is allocated once, when it is made, and never grows: 8 bytes a voxel (a float distance
and a float weight), 12 with color. It covers a fixed region, which suits a workspace of known
size: a table, a bin, a shelf, a room. There is no voxel-block hashing mode; see
[DEVELOPER.md](DEVELOPER.md) for why. The mesh extraction adds scratch of 9 bytes per vertex and
per triangle of its capacity, plus 32 bytes per 256 voxels.

Frames: a pose is world_from_camera, row-major 4x4, for the camera's optical frame (x right, y down,
z forward), as `look_at()` and the synthetic scene make them.

## Operators

Every device output carries the capture time and frame id of the frame it came from. An output
that exists must be connected, and an optional port exists only when configured. Inputs in host
memory are copied to the GPU through a staging buffer, logged once. An operator cannot send a
refused output again: a refused emit fails the frame, with an error that names the port. The two
operators may run on different streams and threads: the volume orders their GPU work with a CUDA
event, so neither waits on the host for the other.

### `TsdfIntegrateOp`

| Port | Type | |
| --- | --- | --- |
| `depth` (input) | `schema::ImageT` | MONO16 (raw units, times `depth_scale`) or MONO32F, `[H, W]` or `[H, W, 1]`, packed, host or device |
| `color` (input, with `color`) | `schema::ImageT` | RGB8, BGR8, RGBA8 or BGRA8 of the depth image's size, aligned with it |
| `pose` (input) | `Tensor` | float32 or float64 `[4, 4]` world_from_camera, host memory (device memory is copied to the host, which waits) |
| `intrinsics` (input, with `intrinsics_input`) | `Tensor` | float32 `[fx, fy, cx, cy]` |
| `integrated_pose` (with `pose_output`) | `Tensor` | float32 `[4, 4]`, host: the pose just integrated, to drive `TsdfMeshOp` |

It runs once per depth image, when the inputs with exactly the same capture time have arrived. Only
the voxels inside the camera's view, out to `depth_max` plus the truncation, are visited, one CUDA
thread per voxel; each voxel projects to its nearest pixel. Depths outside `[depth_min, depth_max]`,
zero and non-finite depths are no measurement.

| `TsdfIntegrateConfig` | Default | |
| --- | --- | --- |
| `volume` | | The shared `TsdfVolume`; required |
| `intrinsics` / `intrinsics_input` | | Pinhole intrinsics for every frame, or the `intrinsics` input; one of the two |
| `depth_scale` | `0.001` | Raw depth to meters: 0.001 for uint16 millimeters, 1 for float32 meters |
| `depth_min`, `depth_max` | `0.1`, `4.0` | The depth range in meters |
| `color` | `false` | Declare `color`; needs a volume with color |
| `pose_output` | `true` | Declare `integrated_pose` |
| `max_width`, `max_height` | `1920`, `1080` | Largest image accepted |

### `TsdfMeshOp`

| Port | Type | |
| --- | --- | --- |
| `pose` (input) | `Tensor` | float32 or float64 `[4, 4]` world_from_camera, host memory: `integrated_pose`, or any pose to look from |
| `vertices` (with `mesh`) | `Tensor` | float32 `[max_vertices, 3]`, world meters |
| `normals` (with `mesh`) | `Tensor` | float32 `[max_vertices, 3]`, unit, pointing out of the surface |
| `colors` (with `mesh`, color volume) | `Tensor` | uint8 `[max_vertices, 3]` |
| `triangles` (with `mesh`) | `Tensor` | uint32 `[max_triangles, 3]`, counterclockwise seen from outside |
| `mesh_counts` (with `mesh`) | `Tensor` | uint32 `[4]`: vertices and triangles written, vertices and triangles the surface needs |
| `raycast_depth` (with `raycast`) | `schema::ImageT` | MONO32F `[H, W]` meters along the camera's z axis, 0 where a ray hits nothing |
| `raycast_normals` (with `raycast`) | `Tensor` | float32 `[H, W, 3]` unit normals in the camera frame, facing the camera; NaN where nothing is hit |
| `raycast_colors` (with `raycast`, color volume) | `schema::ImageT` | RGB8 `[H, W, 3]` |

The ray cast goes out on every pose; the mesh on the `mesh_every`-th, the 2 `mesh_every`-th, and so
on. To mesh on demand, send a pose when a mesh is wanted, with `mesh_every` 1, or call
`MeshExtractor` yourself.

The mesh outputs have a fixed size, so the operator never waits for the GPU to learn the mesh's:
only the first `mesh_counts[0]` vertices and `mesh_counts[1]` triangles are written. When the
surface needs more than the capacity, nothing is written, both counts are 0, and
`mesh_counts[2..3]` say how much it needs. The mesh is indexed: one vertex on each voxel edge the
surface crosses, shared by the triangles around it, so a surface seen all round is closed
(watertight). A cube gives triangles only when all eight of its voxels have been observed. The
order of the vertices and triangles follows the voxels, and the mesh is the same on every run.

Each ray stops at the first crossing from in front of a surface to behind it, between observed
voxels, refined by interpolating the trilinear distance; the normal is the distance's gradient.

| `TsdfMeshConfig` | Default | |
| --- | --- | --- |
| `volume` | | The shared `TsdfVolume`; required |
| `mesh`, `mesh_every` | `true`, `1` | Declare the mesh outputs and mesh every N poses |
| `max_vertices`, `max_triangles` | 2^20, 2^21 | The mesh's capacity; it sizes the outputs and the scratch |
| `raycast`, `raycast_params` | `false` | Declare the ray-cast outputs, and their intrinsics, size and depth range |

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-tsdf 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::tsdf)
```

```cpp
#include <holoscan/tsdf/tsdf_integrate_op.hpp>
#include <holoscan/tsdf/tsdf_mesh_op.hpp>

namespace ht = holoscan::tsdf;

ht::TsdfVolumeConfig config;  // 256^3 voxels of 1 cm around the origin.
config.color = true;
auto volume = std::make_shared<ht::TsdfVolume>(config);

auto integrate = graph.op<ht::TsdfIntegrateOp>(
    "integrate", ht::TsdfIntegrateConfig{.volume = volume, .intrinsics = intrinsics, .color = true});
auto mesh = graph.op<ht::TsdfMeshOp>(
    "mesh", ht::TsdfMeshConfig{.volume = volume, .mesh_every = 30});

graph.add_flow(camera->depth, integrate->depth);
graph.add_flow(camera->color, integrate->color);
graph.add_flow(tracker->pose, integrate->pose);
graph.add_flow(integrate->integrated_pose, mesh->pose);
graph.add_flow(mesh->vertices, planner->vertices);  // And normals, colors, triangles, mesh_counts.

// Later, from any thread:
volume->export_ply("scene.ply");
```

The mesh and ray-cast outputs are in device memory, so bind their GPU when compiling:

```cpp
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "mesh", .output_port = "vertices", .device = holoscan::DeviceId{0}});
```

A lossless (`kBlocking`) edge whose queue is full fails the frame. Keep the camera at most one frame
ahead of the slowest consumer, as the demo's camera does, or connect consumers that only need the
newest frame with `QueuePolicy::kDropOldest`.

Without a graph, on device pointers and a CUDA stream:

```cpp
ht::TsdfVolume volume(config);
volume.integrate(ht::DepthImage{depth, ht::DepthType::kUint16, 640, 480, 0.001F}, nullptr,
                 world_from_camera, ht::IntegrateParams{.intrinsics = k}, stream);
volume.raycast(world_from_camera, ht::RaycastParams{.intrinsics = k},
               ht::RaycastImages{.depth = depth_out, .normals = normals_out}, stream);
ht::MeshExtractor extractor(1 << 20, 1 << 21);
extractor.extract(volume, ht::MeshBuffers{vertices, normals, nullptr, triangles, counts}, stream);
volume.reset(stream);
```

`<holoscan/tsdf/views.hpp>` colors a depth image and shades a ray cast into RGB8 for display, and
`<holoscan/tsdf/synthetic_scene.hpp>` is the synthetic scene of the tests and the demo: a floor, a
box and a sphere (or the sphere alone), rendered into depth and color from any pose with depth
noise, camera orbits around it, and its exact distance and depth functions.

## Accuracy

All on frames rendered at test time, with depth noise of 1 mm plus 2 mm per square meter of depth
(about 3 mm at 1 m), on x86_64; the same tests pass on the Orin.

| What | Against | Result |
| --- | --- | --- |
| The distance's zero crossings, 160^3 (6.3 mm voxels), 60 frames 320x240 round the tabletop | The exact surfaces | 47.7k crossings: mean 0.42 mm (0.07 voxel), 95th percentile 1.06 mm, max 5.1 mm (0.8 voxel) |
| Mesh vertices, the same volume | The exact surfaces | 47.7k vertices, 93.8k triangles: mean 0.42 mm, 95th percentile 1.06 mm, max 5.1 mm; 97.5 % of the normals within 26 degrees of the surfaces'; every triangle wound counterclockwise seen from outside |
| The sphere alone, 96^3 (5.3 mm voxels), 120 frames from all round | Watertightness | 15.5k vertices, 31.0k triangles: every edge in exactly two triangles of opposite direction, Euler characteristic 2, no unused vertex; vertices within 1.5 mm (mean 0.32 mm) |
| The triangle count, 128^3 | After 45 and 60 frames | 52.2k and 52.7k (1.0 % apart); the same mesh when extracted twice |
| Ray-cast depth, 160^3, three poses on and between the orbit's | The exact depth | Over 99.9 % of the surfaces inside the volume seen; mean 1.0 to 1.6 mm (0.15 to 0.26 voxel), 95th percentile 1.5 to 2.1 mm; the rays grazing an edge hit the surface beside it |
| Weight cap 4, a wall moved from 0.50 to 0.53 m, 10 images each | The new wall | The surface at 0.527 m; with a cap of 1000 at 0.515 m, halfway |
| Integration and mesh, 40^3, 5 noisy frames, with and without color | A CPU reference | Identical, bit for bit: distances, weights, colors, vertices, normals and triangles |

The integration, marching cubes and scene kernels are compiled without fused multiply-adds, so they
compute what the CPU computes from the same expressions; host code that checks them exactly must use
`-ffp-contract=off`. Every result is the same on every run.

## Performance

Milliseconds of GPU time, the median of 50 runs, on the tabletop scene after a quarter orbit;
`tsdf_benchmark` measures it. Integration visits only the voxels inside the camera's view; the
mesh extraction reads the whole volume twice; the ray cast marches half a voxel at a time near
surfaces.

| Step | RTX 5090, 640x480 | RTX 5090, 1280x720 | Orin, 640x480 | Orin, 1280x720 |
| --- | --- | --- | --- | --- |
| Integrate, 256^3 | 0.063 | 0.067 | 1.15 | 0.95 |
| Ray cast, 256^3 | 0.22 | 0.36 | 4.09 | 8.52 |
| Mesh, 256^3 (about 120k vertices, 240k triangles) | 0.40 | 0.40 | 8.34 | 7.85 |
| Integrate, 512^3 | 0.50 | 0.42 | 8.88 | 7.01 |
| Ray cast, 512^3 | 1.51 | 0.74 | 11.6 | 15.1 |
| Mesh, 512^3 (about 740k vertices, 1.5M triangles) | 2.95 | 2.90 | 63.5 | 62.1 |

GPU memory: the 256^3 volume 128 MiB, the 512^3 volume 1 GiB (with color 192 MiB and 1.5 GiB); the
benchmark's mesh capacity of 4M vertices and 8M triangles adds 110 to 124 MiB of scratch and 208 MiB
of outputs. The x86_64 machine has an RTX 5090 shared with an inference server holding about 23 GiB
during the runs; the Orin is a Jetson AGX Orin developer kit (64 GB) with JetPack 7.2, also running
other jobs.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13 (with its CUB), CMake 4.0
  or later and a C++20 compiler.
- GPU memory for the volume (see [The volume](#the-volume)).

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/tsdf -B build/tsdf -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/tsdf
ctest --test-dir build/tsdf --output-on-failure
cmake --install build/tsdf --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_TSDF_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests, all on synthetic frames made at test time:

| Test | Needs | Checks |
| --- | --- | --- |
| `tsdf_volume` | GPU | `TsdfVolume` and `MeshExtractor`: the zero crossings, mesh vertices, normals, winding and colors against the exact surfaces; the triangle count settling and repeat extraction; the closed sphere's mesh watertight; the ray cast against the exact depth from poses on and off the orbit; the weight cap and truncation; the CPU reference bit for bit; determinism; empty depth, depths out of range, the camera inside the volume and refused arguments; the capacity and reset; the PLY file; rejected settings |
| `tsdf_operators` | GPU | Both operators through the SDK's operator test harness: uint16 and float32 depth in host and device memory, RGB8 and BGRA8 color, per-frame intrinsics, float32, float64 and device poses, bit for bit against `TsdfVolume`; the mesh every second pose and the ray cast every pose, bit for bit against `MeshExtractor` and `TsdfVolume::raycast()`; the capacity; capture times and frame ids; rejected inputs and settings |
| `tsdf_benchmark` | GPU | Runs the benchmark briefly on 256^3, so that it keeps working |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a graph from host-memory depth images of the synthetic scene through both
operators, checking the mesh against the scene and saving a PLY file. Build it with the install
prefix on `CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/tsdf package holoscan-tsdf --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-tsdf_1.0.0_<arch>.deb` to the current
directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)`, so
`apt install ./holoscan-tsdf_1.0.0_<arch>.deb` needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. The marching cubes triangle table is Paul Bourke's, in the public domain. See
[NOTICE](NOTICE).
