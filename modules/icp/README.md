# holoscan-icp

A Holoscan SDK 5 Module for point-cloud registration on the GPU with iterative closest point
(ICP). `IcpOp` finds the rigid transform that aligns one point cloud with another: two scans of the
same scene, a scan with a stored map, or each frame of a depth camera with the frame before it,
which makes it a depth odometry. It is the step that puts 3D data from different times or places
in one frame: mapping, localization against a map, object pose refinement and multi-view fusion
start from it. The [ICP Odometry demo](../../applications/icp_odometry/) runs it end to end on a
depth camera, after holoscan-point-cloud's `DepthToPointCloudOp`.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-icp)` |
| Targets | `holoscan::icp` |
| Headers | `<holoscan/icp/...>` |
| C++ namespace | `holoscan::icp` |
| Debian package | `holoscan-icp` |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

Clouds are float32 `x, y, z` triplets, `[N, 3]` or organized `[H, W, 3]`, the layout
`DepthToPointCloudOp` emits, so the two connect directly; the module does not depend on
holoscan-point-cloud.

## `IcpOp`

| Mode (`IcpConfig::mode`) | Aligns each `source` cloud to |
| --- | --- |
| `kPairs` | The `target` cloud with the same capture time |
| `kMap` | The latest `target` received, which the operator keeps until another one comes: a stored map |
| `kOdometry` | The previous `source` cloud; the transforms accumulate into a pose relative to the first cloud |

Every source cloud gives one result, also when the registration fails, which the statistics say.

| Port | Direction | Type | When | Content |
| --- | --- | --- | --- | --- |
| `source` | in | `Tensor` | always | float32 `[N, 3]` or `[H, W, 3]`, packed, in host, pinned or device memory; non-finite points are skipped |
| `target` | in | `Tensor` | `kPairs`, `kMap` | As `source` |
| `initial_guess` | in | `Tensor` | `initial_guess_input` | float32 or float64 `[4, 4]` row-major rigid transform, host or device, with the source's capture time: the starting point instead of the identity (`kPairs`, `kMap`) or the motion model's guess (`kOdometry`) |
| `transform` | out | `Tensor` | `transform_output` (on) | float64 `[4, 4]` row-major, host: maps source points into the target's frame |
| `stats` | out | `Tensor` | `stats_output` (on) | float64 `[8]`, host: fitness, RMSE, iterations, converged, status, correspondences, source and target points; `result_from()` decodes it |
| `pose` | out | `Tensor` | `kOdometry`, `pose_output` (on) | float64 `[4, 4]` row-major, host: maps the source's points into the first cloud's frame |
| `aligned` | out | `Tensor` | `aligned_output` (off) | The source moved by `transform`, float32 in the source's shape, on the GPU |

Every output carries the source's capture time and frame id. An output exists only when its flag
is set, and one that exists must be connected: an emit into an unconnected output fails the run.

`IcpConfig::options` is an `IcpOptions`:

| Option | Default | |
| --- | --- | --- |
| `method` | `kPointToPlane` | `kPointToPlane` minimizes each source point's distance to its target point's tangent plane; `kPointToPoint`, the distance between the points |
| `max_correspondence_distance` | 0.05 | Points farther apart do not correspond, in the clouds' units (meters) |
| `min_correspondence_distance` | 0 | With a value below the maximum, the distance halves each time the iterations converge, down to this one: coarse to fine. 0 keeps one distance |
| `max_iterations` | 30 | Gauss-Newton updates in all, 1..1000 |
| `translation_epsilon`, `rotation_epsilon` | 1e-5 | Converged when an update moves less than both (meters, radians) |
| `relative_rmse_epsilon` | 1e-6 | ...or when the correspondences stay the same and the RMSE changes less than this fraction |
| `voxel_size` | 0 | Subsamples both clouds to one point per voxel of this edge; 0 keeps every point |
| `normal_radius` | 0 | Neighbors within it give a target point's normal; 0 uses the maximum correspondence distance |
| `robust_kernel`, `robust_scale` | `kNone`, 0.01 | Huber or Tukey weights of the residuals, with this scale |
| `min_correspondences` | 10 | Fewer end the registration unconverged |

`IcpConfig` also has `constant_velocity` (on: in `kOdometry`, each frame's motion is first guessed
as the previous frame's), `max_points` (2^22: the largest cloud), and `map_max_age` (24 h: in
`kMap`, an older target is not taken up).

The statistics: the fitness is the fraction of the source's points (after subsampling) that have a
correspondence at the final transform and the finest distance, and the RMSE their root mean square
distance. `converged` is set only when the iterations converged at the finest distance; the status
says why a registration ended: `kConverged`, `kMaxIterations`, `kTooFewCorrespondences` (also with
no target, or no overlap) or `kDegenerate` (the equations had no solution). In `kOdometry`, a frame
that fails with too few correspondences or degenerate equations keeps the guessed motion; the first
frame has the identity pose and no iterations.

## How it works, and why

- **Nearest neighbors: a sorted uniform grid.** The target's points are sorted by the key of their
  cubic cell (CUB radix sort), its occupied cells are found by run-length encoding, and a query
  scans the 27 cells around it with a binary search per row of three cells. With cells as wide as
  the correspondence distance, the search is exact: every point within that distance is found, the
  nearest wins and the lower index breaks a tie. Unlike a projective association, which looks up the
  pixel a point projects to, it works for any cloud: organized or not, from one camera or several,
  and for a map, and it does not lose correspondences to large motions or occlusion edges. With
  voxel subsampling, a cell holds few points, so a query costs about as little as a projective
  lookup.
- **Normals on the GPU**, for point-to-plane: each target point's neighbors within the normal radius
  (the same grid, or one of that cell size) give a covariance, whose smallest eigenvector, found in
  closed form in double precision, is the normal, turned toward the sensor at the origin. A point
  with fewer than 3 neighbors, or on a line, has none and takes no part.
- **The solver.** One kernel moves each source point by the current transform, finds its
  correspondence, and adds its Jacobian terms to the 6x6 normal equations; `cub::BlockReduce` sums
  each block, and one pass sums the blocks in double precision, in block order. The host solves the
  6x6 system by Cholesky, with a damping of 1e-9 of the largest diagonal that keeps a degenerate
  geometry (one plane) solvable, and updates the transform by the exponential map. Point-to-point
  uses the same equations with three rows per correspondence, so the robust kernels apply to both.
- **Coarse to fine.** A wide correspondence distance catches a large motion; a narrow one keeps
  points the other cloud does not see from pulling on the result. Halving the distance each time the
  iterations converge gives both.
- **Voxel subsampling keeps a point, not a centroid.** Where noise fills a voxel evenly, a centroid
  is the voxel's center, and a subsampled cloud becomes a lattice fixed to its camera; ICP then
  aligns the two lattices instead of the surfaces, which shortened or lengthened each odometry step
  by up to a voxel in tests at 1280x720. Each voxel keeps instead the point whose index hashes
  lowest: a measured point, spread over the voxel like the others.
- **Deterministic.** Sorts are stable, ties go to the lower index, and every sum runs in a fixed
  order that depends only on the point count, so the same input on the same GPU gives the same bits,
  whatever the stream or the instance. A different GPU may round differently.

## Accuracy

On the module's synthetic scenes with time-of-flight noise (0.5 mm plus 0.5 mm per square meter of
depth), 320x240 clouds subsampled to 2 cm, correspondences from the motion's scale down to 3 cm,
x86 with an RTX 5090:

| Test | Point-to-plane | Point-to-point |
| --- | --- | --- |
| Known transforms, 1° to 28° and 3 cm to 54 cm, three scenes with partial overlap; the largest from a guess 4° and 8 cm off | ≤ 0.02°, ≤ 1.0 mm, 13 to 44 iterations | ≤ 0.10°, ≤ 3.0 mm, 102 to 492 iterations |
| Odometry, 150 frames around a closed 4.9 m path | 0.60% drift (30 mm), 0.43°, 13.6 mm ATE | 2.8% drift (139 mm), 1.8° |
| 10% of the source points 0.5 m to 1 m off their surface, Huber or Tukey | ≤ 0.013°, ≤ 1.4 mm | |
| GPU against a CPU reference, 8 iterations | Same correspondences and RMSE; transforms within 0.1 µm | Same |

Point-to-point needs many more iterations and lands farther from the truth on subsampled clouds:
matching samples to samples, it tends to hold back the motion.

## Timings

Depth odometry per frame, point-to-plane, on the synthetic room along its path: subsampling,
normals, grids and every iteration, until the result is on the host. Correspondences from 10 cm
down to 3 cm (4.5 cm with 3 cm voxels), normals within three voxels; the mean over 150 frames.
`icp_benchmark` measures it, and every CI run prints it (in the uploaded CTest log).

| Resolution | x86, RTX 5090, 2 cm voxels | 3 cm | Jetson Orin, 2 cm | 3 cm | Jetson Thor, 2 cm | 3 cm |
| --- | --- | --- | --- | --- | --- | --- |
| 320x240 | 1.7 ms | 1.3 ms | 9.1 ms | 5.8 ms | in CI | in CI |
| 640x480 | 3.6 ms | 1.9 ms | 30.3 ms | 10.6 ms | in CI | in CI |
| 1280x720 | 6.6 ms | 2.4 ms | 63.5 ms | 15.1 ms | in CI | in CI |

The time grows with the points left after subsampling (57 000, 117 000 and 143 000 with 2 cm
voxels; 37 000, 57 000 and 58 000 with 3 cm) and with the iterations a frame needs (18 to 33). The
Orin is 5 to 10 times slower than the RTX 5090; on it, 3 cm voxels make 640x480 and 1280x720 real
time. They cost accuracy when the camera moves less than a voxel per frame, though: around the same
path in 300 frames instead of 150, 3 cm voxels drift 2.6% at 640x480 and 7.7% at 1280x720, against
0.45% and 1.7% for 2 cm.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-icp 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::icp)
```

```cpp
#include <holoscan/icp/icp_op.hpp>

namespace icp = holoscan::icp;

icp::IcpConfig config;
config.mode = icp::IcpMode::kOdometry;
config.options.max_correspondence_distance = 0.10F;
config.options.min_correspondence_distance = 0.03F;
config.options.voxel_size = 0.02F;
config.options.normal_radius = 0.06F;
config.options.max_iterations = 100;
config.transform_output = false;  // Only the pose and the statistics are connected.
auto odometry = graph.op<icp::IcpOp>("odometry", config);
graph.add_flow(deproject->point_cloud, odometry->source);  // Output<holoscan::Tensor>
graph.add_flow(odometry->pose, consumer->pose);            // Input<holoscan::Tensor>
graph.add_flow(odometry->stats, consumer->stats);
```

The `transform`, `stats` and `pose` outputs are in host memory; only `aligned` is on the GPU, and
needs its device bound when compiling
(`CompileOptions::deployment.bind_tensor_output_device`). A source in host memory is copied to the
GPU, which the operator logs once.

The registration needs no Holoscan graph: `IcpRegistration` and `IcpOdometry` in
`<holoscan/icp/registration.hpp>` take clouds in device memory and a CUDA stream, and
`<holoscan/icp/rigid_transform.hpp>` has the transform helpers. For tests and demos,
`<holoscan/icp/synthetic_scene.hpp>` ray casts scenes of planes, boxes and spheres into the depth
image or point cloud of a camera at a known pose, with depth noise, along a closed camera path.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13 with its CUB, CMake 4.0
  or later and a C++20 compiler.

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/icp -B build/icp -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/icp
ctest --test-dir build/icp --output-on-failure
cmake --install build/icp --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_ICP_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests generate all their data:

| Test | Needs | Checks |
| --- | --- | --- |
| `icp_registration` | GPU | Known rigid transforms, small to large, recovered by both methods on three scenes (a room, objects on a floor, a corner) with noise and partial overlap; Huber and Tukey against outliers; failures reported (no overlap, too few iterations, no target, an empty source, a bad guess); bit-identical repeated runs on one instance and on another on another stream; the GPU against the CPU reference on a small cloud; odometry drift around a closed path; option validation |
| `icp_op` | GPU | `IcpOp` through the SDK's operator test harness: pairs from host and device memory, float32 and float64 initial guesses, the aligned cloud against the transform, a stored target, odometry poses against the truth, a failed registration still reported, the capture time on every output, and unusable inputs and settings rejected |
| `icp_benchmark`, `icp_benchmark_coarse` | GPU | Odometry's time per frame at 320x240, 640x480 and 1280x720 with 2 cm and 3 cm voxels, printed; fails only if frames fail to register |
| `icp_host` | none | The rigid transform helpers and the 6x6 solve (also built on their own by the hosted CI job, from [`tests/host/`](tests/host/)) |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a small graph that renders clouds of the synthetic room into device memory,
tracks them with `IcpOp` in odometry mode and checks every pose against the camera's true one.
Build it with the install prefix on `CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/icp package holoscan-icp --language cpp --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-icp_1.0.0_<arch>.deb` to the current
directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)`, so
`apt install ./holoscan-icp_1.0.0_<arch>.deb` needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. The library compiles in templates of CUB, from the CUDA Toolkit (Apache-2.0 with LLVM
Exceptions). See [NOTICE](NOTICE).
