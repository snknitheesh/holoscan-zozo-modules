# holoscan-normals

A Holoscan SDK 5 Module for the shape of point clouds on the GPU: surface normals, curvature, and
the planes a scene is made of, for grasping, mapping, floor and table detection, or the clean-up
before object segmentation.

- `NormalEstimationOp` estimates each point's normal, oriented toward a viewpoint, and its
  curvature, by principal component analysis of the point's neighborhood: a window of the image
  grid for organized clouds, or a radius or k-nearest search over a GPU grid for any cloud.
- `PlaneSegmentationOp` finds up to `max_planes` planes, one after another, by RANSAC with normal
  agreement, and labels every point with its plane. It emits each plane's coefficients, centroid
  and inlier count, and can merge coplanar planes.

The [Normals and Planes demo](../../applications/normals_planes/) runs both end to end after
holoscan-point-cloud's `DepthToPointCloudOp`, and shows the normals and planes in HoloViz.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-normals)` |
| Targets | `holoscan::normals` |
| Headers | `<holoscan/normals/...>` |
| C++ namespace | `holoscan::normals` |
| Debian package | `holoscan-normals` |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

## The cloud

The operators take clouds in the layout of [holoscan-point-cloud](../point-cloud/):
`DepthToPointCloudOp`'s `point_cloud` connects to them directly. The module does not depend on
holoscan-point-cloud; the layout is the contract.

| Port | Type | |
| --- | --- | --- |
| `points` (input) | `holoscan::Tensor` | float32 `[N, 3]`, or organized `[H, W, 3]`, packed; in host, pinned or device memory on the operator's GPU. A point with a non-finite coordinate (NaN, as `DepthToPointCloudOp` writes) is invalid |

Every output is in device memory and carries the cloud's capture time and frame id. An output that
exists must be connected, and an optional port exists only when configured. Inputs in host memory
are copied to the GPU through a staging buffer. An operator cannot send a refused output again: a
refused emit fails the frame, with an error that names the port.

## Operators

### `NormalEstimationOp`

For each valid point it gathers the neighborhood, takes the covariance of the neighbors about their
mean, and solves it by Jacobi rotations: the eigenvector of the smallest eigenvalue is the normal,
flipped to face `viewpoint`, and the curvature is the surface variation
`lambda0 / (lambda0 + lambda1 + lambda2)`, 0 on a plane, as PCL's `NormalEstimation` computes it. A
point with fewer than `min_neighbors` neighbors (itself included), or whose neighbors lie on a line,
has no normal: NaN.

- `NeighborSearch::kWindow` (organized clouds): the `(2 * window_radius + 1)^2` pixels around the
  point. A pixel counts when its range from the viewpoint differs from the point's by at most
  `depth_change_factor` of the point's range per pixel of distance, so the window does not reach
  across depth edges. The window is in pixels: at a higher resolution, widen it to cover the same
  surface (the demo uses 3 at 640x480 and 6 at 1280x720).
- `NeighborSearch::kRadius`: every point within `radius`.
- `NeighborSearch::kKnn`: the `k` nearest points within `radius`, exact, ties to the lower index.
- `NeighborSearch::kAuto` (default): the window for `[H, W, 3]`, the k nearest for `[N, 3]`.

The radius and k-nearest searches sort the points into a uniform grid by cell key (a stable CUB
radix sort) and look each neighbor cell up by binary search; no FLANN or PCL.

| Port | Type | |
| --- | --- | --- |
| `colors` (input, with `color`) | `holoscan::Tensor` | uint8 RGB of the points' shape |
| `filtered_points` | `holoscan::Tensor` | float32 in the configured `layout`: `kKeep`, the input's shape, points without a normal NaN; `kDense`, `[M, 3]` of the points with a normal, in input order |
| `filtered_colors` (with `color`) | `holoscan::Tensor` | uint8 RGB of `filtered_points`' shape, black where a point has no normal |
| `normals` | `holoscan::Tensor` | float32 unit normals of `filtered_points`' shape, NaN where a point has none |
| `curvature` | `holoscan::Tensor` | float32, one per point of `filtered_points` (`[H, W]`, `[N]` or `[M]`), NaN where a point has no normal |

`kKeep` never waits for the GPU; `kDense` waits once per frame for the count before allocating its
outputs. With colors, the operator runs once per cloud when the colors with exactly the same capture
time have arrived.

`NormalEstimationConfig::normals` is a `NormalParams`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `search` | `NeighborSearch::kAuto` | `kAuto`, `kWindow`, `kRadius` or `kKnn`, as above |
| `window_radius` | `3` | Window: half its side in pixels, 1..15 |
| `depth_change_factor` | `0.02` | Window: the range step allowed per pixel, as a fraction of the point's range |
| `radius` | `0.05` | Radius, k-nearest: the neighborhood in meters |
| `k` | `24` | K-nearest: neighbors, the point included, 3..64 |
| `min_neighbors` | `5` | Fewer neighbors give no normal, at least 3 |
| `viewpoint` | `(0, 0, 0)` | Normals face this point: the camera, for a camera's cloud |

and `color` (`false`), `layout` (`OutputLayout::kKeep`) and `max_points` (`2073600`, 1920x1080:
the largest cloud accepted, which sizes the outputs).

### `PlaneSegmentationOp`

For each plane, every hypothesis is drawn at once, one GPU thread each: a random unassigned point
and two more whose normals agree with its own, from a SplitMix64 sequence of `seed`, so the same
cloud gives the same planes on every run. A hypothesis whose plane disagrees with any of its three
normals is dropped. Each is scored on an evenly spread sample of `samples` unassigned points,
counting a point within `distance_threshold` of the plane whose normal is within
`max_normal_angle_deg` of the plane's (either way round). The best hypothesis (the lowest index on a
tie) is refit to all its inliers by least squares, `refine_iterations` times (sums in double reduced
in a fixed order, and the smallest eigenvector of their covariance), its inliers get its label, and
the plane reported is the least-squares plane of exactly those inliers. The next plane is searched
among the points left; a plane with fewer than `min_inliers` ends the search. With
`merge_coplanar`, a plane whose normal is within `merge_angle_deg` of an earlier plane's, with each
centroid within `merge_distance` of the other's plane, is merged into it, refit, and the planes are
renumbered. Everything is decided on the GPU: the host never waits.

| Port | Type | |
| --- | --- | --- |
| `normals` (input) | `holoscan::Tensor` | float32 of the points' shape, as `NormalEstimationOp` emits them; a point with a non-finite normal is on no plane |
| `labels` | `holoscan::Tensor` | uint8, one per point (`[H, W]` or `[N]`): plane `k` is `k + 1`; `kLabelNoPlane` (0) on no plane; `kLabelInvalid` (255) for an invalid point or one without a normal |
| `plane_coefficients` | `holoscan::Tensor` | float32 `[max_planes, 4]` `(a, b, c, d)`: `a x + b y + c z + d = 0`, `(a, b, c)` a unit normal toward `viewpoint`; NaN past the planes found |
| `plane_centroids` | `holoscan::Tensor` | float32 `[max_planes, 3]`, the mean of each plane's inliers; NaN past the planes found |
| `plane_inliers` | `holoscan::Tensor` | uint32 `[max_planes]`, each plane's inliers; 0 past the planes found |

The operator runs once per cloud when the normals with exactly the same capture time have arrived.
Planes come in the order found, which is mostly largest first.

`PlaneSegmentationConfig::planes` is a `PlaneParams`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_planes` | `8` | Planes to find, 1..64 |
| `distance_threshold` | `0.02` | Meters from a plane that are on it |
| `max_normal_angle_deg` | `20` | Largest angle between a point's normal and the plane's, 0..90 |
| `iterations` | `256` | Hypotheses per plane, up to 16384 |
| `samples` | `8192` | Unassigned points each hypothesis is scored on, up to 65536 |
| `min_inliers` | `500` | A plane needs this many points, at least 3 |
| `seed` | `0x5EED` | The hypotheses' random sequence |
| `refine_iterations` | `2` | Least-squares refits per plane, 0..8 |
| `viewpoint` | `(0, 0, 0)` | Plane normals face this point |
| `merge_coplanar` | `false` | Merge coplanar planes |
| `merge_angle_deg`, `merge_distance` | `3`, `0.03` | When two planes are coplanar |

and `max_points` (`2073600`).

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-normals 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::normals)
```

```cpp
#include <holoscan/normals/normal_estimation_op.hpp>
#include <holoscan/normals/plane_segmentation_op.hpp>

namespace hn = holoscan::normals;

auto normals = graph.op<hn::NormalEstimationOp>(
    "normals", hn::NormalEstimationConfig{.max_points = 640 * 480});
auto planes = graph.op<hn::PlaneSegmentationOp>(
    "planes", hn::PlaneSegmentationConfig{.max_points = 640 * 480});

graph.add_flow(deproject->point_cloud, normals->points);  // DepthToPointCloudOp
graph.add_flow(normals->filtered_points, planes->points);
graph.add_flow(normals->normals, planes->normals);
graph.add_flow(normals->curvature, grasping->curvature);
graph.add_flow(planes->labels, segmenter->labels);
graph.add_flow(planes->plane_coefficients, planner->planes);
graph.add_flow(planes->plane_centroids, planner->centroids);
graph.add_flow(planes->plane_inliers, planner->inliers);
```

The outputs are in device memory, so bind their GPU when compiling, for every output of both
operators:

```cpp
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "planes", .output_port = "labels", .device = holoscan::DeviceId{0}});
```

A lossless (`kBlocking`) edge whose queue is full fails the frame. Keep the producer at most one
frame ahead of the slowest consumer, as the demo's source does, or connect consumers that only need
the newest frame with `QueuePolicy::kDropOldest`.

The kernels need no Holoscan graph: `NormalEstimator` in `<holoscan/normals/normals.hpp>` and
`PlaneSegmenter` in `<holoscan/normals/planes.hpp>` work on device pointers and a CUDA stream.
`<holoscan/normals/views.hpp>` colors normals and labels into RGB8 images for display, and
`<holoscan/normals/synthetic_scene.hpp>` is the synthetic depth camera of the tests and the demo: a
room with a floor, two walls, a table with a box on it, a box on the floor and a moving sphere, with
depth noise and missing pixels, every pixel's surface and exact normal known, and every flat
surface's plane.

## Accuracy

All on clouds made at test time; the scene has 2 mm of depth noise and 1 % of pixels without depth.
"Inside" a surface means the neighborhood sees only that surface.

| What | Against | Result |
| --- | --- | --- |
| Normals, window 7x7, 640x480 | The scene's analytic normals, inside each surface (281k points) | Median 1.09 degrees, 95th percentile 3.37; none facing away; every valid point has a normal |
| Normals, window 11x11, 640x480 | The same | Median 0.45 degrees, 95th percentile 1.42 |
| Normals, k-nearest 24, shuffled `[N, 3]` | The same, where the rows are close enough on the surface for a neighborhood | Median 2.63 degrees, 95th percentile 6.31 |
| Normals, radius 4 cm, shuffled `[N, 3]` | The same | Median 0.49 degrees, 95th percentile 1.56 |
| Curvature, window 11x11 | Flat surfaces and the sphere | Without noise: flat 95th percentile 1.1e-7, sphere median 8.7e-4; with 1 mm noise: flat median 5.8e-4, sphere median 2.1e-3 |
| Normals, every search | A CPU reference (the same neighbors, covariance and eigenproblem in double) on small clouds | The same points have normals; within 0.025 degrees and 2e-7 in curvature, same orientation |
| Floor, back wall, side wall, table top, 640x480 | Their true planes, from organized and unorganized clouds, two frames, two seeds | Within 0.016 degrees and 0.35 mm; over 98 % of each surface's points carry its label; over 99.7 % of a plane's labels are on it for the floor and walls, over 97 % for the table top (its rim, whose normals lean toward it) |
| Planes, labels, centroids, inliers | A CPU reference of the whole algorithm, with and without merging | Identical, bit for bit |

The normals, plane and scene kernels are compiled without fused multiply-adds, so they compute what
the CPU computes from the same expressions; host code that checks them exactly must use
`-ffp-contract=off`. Every result is the same on every run.

## Performance

Each step as its operator runs it in the `kKeep` layout, on organized clouds of the synthetic scene,
mean milliseconds of GPU time over 100 runs; `normals_benchmark` measures it.

| Step | RTX 5090, 640x480 | RTX 5090, 1280x720 | Orin, 640x480 | Orin, 1280x720 |
| --- | --- | --- | --- | --- |
| NormalEstimationOp (window 7x7) | 0.039 | 0.098 | 0.81 | 2.16 |
| NormalEstimationOp (window 11x11) | 0.071 | 0.184 | 2.11 | 4.14 |
| NormalEstimationOp (radius 4 cm) | 0.337 | 1.497 | 4.58 | 26.5 |
| NormalEstimationOp (k-nearest 24 within 10 cm) | 1.807 | 4.910 | 18.7 | 63.2 |
| PlaneSegmentationOp (8 planes, 256 hypotheses each) | 1.587 | 1.897 | 4.07 | 8.46 |
| Both (window 7x7, then the planes) | 1.623 | 1.989 | 4.80 | 10.6 |

The x86_64 machine has an RTX 5090, shared with other work during the runs; the Orin is a Jetson AGX
Orin developer kit (64 GB) with JetPack 7.2, in the power mode it was set to, also running other
jobs. The window search is the fast one, and the one to use for a depth camera's organized cloud.
The plane search is a fixed number of small kernels per plane (about a dozen), so its time grows
little with the cloud. The grid searches pay for a sort and for visiting cells: prefer the radius
search when a cloud is unorganized and dense, and keep the k-nearest search for sparse or uneven
clouds such as lidar's.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13 (with its CUB), CMake 4.0
  or later and a C++20 compiler.

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/normals -B build/normals -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/normals
ctest --test-dir build/normals --output-on-failure
cmake --install build/normals --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_NORMALS_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests, all on synthetic clouds made at test time:

| Test | Needs | Checks |
| --- | --- | --- |
| `normals_normals` | GPU | `NormalEstimator`: the scene's GPU render against its host version and its true planes; every search against the analytic normals (median, 95th percentile, orientation) on organized and shuffled unorganized clouds; curvature on the flat surfaces and the sphere, with and without noise; the CPU reference; determinism across runs and fresh instances; empty, all-invalid, single-point, collinear and flat clouds; rejected settings |
| `normals_planes` | GPU | `PlaneSegmenter`: the floor, both walls and the table top against the truth (angle, offset, label recall and precision) from organized and unorganized clouds; the CPU reference, bit for bit, with and without merging; determinism and another seed; empty, all-invalid, normal-less, single-plane and plane-free clouds; merging two parallel planes; rejected settings |
| `normals_operators` | GPU | Both operators through the SDK's operator test harness: organized and unorganized input in host and device memory, with and without colors, both layouts, bit for bit against the kernels run directly; the capture time and frame id of every output; empty and all-invalid clouds; rejected inputs and settings |
| `normals_benchmark` | GPU | Runs the benchmark briefly, so that it keeps working |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a graph from host-memory clouds of the synthetic scene with colors through both
operators, checking that the floor and the back wall are found. Build it with the install prefix on
`CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/normals package holoscan-normals --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-normals_1.0.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)`, so
`apt install ./holoscan-normals_1.0.0_<arch>.deb` needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. See [NOTICE](NOTICE).
