# holoscan-normals

A Holoscan SDK 5 Module for the shape of point clouds on the GPU: surface normals, curvature, and
the planes a scene is made of, for grasping, mapping, floor and table detection, or the clean-up
before object segmentation. Its operators take and send the `PointCloud` message of
[holoscan-messages](../messages/).

- `NormalEstimationOp` estimates each point's normal, oriented toward a viewpoint, and its
  curvature, by principal component analysis of the point's neighborhood: a window of the image
  grid for organized clouds, or a radius or k-nearest search over a GPU grid for any cloud. It
  sends the cloud on with the normals in it, its colors and intensities carried through.
- `PlaneSegmentationOp` finds up to `max_planes` planes in a cloud with normals, one after another,
  by RANSAC with normal agreement, and labels every point with its plane. It emits each plane's
  coefficients, centroid and inlier count, and can merge coplanar planes.

The [Normals and Planes demo](../../applications/normals_planes/) runs both end to end after
holoscan-point-cloud's `DepthToPointCloudOp`, and shows the normals and planes in HoloViz.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-normals)` |
| Targets | `holoscan::normals` |
| Headers | `<holoscan/normals/...>` |
| C++ namespace | `holoscan::normals` |
| Debian package | `holoscan-normals` |
| Platforms | x86_64; aarch64 (Jetson Thor and Jetson Orin, JetPack 7.2) |

## The cloud

Both operators take a `PointCloud` message (see
[holoscan-messages](../messages/README.md#pointcloud)) on `point_cloud`: float32 points `[N, 3]`,
or organized `[H, W, 3]`, with optional colors, normals and intensities of the points' shape, in
host, pinned host or device memory on the operator's GPU. A point with a non-finite coordinate
(NaN, as `DepthToPointCloudOp` writes) is invalid. `DepthToPointCloudOp`'s `point_cloud` connects
to `NormalEstimationOp` directly; `PlaneSegmentationOp` needs a cloud with normals, such as
`NormalEstimationOp`'s `point_cloud_with_normals`. Neither needs the camera: the normals face a
viewpoint given in the cloud's frame.

Every output is in device memory and carries the cloud's capture time and frame id (and, for the
cloud, its header). An operator cannot send a refused output again: a refused emit fails the
frame, with an error that names the port. Each operator is configured only by the config passed
to its constructor; there are no parameters to set afterwards, and no Python bindings.

### Waits and copies

What the host waits for and what is copied, per frame, once each operator's scratch has reached
the size of the largest cloud so far (growing it, by at least half, waits for the stream once per
buffer that grows; the first frame allocates without waiting):

| Input memory | Copies to the GPU | Host waits |
| --- | --- | --- |
| Device (`kCudaDevice`) | None: read in place | None (`NormalEstimationOp` with `kDense`: one, for the count) |
| Pinned host (`kPinnedHost`) | Each field the cloud carries, through a staging buffer | None when the memory can be held until the copy's stream is done (`Tensor::set_deallocation_stream()`); else one per field, for its copy |
| Host (`kHost`) | Each field the cloud carries, through a staging buffer | One per field for the producer, and the copy itself, which the CUDA runtime stages from pageable memory before returning |

The first cloud in host or pinned host memory is logged once, with the bytes copied. Send clouds
in device memory, as `DepthToPointCloudOp` does, and nothing is copied.

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
- `NeighborSearch::kRadius`: every point within `radius`. There is no cap on the neighbors: each
  point visits every point in the 27 cells of side `radius` around it, so its time grows with the
  points per radius (see [Performance](#performance)); keep the radius small for dense, near clouds.
- `NeighborSearch::kKnn`: the `k` nearest points within `radius`, exact, ties to the lower index.
- `NeighborSearch::kAuto` (default): the window for `[H, W, 3]`, the k nearest for `[N, 3]`.

The radius and k-nearest searches sort the points into a uniform grid by cell key (a stable CUB
radix sort) and look each neighbor cell up by binary search; no FLANN or PCL.

| Port | Type | |
| --- | --- | --- |
| `point_cloud` (input) | `PointCloud` | The cloud; normals it carries are replaced |
| `point_cloud_with_normals` | `PointCloud` | The input's points, colors and intensities with the normals, in the configured `layout`: `kKeep`, the input's shape, a point without a normal NaN (black, NaN intensity), not counted (`valid_count` -1); `kDense`, `[M, 3]` of the points with a normal, in input order, `valid_count` M |
| `curvature` | `holoscan::Tensor` | float32, one per point of the output cloud (`[H, W]`, `[N]` or `[M]`), NaN where a point has no normal |

The cloud goes out before the curvature. `NormalEstimationConfig::normals` is a `NormalParams`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `search` | `NeighborSearch::kAuto` | `kAuto`, `kWindow`, `kRadius` or `kKnn`, as above |
| `window_radius` | `3` | Window: half its side in pixels, 1..15 |
| `depth_change_factor` | `0.02` | Window: the range step allowed per pixel, as a fraction of the point's range |
| `radius` | `0.05` | Radius, k-nearest: the neighborhood in meters |
| `k` | `24` | K-nearest: neighbors, the point included, 3..64 |
| `min_neighbors` | `5` | Fewer neighbors give no normal; at least 3, and no more than the window's pixels or `k` for the searches that use them |
| `viewpoint` | `(0, 0, 0)` | Normals face this point: the camera, for a camera's cloud |

and:

| Setting | Default | Meaning |
| --- | --- | --- |
| `colors`, `intensity` | `false` | Room in the output for the clouds' colors and intensities, which it then carries through. A cloud that carries one without room fails the frame; a cloud without it leaves it absent |
| `layout` | `OutputLayout::kKeep` | `kKeep` or `kDense`, as above |
| `max_points` | `2073600` (1920x1080) | The largest cloud accepted; it sizes the outputs' pools, whose every slot holds that many points (24.9 MB of points and as much of normals at the default): set it to your clouds' size |

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
renumbered. Every decision is made on the GPU: the host does not wait for one (see
[Waits and copies](#waits-and-copies)).

| Port | Type | |
| --- | --- | --- |
| `point_cloud` (input) | `PointCloud` | A cloud with normals; a point with a non-finite normal is on no plane. A cloud without normals fails the frame |
| `labels` | `holoscan::Tensor` | uint8, one per point (`[H, W]` or `[N]`): plane `k` is `k + 1`; `kLabelNoPlane` (0) on no plane; `kLabelInvalid` (255) for an invalid point or one without a normal |
| `plane_coefficients` | `holoscan::Tensor` | float32 `[max_planes, 4]` `(a, b, c, d)`: `a x + b y + c z + d = 0`, `(a, b, c)` a unit normal toward `viewpoint`; NaN past the planes found |
| `plane_centroids` | `holoscan::Tensor` | float32 `[max_planes, 3]`, the mean of each plane's inliers; NaN past the planes found |
| `plane_inliers` | `holoscan::Tensor` | uint32 `[max_planes]`, each plane's inliers; 0 past the planes found |

The outputs go out in the order above. Planes come in the order found, which is mostly largest
first. The labels and the plane tables are tensors of their own rather than fields of the cloud:
the `PointCloud` message has no field for them.

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
find_package(holoscan-messages 1.0 REQUIRED CONFIG)
find_package(holoscan-point-cloud 2.0 REQUIRED CONFIG)
find_package(holoscan-normals 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::normals holoscan::point_cloud)
```

A depth camera's frames, its camera from a `camera_info` file, the cloud, its normals and its
planes, every frame lossless, on one worker:

```cpp
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/normals/normal_estimation_op.hpp>
#include <holoscan/normals/plane_segmentation_op.hpp>
#include <holoscan/point_cloud/depth_to_point_cloud_op.hpp>

namespace hm = holoscan::messages;
namespace hn = holoscan::normals;
namespace pc = holoscan::point_cloud;

auto camera = graph.op<hm::CameraInfoSourceOp>(
    "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = "depth.yaml"});
auto deproject = graph.op<pc::DepthToPointCloudOp>(
    "deproject", pc::DepthToPointCloudConfig{.max_width = 640, .max_height = 480});
auto normals = graph.op<hn::NormalEstimationOp>(
    "normals", hn::NormalEstimationConfig{.max_points = 640 * 480});
auto planes = graph.op<hn::PlaneSegmentationOp>(
    "planes", hn::PlaneSegmentationConfig{.max_points = 640 * 480});

// `depth_camera` is your camera's operator, sending holoscan::schema::ImageT depth images.
const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
graph.add_flow(depth_camera->depth, deproject->depth, lossless);
graph.add_flow(camera->camera_info, deproject->camera_info, hm::latest_camera_info_connection());
graph.add_flow(deproject->point_cloud, normals->point_cloud, lossless);
graph.add_flow(normals->point_cloud_with_normals, planes->point_cloud, lossless);
// planes->labels, plane_coefficients, plane_centroids and plane_inliers, and
// normals->curvature, to your operators.
graph.partition("my_app").include_all();
```

The outputs are in device memory, so bind their GPU when compiling, for every output of both
operators:

```cpp
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "planes", .output_port = "labels", .device = holoscan::DeviceId{0}});
```

A lossless (`kBlocking`) edge whose queue is full fails the frame, and across workers a stage that
falls behind meets one. Keep a lossless chain on one worker, as above and as the demo does (see
[holoscan-messages](../messages/README.md#run-a-graph-on-one-worker)), with a source that keeps its
frames in flight below the queues' depth (two) and sends a refused frame again; or connect
consumers that only need the newest frame with `QueuePolicy::kDropOldest`, depth 1.

The kernels need no Holoscan graph: `NormalEstimator` in `<holoscan/normals/normals.hpp>` and
`PlaneSegmenter` in `<holoscan/normals/planes.hpp>` work on device pointers and a CUDA stream.
`<holoscan/normals/views.hpp>` colors normals and labels into RGB8 images for display, and
`<holoscan/normals/synthetic_scene.hpp>` is the synthetic depth camera of the tests and the demo: a
room with a floor, two walls, a table with a box on it, a box on the floor and a moving sphere, with
depth noise and missing pixels, every pixel's surface and exact normal known, and every flat
surface's plane. `synthetic_camera_info()` gives its camera as a `CameraInfo`, to write as a
`camera_info` file with `write_camera_info_yaml()`.

The library is `libholoscan_normals.so.1`: its soname is the major version, which a change to the
public headers that breaks source or binary compatibility raises, as the CMake package's
`SameMajorVersion` check expects.

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
the CPU computes from the same expressions. The library compiles its own host code with
`-ffp-contract=off` but does not pass that on: host code of yours that compares the synthetic
scene's host functions with the GPU bit for bit must be compiled with `-ffp-contract=off` too, or
the compiler may fuse its multiply-adds and give other bits. Every result is the same on every run.

On the local recordings (real depth, no ground truth), `normals_real_data` checks the results'
own rules: unit normals facing the camera, curvature within its range, the same bits on a second
run, and each plane's labels within its distance and angle (the reported plane is the refit of its
inliers, so up to 3 % of them may sit past the distance it took them at, none past 2.5 times it).

## Performance

GPU time of each step's kernels, as the operators launch them in the `kKeep` layout, on organized
clouds of the synthetic scene: the mean of 100 runs on one stream, between two CUDA events, in
milliseconds. It leaves out the operators' receive, allocation and emit and any host time;
`normals_benchmark` measures it. The near rows are the same clouds at a quarter of their distances,
16 times the points per area.

| Kernels | RTX 5090, 640x480 | RTX 5090, 1280x720 | Jetson Orin, 640x480 | Jetson Orin, 1280x720 |
| --- | --- | --- | --- | --- |
| Normals (window 7x7) | 0.039 | 0.098 | 0.746 | 2.23 |
| Normals (window 11x11) | 0.071 | 0.183 | 1.42 | 4.22 |
| Normals (radius 4 cm) | 0.337 | 1.50 | 5.11 | 26.6 |
| Normals (k-nearest 24 within 10 cm) | 1.79 | 4.91 | 20.1 | 69.1 |
| Normals (radius 4 cm), near | 2.08 | 12.9 | 28.2 | 246 |
| Normals (k-nearest 24 within 10 cm), near | 4.84 | 22.5 | 40.7 | 235 |
| Planes (8 planes, 256 hypotheses each) | 1.59 | 1.90 | 4.18 | 8.43 |
| Both (window 7x7, then the planes) | 1.63 | 1.99 | 5.00 | 10.6 |

The RTX 5090 ran driver 595.99.02 and the Jetson Orin JetPack 7.2 in power mode MAXN, both with CUDA
13.2 and Holoscan SDK 5.0.0 in the repository's CI image. The RTX 5090 is shared with other work,
which was idle during the runs (a resident inference server held memory at 0-1 % GPU use); the
Jetson Orin had another job's container running, at 0 % GPU load before and after. The window search
is the fast one, and the one to use for a depth camera's organized cloud. The plane search is a
fixed number of small kernels per plane (about a dozen), so its time is mostly the gaps between
launches and grows little with the cloud. The grid searches pay for a sort and for visiting cells.
The radius search has no cap on its neighbors: its time grows with the points per radius, 5 to 9
times for 16 times the points per area here, so keep its radius small for dense, near clouds; the
k-nearest search grows less (2 to 5 times) from a higher start, and suits sparse or uneven clouds
such as a lidar's.

## Requirements

- holoscan-messages 1.0 (installed, or built in the same project before this module).
- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13 (with its CUB), CMake 4.0
  or later and a C++20 compiler.

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above, after
installing holoscan-messages into `install/`:

```bash
cmake -S modules/normals -B build/normals -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/normals
ctest --test-dir build/normals --output-on-failure
cmake --install build/normals --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_NORMALS_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests, on synthetic clouds made at test time but for `normals_real_data`:

| Test | Needs | Checks |
| --- | --- | --- |
| `normals_normals` | GPU | `NormalEstimator`: the scene's GPU render against its host version and its true planes; every search against the analytic normals (median, 95th percentile, orientation) on organized and shuffled unorganized clouds; curvature on the flat surfaces and the sphere, with and without noise; the CPU reference; determinism across runs and fresh instances; scratch that grows by half, and an estimator whose clouds grow giving what a fresh one gives; empty, all-invalid, single-point, collinear and flat clouds; rejected settings, `min_neighbors` beyond the window or `k` among them |
| `normals_planes` | GPU | `PlaneSegmenter`: the floor, both walls and the table top against the truth (angle, offset, label recall and precision) from organized and unorganized clouds; the CPU reference, bit for bit, with and without merging; determinism and another seed; empty, all-invalid, normal-less, single-plane and plane-free clouds; merging two parallel planes; rejected settings |
| `normals_operators` | GPU | Both operators through the SDK's operator test harness, on `PointCloud` messages: organized and unorganized clouds in pinned host and device memory, with and without colors and intensities, both layouts, bit for bit against the kernels run directly; the header, capture time and frame id of every output; empty and all-invalid clouds; refused clouds (bad layouts, too many points, fields without room, no normals) and settings. Through a graph, clouds from a producer's host and pinned host pools; and the holding of pinned host input until its copy is done |
| `normals_real_data` | GPU, the local recordings | The kernels on each clip's first depth images with its `depth.yaml` (see [Accuracy](#accuracy)); runs when `HOLOSCAN_TEST_DATA` names the folder that holds `stereo-camera/`, and is skipped otherwise, as in CI |
| `normals_benchmark` | GPU | Runs the benchmark briefly, so that it keeps working; it measures nothing. Label `benchmark`: `ctest -LE benchmark` leaves it out |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a lossless graph, on one worker, from host-memory clouds of the synthetic scene
with colors through both operators, checking that the floor and the back wall are found and that
the cloud keeps its colors. It fails unless the graph runs on one worker. Build it with the install
prefix on `CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/normals package holoscan-normals --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture (`all-major`: the Jetson Orin runs the sm_80 code, the
Jetson Thor sm_110's) and writes `holoscan-normals_1.0.0_<arch>.deb` to the current directory. It
installs to `/usr` and depends on `holoscan (>= 5.0.0)`, on `holoscan-messages` 1.x, whose types
its ports carry, and on the CUDA runtime's development files (`libcudart.so.13-dev`), which its
CMake package needs, so
`apt install ./holoscan-messages_1.0.0_<arch>.deb ./holoscan-normals_1.0.0_<arch>.deb` needs
NVIDIA's CUDA repository.

## Licenses

Apache-2.0. CUB, whose radix sort and scan are compiled into the library, is BSD-3-Clause; its
license is in [NOTICE](NOTICE), which the package installs.
