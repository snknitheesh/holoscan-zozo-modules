# holoscan-pcl-filters

A Holoscan SDK 5 Module of point-cloud filters on the GPU: the steps between a raw cloud from a depth
camera, a stereo matcher or a lidar and the mapping, obstacle detection or grasping that use it.

- `VoxelDownsampleOp` thins a cloud to one point per voxel: the centroid or the first point, with
  the voxel's mean color, in a deterministic order.
- `CropBoxOp` keeps the points inside (or outside) an axis-aligned box in a given frame, and
  within a range of distance from that frame's origin. With one axis limited it is a pass-through
  filter, with only the distance a range filter.
- `GroundRemovalOp` finds the ground plane by RANSAC, removes the points on it, and emits the
  plane's coefficients. It can also cut at a fixed height.
- `OutlierRemovalOp` removes points far from their neighbors (statistical) or with too few of them
  (radius), with a grid on the GPU; no FLANN or PCL.

The [PCL Filters demo](../../applications/pcl_filters/) runs them all end to end, after
holoscan-point-cloud's `DepthToPointCloudOp`.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-pcl-filters)` |
| Targets | `holoscan::pcl_filters` |
| Headers | `<holoscan/pcl_filters/...>` |
| C++ namespace | `holoscan::pcl_filters` |
| Debian package | `holoscan-pcl-filters` |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

## The cloud

The operators take and give clouds in the layout of
[holoscan-point-cloud](../point-cloud/): `DepthToPointCloudOp`'s outputs connect to them directly,
and their outputs to `PointCloudToDepthOp`. The module does not depend on holoscan-point-cloud; the
layout is the contract.

| Port | Type | |
| --- | --- | --- |
| `points` (input) | `holoscan::Tensor` | float32 `[N, 3]`, or organized `[H, W, 3]`, packed; in host, pinned or device memory on the operator's GPU. A point with a non-finite coordinate (NaN, as `DepthToPointCloudOp` writes) is invalid, and no filter keeps one |
| `colors` (input, with `color`) | `holoscan::Tensor` | uint8 RGB of the points' shape |
| `filtered_points` (output) | `holoscan::Tensor` | float32 in device memory, in the configured `layout` (below) |
| `filtered_colors` (output, with `color`) | `holoscan::Tensor` | uint8 RGB of `filtered_points`' shape; black where a point was removed |
| `plane` (output, `GroundRemovalOp` with `emit_plane`) | `holoscan::Tensor` | float32 `[4]` `(a, b, c, d)` in device memory: `a x + b y + c z + d = 0`, `(a, b, c)` a unit normal toward `up`; NaN when no plane was found |

The output layout is each operator's `layout` setting:

- `OutputLayout::kKeep` (default): the input's shape, removed points NaN. An organized cloud stays
  organized, so pixel neighbors stay neighbors, and nothing waits for the GPU.
- `OutputLayout::kDense`: `[M, 3]` of the kept points in their input order; `M` is the count. The
  size is known only once the filter has run, so the operator waits for its GPU work once per
  frame, about 0.2 ms, before allocating the output.

`VoxelDownsampleOp`'s output is always dense. With colors, an operator runs once per cloud when the
colors with exactly the same capture time have arrived, as `DepthToPointCloudOp` sends them.
Outputs carry the cloud's capture time and frame id; the points go out first, then the colors, then
the plane. An optional port exists only when configured, and an output that exists must be
connected. Inputs in host memory are copied to the GPU through a staging buffer.

Every operator has these settings, besides its own:

| Setting | Default | Meaning |
| --- | --- | --- |
| `color` | `false` | Declare the `colors` input and the `filtered_colors` output |
| `layout` | `OutputLayout::kKeep` | `kKeep` or `kDense`, as above (not for `VoxelDownsampleOp`) |
| `max_points` | `2073600` (1920x1080) | Largest cloud accepted, up to 2^30 points; a larger one fails the frame. It sizes the outputs' memory pools |

## Operators

### `VoxelDownsampleOp`

The voxel of a point `p` is `floor(p / voxel_size)` on each axis, within 2^20 voxels of the origin
on each (points beyond are dropped). Each point's voxel key is sorted with its index by a stable
radix sort (CUB), so the points of a voxel come together in input order; one thread per voxel then
sums them in double. Voxels come out in ascending order of their `(x, y, z)` index, `x` slowest, and
colors are the voxel's mean color, rounded, in both modes.

| Setting | Default | Meaning |
| --- | --- | --- |
| `voxel_size` | `0.05` | Voxel side in meters |
| `mode` | `VoxelMode::kCentroid` | `kCentroid`, the mean of the voxel's points, or `kFirstPoint`, its point with the lowest input index |

### `CropBoxOp`

A point is inside when, after the optional transform into the box frame, `min <= p <= max` on every
axis and its distance from the box frame's origin is within `[min_distance, max_distance]`. The
output points stay in the input frame. A separate pass-through operator would do the same with one
axis, so there is none.

`CropBoxConfig::box` is a `CropBoxParams`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `min`, `max` | `-inf`, `+inf` on each axis | The box, in meters; an axis left infinite is not limited |
| `min_distance`, `max_distance` | `0`, `+inf` | The distance range from the box frame's origin |
| `use_transform`, `transform` | `false`, identity | 3x4 row-major `[R \| t]` from the cloud's frame into the box frame |
| `keep_outside` | `false` | Keep the points outside the region instead |

### `GroundRemovalOp`

RANSAC on the GPU scores every hypothesis at once. One thread per hypothesis draws three valid
points with a SplitMix64 sequence from `seed`, so the same cloud gives the same plane on every run;
a plane more than `max_tilt_deg` from `up` is not ground (so a wall never is). One thread per
hypothesis then counts its inliers over tiles of points in shared memory, and the hypothesis with
the most wins (the lowest index on a tie). With `refine`, the winner is refit to its inliers by
least squares: sums in double reduced in a fixed order, and the smallest eigenvector of their
covariance. Points within `distance_threshold` of the final plane are ground. Nothing waits for the
GPU. With `GroundMode::kHeight` the plane is `up . p = height` and every point at or below it is
ground.

`GroundRemovalConfig::ground` is a `GroundParams`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `mode` | `GroundMode::kRansac` | `kRansac` or `kHeight` |
| `up` | `(0, 0, 1)` | Which way is up in the cloud's frame; for a camera optical frame, `(0, -1, 0)` turned by the camera's pitch |
| `distance_threshold` | `0.05` | Meters from the plane that count as ground |
| `max_iterations` | `256` | Hypotheses, up to 65536 |
| `seed` | `0x5EED` | The hypotheses' random sequence |
| `max_tilt_deg` | `30` | Largest angle between the plane's normal and `up`, 0..90 |
| `refine` | `true` | Refit the best hypothesis to its inliers |
| `height` | `0` | `kHeight`: the cut, in meters along `up` |

and `GroundRemovalConfig::emit_plane` (default `true`) declares the `plane` output.

### `OutlierRemovalOp`

The points are sorted into a uniform grid by cell key, the occupied cells listed with where their
points start, and each point looks up its neighbor cells by binary search.

- Statistical (`OutlierMode::kStatistical`): each point's mean distance to its `k` nearest
  neighbors within `search_radius`. The search visits rings of cells (a quarter of the radius a
  side) outward, and stops once no unvisited point can be nearer, so the neighbors are exact. Points
  whose mean is above `mean + std_multiplier * standard deviation` of all the means are removed, as
  in PCL's `StatisticalOutlierRemoval`. A point with fewer than `k` other points within
  `search_radius` is removed too, and left out of the mean and deviation; choose the radius well
  above the spacing of the points.
- Radius (`OutlierMode::kRadius`): a point with fewer than `min_neighbors` other points within
  `radius` is removed. It stops counting at `min_neighbors`, so it is cheap.

`OutlierRemovalConfig::outliers` is an `OutlierParams`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `mode` | `OutlierMode::kStatistical` | `kStatistical` or `kRadius` |
| `k` | `8` | Statistical: neighbors, 1..64 |
| `std_multiplier` | `1.0` | Statistical: deviations above the mean that are kept |
| `search_radius` | `0.1` | Statistical: meters within which neighbors are searched |
| `radius` | `0.05` | Radius: the neighborhood in meters |
| `min_neighbors` | `2` | Radius: neighbors a point needs to stay |

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-pcl-filters 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::pcl_filters)
```

```cpp
#include <holoscan/pcl_filters/ground_removal_op.hpp>
#include <holoscan/pcl_filters/voxel_downsample_op.hpp>

namespace pf = holoscan::pcl_filters;

pf::GroundRemovalConfig ground{.color = true, .max_points = 640 * 480};
ground.ground.up[0] = 0.0F;  // A camera optical frame, level: up is -y.
ground.ground.up[1] = -1.0F;
ground.ground.up[2] = 0.0F;
ground.ground.distance_threshold = 0.03F;
auto ground_filter = graph.op<pf::GroundRemovalOp>("ground", ground);
auto voxels = graph.op<pf::VoxelDownsampleOp>(
    "voxels", pf::VoxelDownsampleConfig{.voxel_size = 0.02F, .color = true, .max_points = 640 * 480});

graph.add_flow(deproject->point_cloud, ground_filter->points);  // DepthToPointCloudOp
graph.add_flow(deproject->colors, ground_filter->colors);
graph.add_flow(ground_filter->filtered_points, voxels->points);
graph.add_flow(ground_filter->filtered_colors, voxels->colors);
graph.add_flow(ground_filter->plane, planner->ground);
graph.add_flow(voxels->filtered_points, mapper->points);
graph.add_flow(voxels->filtered_colors, mapper->colors);
```

The outputs are in device memory, so bind their GPU when compiling, for every output of every
filter:

```cpp
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "voxels", .output_port = "filtered_points", .device = holoscan::DeviceId{0}});
```

An operator cannot send a refused output again: a lossless (`kBlocking`) edge whose queue is full
fails the frame. Keep the producer at most one frame ahead of the slowest consumer, as the demo's
source does, or connect consumers that only need the newest cloud with `QueuePolicy::kDropOldest`.

The filters need no Holoscan graph: `launch_crop_box()`, `apply_mask()`, `Compactor`, `VoxelGrid`,
`GroundSegmenter` and `OutlierFilter` in `<holoscan/pcl_filters/filters.hpp>` work on device
pointers and a CUDA stream. `<holoscan/pcl_filters/synthetic_scene.hpp>` is the synthetic depth
camera of the tests and the demo: ground, boxes and a moving sphere, with depth noise, flying pixels
and missing depth, every pixel labeled and the ground plane known exactly.

## Accuracy

The tests compare against CPU references written from the definitions above, on clouds made at
test time:

| Filter | Against | Result |
| --- | --- | --- |
| Crop box, pass-through, distance range, transforms | CPU reference | Every point's decision identical |
| Voxel grid, both modes, with colors | CPU reference (stable sort, double sums) | Same voxel count, same order, same centroids bit for bit, same colors |
| Statistical outliers | Brute-force k nearest neighbors | Every mean distance identical, threshold within 1e-9, every decision identical |
| Radius outliers | Brute force | Every decision identical |
| Ground plane, refined | The scene's true plane, pitches 15 to 40 degrees | Within 0.2 degrees and 3 mm (typically 0.01 degrees and 0.5 mm); under 0.5 % of the ground left, over 93 % of object points kept |
| Outlier rates on the 640x480 scene | The scene's labels | Statistical (k 8, 2 deviations): 98 % of flying pixels removed, 96 % of surface points kept; radius (3 cm, 3): 98 % removed, 99.3 % kept |

The crop box, voxel grid and outlier kernels are compiled without fused multiply-adds, so they
compute what the CPU computes from the same expressions; host code that checks them exactly must use
`-ffp-contract=off`.

## Performance

Each filter as its operator runs it (the filter, then an organized output, or for dense outputs
the wait for the count), on organized clouds of the synthetic scene, mean milliseconds of GPU time
over 100 runs; `pcl_filters_benchmark` measures it.

PERFORMANCE_TABLE

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13 (with its CUB), CMake 4.0
  or later and a C++20 compiler.

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/pcl-filters -B build/pcl-filters -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/pcl-filters
ctest --test-dir build/pcl-filters --output-on-failure
cmake --install build/pcl-filters --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_PCL_FILTERS_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests, all on synthetic clouds made at test time:

| Test | Needs | Checks |
| --- | --- | --- |
| `pcl_filters_filters` | GPU | The filters without a graph: the synthetic scene's GPU render against its host version; the crop box (boxes, pass-through, distance range, rigid and affine transforms, keep outside) and both mask outputs against CPU references; the voxel grid at four sizes in both modes, with far points dropped, against a CPU reference; the ground plane against the truth at three pitches, raw and refined, the height mode, a wall that is no ground, and rejected settings; statistical and radius outliers against brute force, and their rates on the scene; determinism across runs and fresh instances; empty, all-invalid and single-point clouds; a 4096x2048 cloud (8.4 million points) |
| `pcl_filters_operators` | GPU | The four operators through the SDK's operator test harness: organized and dense input in host and device memory, with and without colors, both output layouts, against the filters and the CPU references; the plane output; the capture time and frame id of every output; empty and all-invalid clouds; rejected inputs and settings |
| `pcl_filters_benchmark` | GPU | Runs the benchmark briefly, so that it keeps working |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a graph that chains all four operators on host-memory clouds of the synthetic
scene with colors and checks the plane and the voxels. Build it with the install prefix on
`CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/pcl-filters package holoscan-pcl-filters --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-pcl-filters_1.0.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)`, so
`apt install ./holoscan-pcl-filters_1.0.0_<arch>.deb` needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. See [NOTICE](NOTICE).
