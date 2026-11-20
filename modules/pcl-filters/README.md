# holoscan-pcl-filters

A Holoscan SDK 5 Module of point-cloud filters on the GPU: the steps between a raw cloud from a depth
camera, a stereo matcher or a lidar and the mapping, obstacle detection or grasping that use it.

- `VoxelDownsampleOp` thins a cloud to one point per voxel: the centroid or the first point, with
  the voxel's mean color, normal and intensity, in a deterministic order.
- `CropBoxOp` keeps the points inside (or outside) an axis-aligned box in a given frame, and
  within a range of distance from that frame's origin. With one axis limited it is a pass-through
  filter, with only the distance a range filter.
- `GroundRemovalOp` finds the ground plane by RANSAC, removes the points on it, and emits the
  plane's coefficients. It can also cut at a fixed height.
- `OutlierRemovalOp` removes points far from their neighbors (statistical) or with too few of them
  (radius), with a grid on the GPU; no FLANN or PCL.

Each takes and sends the [holoscan-messages](../messages/) `PointCloud` message, so they chain with
each other and with any operator that sends or takes one. The [PCL Filters
demo](../../applications/pcl_filters/) runs them all end to end on synthetic clouds.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-pcl-filters)` |
| Targets | `holoscan::pcl_filters` |
| Headers | `<holoscan/pcl_filters/...>` |
| C++ namespace | `holoscan::pcl_filters` |
| Debian package | `holoscan-pcl-filters` |
| Depends on | [holoscan-messages](../messages/) |
| Platforms | x86_64; aarch64 (Jetson Thor and Jetson Orin, with JetPack 7.2) |

## The cloud

| Port | Type | |
| --- | --- | --- |
| `point_cloud` (input) | `holoscan::messages::schema::PointCloudT` | Points float32 `[N, 3]`, or organized `[H, W, 3]`, with any of the optional colors, normals and intensity; each tensor packed, in host or device memory on the operator's GPU. A point with a non-finite coordinate (NaN where a depth camera has no depth) is invalid, and no filter keeps one |
| `filtered_point_cloud` (output) | `holoscan::messages::schema::PointCloudT` | The filtered cloud in device memory, in the configured `layout` (below), with the fields the input has |
| `plane` (output, `GroundRemovalOp` with `emit_plane`) | `holoscan::Tensor` | float32 `[4]` `(a, b, c, d)` in device memory: `a x + b y + c z + d = 0` in metres in the cloud's frame, `(a, b, c)` a unit normal toward `up`; NaN when no plane was found |

The output carries every field the input carries and no other: an absent field stays the messages
module's empty `[0]`. A removed point is NaN, with black, a NaN normal and a NaN intensity. The
output keeps the input's header (frame id and capture time) and its sample's capture time and frame
number; the plane goes out after the cloud, with the same. The input and the output cannot both be
called `point_cloud`: the SDK refuses an operator whose input and output share a name.

The output's layout is each operator's `cloud.layout`:

| `OutputLayout` | Shape | `organized` | `valid_count` | Host waits per frame |
| --- | --- | --- | --- | --- |
| `kKeep` (default) | The input's: an organized cloud stays organized, so pixel neighbors stay neighbors | The input's | -1 | None |
| `kDense` | `[M, 3]`: the `M` kept points in input order | false | `M` | One: the size is known only once the filter has run, so the operator waits for its GPU work before allocating the output |
| `kPadded` | `[N, 3]` for an input of `N` points: the kept points first, in input order, then removed points | false | -1 | None: the output is allocated at the input's size and the padding written on the GPU after the count |

`VoxelDownsampleOp` takes `kDense` (its default) or `kPadded`, as a voxel has no place in the
input's shape.

What each frame costs besides the filter, per operator:

| Input | Copies | Host waits |
| --- | --- | --- |
| Device memory | None: the filter's kernels read the input in place and write the output once | None with `kKeep` and `kPadded`; one with `kDense` (and a 4-byte read of the count) |
| Host memory, pinned or pageable | One host-to-device copy per field, asynchronous | Up to one more: the producer may reuse the memory once `compute()` returns, and CUDA does not promise that an asynchronous copy from pageable memory has read it by then, so the operator waits for the copies unless the memory's allocator can wait for them itself |

The first frame in host memory is logged once, at `INFO`, with the fields and bytes it copies.

`GroundRemovalOp` writes the plane straight into its output. Scratch buffers grow with the largest
cloud seen and are freed when the run stops; growing one waits once for the stream and, as
`cudaFree` does, for the whole GPU.

Every operator's config has a `cloud` (`CloudSettings`), besides its own settings:

| Setting | Default | Meaning |
| --- | --- | --- |
| `layout` | `OutputLayout::kKeep` (`kDense` for `VoxelDownsampleOp`) | As above |
| `max_points` | `2073600` (1920x1080) | Largest cloud accepted, up to 2^30 points; a larger one fails the run. It sizes the output's memory pool |
| `fields` | colors, normals and intensity | The optional fields the output has room for: each takes `max_points` values in every output buffer. A cloud carrying another fails the run, so turn off only what the producer never sends |

A cloud the operator cannot take fails the run: `compute()` returns the error, and the SDK ends the
run. With the defaults, every slot of an operator's output pool takes about 64 MB (31 bytes for each
of the 2073600 points); size `max_points` and `fields` to the producer to keep the pools small.

## Operators

### `VoxelDownsampleOp`

The voxel of a point `p` is `floor(p / voxel_size)` on each axis, within 2^20 voxels of the origin
on each (points beyond are dropped). Each point's voxel key is sorted with its index by a stable
radix sort (CUB), so the points of a voxel come together in input order; one thread per voxel then
sums them in double. Voxels come out in ascending order of their `(x, y, z)` index, `x` slowest. In
both modes the other fields are the voxel's means: colors rounded to the nearest; normals over its
points' finite normals, normalized (NaN when there are none, or they cancel out); intensity over its
points' finite intensities (NaN when there are none).

| Setting | Default | Meaning |
| --- | --- | --- |
| `voxel_size` | `0.05` | Voxel side in metres |
| `mode` | `VoxelMode::kCentroid` | `kCentroid`, the mean of the voxel's points, or `kFirstPoint`, its point with the lowest input index |

### `CropBoxOp`

A point is inside when, after the optional transform into the box frame, `min <= p <= max` on every
axis and its distance from the box frame's origin is within `[min_distance, max_distance]`. The
output points stay in the input frame. A separate pass-through operator would do the same with one
axis, so there is none.

`CropBoxConfig::box` is a `CropBoxParams`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `min`, `max` | `-inf`, `+inf` on each axis | The box, in metres; an axis left infinite is not limited |
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
| `height` | `0` | `kHeight`: the cut, in metres along `up` |

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
| `search_radius` | `0.1` | Statistical: metres within which neighbors are searched |
| `radius` | `0.05` | Radius: the neighborhood in metres |
| `min_neighbors` | `2` | Radius: neighbors a point needs to stay |

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-pcl-filters 1.0 REQUIRED CONFIG)  # Finds holoscan-messages too.
target_link_libraries(my_app PRIVATE holoscan::pcl_filters)
```

```cpp
#include <holoscan/pcl_filters/ground_removal_op.hpp>
#include <holoscan/pcl_filters/voxel_downsample_op.hpp>

namespace pf = holoscan::pcl_filters;

// The camera sends points and colors, up to 640x480 of them.
const pf::CloudSettings cloud{.max_points = 640 * 480, .fields = {.colors = true}};
pf::GroundRemovalConfig ground{.cloud = cloud};
ground.ground.up[0] = 0.0F;  // A camera optical frame, level: up is -y.
ground.ground.up[1] = -1.0F;
ground.ground.up[2] = 0.0F;
ground.ground.distance_threshold = 0.03F;
auto ground_filter = graph.op<pf::GroundRemovalOp>("ground", ground);
auto voxels = graph.op<pf::VoxelDownsampleOp>(
    "voxels", pf::VoxelDownsampleConfig{.voxel_size = 0.02F, .cloud = {.layout = pf::OutputLayout::kDense,
                                                                       .max_points = 640 * 480,
                                                                       .fields = {.colors = true}}});

// Live: a consumer that falls behind gets the newest cloud, and nothing waits.
const holoscan::ConnectionOptions live{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                       .queue_depth = 1U};
graph.add_flow(camera->point_cloud, ground_filter->point_cloud, live);
graph.add_flow(ground_filter->filtered_point_cloud, voxels->point_cloud, live);
graph.add_flow(ground_filter->plane, planner->ground, live);
graph.add_flow(voxels->filtered_point_cloud, mapper->point_cloud, live);
```

The outputs are in device memory, so bind their GPU when compiling, for every output of every
filter:

```cpp
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "voxels", .output_port = "filtered_point_cloud", .device = holoscan::DeviceId{0}});
```

An operator cannot send a refused output again: a lossless (`kBlocking`) edge whose queue is full
fails the run. A live graph, like the one above, connects every edge with
`QueuePolicy::kDropOldest` and depth 1, so that a consumer that falls behind drops old clouds
instead. A graph with a second source, or one that forks or joins (above, the ground filter feeds
the voxel filter and the planner), is split by the SDK's automatic partitioning onto several
workers. Across workers, the SDK 5.0 runs a producer even while its consumer's lossless queue is
full, and refuses what it sends: an emit fails with `kBackpressured`, and an allocation with
`kResourceExhausted` while every slot of the output's pool is still held. A graph that must be
lossless does what the demo does: `graph.partition("name").include_all()` runs the whole graph on
one worker, and the source sends a cloud only once the previous one is through. With both, the demo
and the tests have never had an emit refused on the SDK 5.0.0; the SDK's headers do not promise it
for one worker alone.

The filters need no Holoscan graph: `launch_crop_box()`, `apply_mask()`, `Compactor`, `pad_cloud()`,
`VoxelGrid`, `GroundSegmenter` and `OutlierFilter` in `<holoscan/pcl_filters/filters.hpp>` work on
device pointers (`CloudFields` in, `CloudOutput` out) and a CUDA stream.
`<holoscan/pcl_filters/synthetic_scene.hpp>` is the synthetic depth camera of the tests and the
demo: ground, boxes and a moving sphere, with depth noise, flying pixels and missing depth, every
pixel labeled and the ground plane known exactly.

## Accuracy

The tests compare against CPU references written from the definitions above, on clouds made at
test time:

| Filter | Against | Result |
| --- | --- | --- |
| Crop box, pass-through, distance range, transforms | CPU reference | Every point's decision identical |
| Voxel grid, both modes, with colors, normals and intensity | CPU reference (stable sort, double sums) | Same voxel count, same order, same centroids, normals and intensities bit for bit, same colors |
| Statistical outliers | Brute-force k nearest neighbors | Every mean distance identical, threshold within 1e-9, every decision identical |
| Radius outliers | Brute force | Every decision identical |
| Ground plane, refined | The scene's true plane, pitches 15 to 40 degrees | Within 0.2 degrees and 3 mm (measured: under 0.02 degrees and 0.4 mm); under 0.5 % of the ground left, over 93 % of object points kept |
| Outlier rates on the 640x480 scene | The scene's labels | Statistical (k 8, 2 deviations): 98 % of flying pixels removed, 96 % of surface points kept; radius (3 cm, 3): 98 % removed, 99.3 % kept |

The crop box, voxel grid and outlier kernels are compiled without fused multiply-adds, so they
compute what the CPU computes from the same expressions; host code that checks them exactly must use
`-ffp-contract=off`.

## Performance

Each filter as its operator runs it (the filter, then the output in a layout: `kKeep` writes it,
`kDense` waits for the count and packs, `kPadded` packs and pads), on organized clouds of the
synthetic scene, in milliseconds of GPU time, waits included: the median of five runs of the mean
over 100; `pcl_filters_benchmark` measures it.

| Filter | RTX 5090, 640x480 | RTX 5090, 1280x720 | Jetson Orin, 640x480 | Jetson Orin, 1280x720 |
| --- | --- | --- | --- | --- |
| CropBoxOp (`kKeep`) | 0.009 | 0.018 | 0.100 | 0.282 |
| CropBoxOp (`kDense`) | 0.023 | 0.039 | 0.160 | 0.426 |
| CropBoxOp (`kPadded`) | 0.019 | 0.032 | 0.145 | 0.425 |
| VoxelDownsampleOp (2 cm, `kDense`) | 0.146 | 0.266 | 0.831 | 2.362 |
| VoxelDownsampleOp (2 cm, `kPadded`) | 0.115 | 0.214 | 0.769 | 2.559 |
| VoxelDownsampleOp (5 cm, `kDense`) | 0.243 | 0.699 | 1.158 | 4.008 |
| GroundRemovalOp (RANSAC, 256 iterations) | 0.288 | 0.341 | 0.857 | 1.940 |
| OutlierRemovalOp (statistical, k 8, 10 cm) | 1.567 | 5.726 | 14.357 | 55.910 |
| OutlierRemovalOp (statistical, k 8, 4 cm) | 1.307 | 3.465 | 17.404 | 45.472 |
| OutlierRemovalOp (radius, 5 cm, 3) | 0.302 | 0.831 | 1.380 | 5.179 |

All runs used the repository's CI image (CUDA 13.2, Holoscan SDK 5.0.0). The RTX 5090 had driver
595.91.07. The Jetson Orin had JetPack 7.2 (L4T R39.2.1) with driver 595.78, in power mode MAXN
without `jetson_clocks`, as it was found when checked afterwards, with no reboot in between. On an
idle GPU, `kDense`'s wait for the count costs a few tens of microseconds on the RTX 5090 and up to
about 0.1 ms on the Jetson Orin. On a GPU that another process keeps busy it costs a time slice:
with another process holding the RTX 5090 fully, `kDense` took 2.5 to 3.5 ms where `kPadded` took
what it takes on an idle GPU. `kPadded` also never holds the operator's thread, so the runtime can
queue the next operators' work meanwhile; its cost is a full-size output, which the next operator
reads in full. The statistical mode is the costly one: every flying pixel searches every ring of
cells for its neighbors, one binary search per cell, and a warp waits for its slowest point. For a
real-time pipeline prefer the radius mode, which stops counting at `min_neighbors`; it removes the
same flying pixels here.

## Requirements

- [holoscan-messages](../messages/), installed.
- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler. CUB, header-only, comes from the CCCL found first on the include path: the SDK's
  (CCCL 3.3.3 in SDK 5.0.0), then the CUDA Toolkit's.

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above, after
installing holoscan-messages into `install/` (see its README):

```bash
cmake -S modules/pcl-filters -B build/pcl-filters -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH="$PWD/install"
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
| `pcl_filters_filters` | GPU | The filters without a graph: the scene's render against its host version; the crop box against a CPU reference; masked, packed and padded clouds against a host mask; the voxel grid at four sizes in both modes, with every field, against a CPU reference, its last cell among invalid points, and a voxel of 17.8 million points; the ground plane against the truth at three pitches, the height mode (on the plane, as the host decides), a wall, and bad settings; outliers against brute force, and their rates; determinism; growing clouds on one instance; a failed kernel launch, which ends each filter; empty, single-point, 8.4-million-point and over-2^30-point clouds |
| `pcl_filters_operators` | GPU | The four operators through the SDK's operator test harness on `PointCloud` messages: organized and unorganized clouds in device, pinned and pageable memory, with points only and with every field, in every output layout, field for field against the filters and the CPU references; the header, capture time, frame number, `organized` and `valid_count` of every output; the plane output; empty and all-invalid clouds; rejected clouds (wrong types and shapes, too many points, a field without room) and settings; growing clouds in pageable memory whose copy is made to read the host memory late, which must be done when `compute()` returns; the one-time log of a host-to-device copy; emit errors, a stop's left unchanged |
| `pcl_filters_benchmark` | GPU | Runs the benchmark briefly, so that it keeps working |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a graph that chains all four operators on host-memory `PointCloud` messages of
the synthetic scene with colors and checks the plane and the voxels. Build it with the install prefix on
`CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/pcl-filters package holoscan-pcl-filters --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-pcl-filters_1.0.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)` and
`holoscan-messages (>= 1.0.0)`, so
`apt install ./holoscan-pcl-filters_1.0.0_<arch>.deb` needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. CUB, compiled into the library from the SDK's or the CUDA Toolkit's CCCL, is
BSD-3-Clause; its notice is in [NOTICE](NOTICE), which the package installs.
