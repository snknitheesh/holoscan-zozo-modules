# holoscan-height-map

A Holoscan SDK 5 Module for 2.5D elevation maps on the GPU: where the ground is, how high every
obstacle stands, and where a robot can drive, from a depth camera's or a lidar's point clouds.

- `HeightMapOp` maps point clouds, with the sensor's pose, into a fixed-size grid, fixed in the map
  frame or following the sensor (a moving window). Each cell holds the elevation (the highest or
  mean point of its latest observation, or a variance-weighted Kalman estimate integrated over
  time), its variance, the latest observation's point count and lowest and highest point, and the
  time since it was updated, with height and range limits and a clearing and decay policy.
- `TraversabilityOp` turns the elevation into slope and roughness (a local plane fit), step height
  and a 0 to 1 cost, with an unknown mask.

Both can emit color images of their layers for HoloViz. The
[Height Map demo](../../applications/height_map/) runs both end to end after holoscan-point-cloud's
`DepthToPointCloudOp`, on a synthetic depth camera driving over a course.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-height-map)` |
| Targets | `holoscan::height_map` |
| Headers | `<holoscan/height_map/...>` |
| C++ namespace | `holoscan::height_map` |
| Debian package | `holoscan-height-map` |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

## The grid

A map is `rows` x `cols` square cells of `resolution` meters in the map frame's x-y plane, z up.
Cell `(row, col)` covers x in `[origin_x + col * resolution, ... + resolution)` and y in
`[origin_y + row * resolution, ... + resolution)`: columns run along +x, rows along +y, and
`(origin_x, origin_y)` is the corner of cell (0, 0). Every layer is a `holoscan::Tensor`
`[rows, cols]`, row-major, NaN where the cell is unknown.

With every map goes `map_info`, float64 `[5]` in host memory: `origin_x`, `origin_y`, `resolution`,
`rows`, `cols` (`kInfoOriginX` .. `kInfoCols` in `grid.hpp`). The map's time is the sample's
capture time, which every output carries with the cloud's frame id; the map frame's name is the
operators' `frame_id`, written with the capture time into the color images' headers.

## The cloud

The operators take clouds in the layout of [holoscan-point-cloud](../point-cloud/):
`DepthToPointCloudOp`'s `point_cloud` connects to `HeightMapOp` directly. The module does not depend
on holoscan-point-cloud; the layout is the contract.

| Port | Type | |
| --- | --- | --- |
| `points` (input) | `holoscan::Tensor` | float32 `[N, 3]`, or organized `[H, W, 3]`, packed; in host, pinned or device memory on the operator's GPU, in the sensor's frame. A point with a non-finite coordinate (NaN, as `DepthToPointCloudOp` writes) is invalid |
| `pose` (input, with `pose_input`) | `holoscan::Tensor` | float32 or float64 `[4, 4]`, row-major, in host (preferred) or device memory: the rigid transform from the cloud's frame into the map frame, with the cloud's capture time |

Every layer is in device memory. An output that exists must be connected, and an optional port
exists only when configured. Clouds in host memory are copied to the GPU through a staging buffer,
and a pose in device memory is read back with a wait; each logs once. An operator cannot send a
refused output again: a refused emit fails the frame, with an error that names the port.

## Operators

### `HeightMapOp`

Each cloud, on the GPU, without the host waiting:

1. With `MapMode::kFollow`, the grid moves by whole cells so that the sensor stays in its middle.
   Cells still inside keep what they hold; cells that enter are unknown. The window's corner is
   `rows / 2` and `cols / 2` cells from the cell the sensor is in, on the lattice through
   `origin_x`, `origin_y`. With `kFixed` the grid stays at the configured origin.
2. The map ages. Without `accumulate` every cell is cleared, so that the map is the latest cloud's;
   otherwise a cell not observed for longer than `max_age` seconds is cleared, and with the fused
   elevation every known cell's variance grows by `variance_growth` per second (the process noise),
   and a cell whose variance exceeds `max_variance` is cleared.
3. The valid points within `[min_range, max_range]` of the sensor and `[min_height, max_height]` in
   the map frame are binned by cell, by a stable CUB radix sort, so that every cell sees its points
   in input order. Each cell that received points gets this observation's count, lowest, highest
   and mean height, and spread (the population variance of the heights), one thread per cell.
4. `ElevationMode::kFused` fuses the observation into the cell's estimate, a scalar Kalman filter.
   The measurement is the inverse-variance weighted mean of the points' heights, each point's
   standard deviation `noise_base + noise_quadratic * range^2`, and its variance is
   `1 / sum(1 / variance)` plus the spread, at least `min_variance`. A measurement further than
   `mahalanobis_gate` standard deviations from the estimate is not fused: a higher one replaces the
   estimate (something now stands there); a lower one, which may only have seen past an edge, is
   dropped and adds `rejection_variance` to the estimate's variance, so that lower terrain that
   persists is taken after a few observations.

| Port | Type | |
| --- | --- | --- |
| `elevation` | `holoscan::Tensor` | float32 `[rows, cols]`, meters: the latest observation's highest (`kMax`) or mean (`kMean`) height, or the fused estimate (`kFused`, default) |
| `variance` (`outputs.variance`, default on) | `holoscan::Tensor` | float32, m^2: the fused estimate's variance, or the latest observation's spread |
| `count` (`outputs.count`) | `holoscan::Tensor` | float32, the latest observation's points |
| `min_elevation`, `max_elevation` (`outputs.min_max`) | `holoscan::Tensor` | float32, meters, the latest observation's lowest and highest point |
| `time_since_update` (`outputs.time_since_update`) | `holoscan::Tensor` | float32, seconds since the latest observation |
| `map_info` | `holoscan::Tensor` | float64 `[5]`, host memory: the grid of this map, which moves with `kFollow` |
| `elevation_image` (`outputs.image`) | `holoscan::schema::ImageT` | RGB8 `[rows, cols, 3]`, the elevation in a rainbow from `image_low` (dark blue) to `image_high` (red) meters, or each map's own range when `image_low >= image_high`; unknown cells dark gray |

The operator runs once per cloud; with `pose_input`, when the pose with exactly the same capture time
has arrived. Without it, `sensor_pose` takes every cloud into the map frame. The map's time is the
cloud's capture time, or the activation time when it has none. The outputs go out in the order
above.

`HeightMapConfig::map` is a `HeightMapParams`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `geometry` | 0.05 m, 200 x 200, origin (-5, -5) | `resolution`, `rows`, `cols` (1..8192), `origin_x`, `origin_y`, as above |
| `mode` | `MapMode::kFixed` | `kFixed`, or `kFollow`: the moving window |
| `elevation` | `ElevationMode::kFused` | `kMax`, `kMean` or `kFused` |
| `min_height`, `max_height` | -inf, +inf | Points outside these map-frame heights are dropped |
| `min_range`, `max_range` | 0, +inf | Points outside these distances from the sensor are dropped |
| `noise_base`, `noise_quadratic` | 0.005, 0.002 | A point's height standard deviation, `noise_base + noise_quadratic * range^2` meters |
| `min_variance` | 1e-6 | Floor of a measurement's variance, m^2 |
| `mahalanobis_gate` | 3 | `kFused`: standard deviations beyond which a measurement is not fused; 0 fuses all |
| `rejection_variance` | 1e-4 | `kFused`: variance added for each lower measurement the gate drops, m^2 |
| `accumulate` | `true` | Keep cells between clouds |
| `max_age` | 0 | Clear cells observed longer ago than this, seconds; 0 never |
| `variance_growth` | 0 | `kFused`: variance added per second, m^2/s |
| `max_variance` | 0 | `kFused`: clear cells whose variance exceeds this, m^2; 0 never |

and `pose_input` (`false`), `sensor_pose` (identity), `frame_id` (`"map"`), `outputs`, `image_low`
and `image_high` (`0`, `0`: automatic) and `max_points` (`2073600`, 1920x1080: the largest cloud
accepted).

### `TraversabilityOp`

One thread per cell. For a known cell, the least-squares plane through the known cells of the
`(2 window_radius + 1)^2` window around it, at least `min_valid_fraction` of the window, gives the
slope, `atan(|gradient|)`, and the roughness, the root mean square of the cells' heights about the
plane. The step height is the highest minus the lowest known cell of the `(2 step_radius + 1)^2`
window. The cost is 1 when the slope, the step height or the roughness reaches its maximum;
otherwise it is their weighted mean, each divided by its maximum, so it grows with each of them and
lies in 0..1. A cell whose elevation is unknown, or whose window holds too few known cells, or only
collinear ones, is unknown.

| Port | Type | |
| --- | --- | --- |
| `elevation` (input) | `holoscan::Tensor` | float32 `[rows, cols]`, NaN where unknown, host or device memory, as `HeightMapOp` emits it |
| `map_info` (input) | `holoscan::Tensor` | float64 `[5]` with the elevation's capture time, for the grid |
| `cost` | `holoscan::Tensor` | float32 `[rows, cols]`, 0..1, NaN where unknown |
| `unknown` | `holoscan::Tensor` | uint8 `[rows, cols]`, 1 where the cost is unknown |
| `slope`, `step_height`, `roughness` (with `components`) | `holoscan::Tensor` | float32 `[rows, cols]`: degrees, meters, meters |
| `cost_image` (with `image`) | `holoscan::schema::ImageT` | RGB8 `[rows, cols, 3]`, the cost from green (0) through yellow to red (1); unknown cells dark gray |

The operator runs once per elevation layer, when the `map_info` with exactly the same capture time
has arrived.

`TraversabilityConfig::traversability` is a `TraversabilityParams`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `window_radius` | `2` | The plane fit's window, 1..10 cells around the cell |
| `min_valid_fraction` | `0.5` | The share of that window that must be known |
| `step_radius` | `2` | The step height's window, 1..10 cells around the cell |
| `max_slope_deg`, `max_step`, `max_roughness` | `30`, `0.2`, `0.05` | Where the cost reaches 1: degrees, meters, meters |
| `slope_weight`, `step_weight`, `roughness_weight` | `0.5`, `0.25`, `0.25` | The cost's weights below the maxima |

and `components` (`false`), `image` (`false`), `frame_id` (`"map"`), and `max_rows`, `max_cols`
(`1024`: the largest map accepted).

A mean or fused elevation mixes the points on a riser (a step's or a box's vertical face) into the
cells next to it; a step window two cells wide on either side still reaches clear cells, which is
why `step_radius` is 2. The plane fit across an edge reads a step as a slope too, so a step is
usually lethal by its slope before its height reaches `max_step`, unless `max_slope_deg` is steep.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-height-map 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::height_map)
```

```cpp
#include <holoscan/height_map/height_map_op.hpp>
#include <holoscan/height_map/traversability_op.hpp>

namespace hm = holoscan::height_map;

hm::HeightMapConfig map;
map.map.geometry = {.resolution = 0.04F, .rows = 200, .cols = 200, .origin_x = 0.0, .origin_y = 0.0};
map.map.mode = hm::MapMode::kFollow;
map.pose_input = true;
map.max_points = 640 * 480;
auto mapper = graph.op<hm::HeightMapOp>("height_map", map);
auto traversability = graph.op<hm::TraversabilityOp>(
    "traversability", hm::TraversabilityConfig{.max_rows = 200, .max_cols = 200});

graph.add_flow(deproject->point_cloud, mapper->points);  // DepthToPointCloudOp
graph.add_flow(localization->pose, mapper->pose);        // float64 [4, 4], camera to map
graph.add_flow(mapper->elevation, traversability->elevation);
graph.add_flow(mapper->map_info, traversability->map_info);
graph.add_flow(mapper->variance, planner->variance);
graph.add_flow(mapper->map_info, planner->map_info);
graph.add_flow(traversability->cost, planner->cost);
graph.add_flow(traversability->unknown, planner->unknown);
```

The layers are in device memory, so bind their GPU when compiling, for every layer output of both
operators (`map_info` is in host memory):

```cpp
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "height_map", .output_port = "elevation", .device = holoscan::DeviceId{0}});
```

A lossless (`kBlocking`) edge whose queue is full fails the frame. Keep the producer at most one
frame ahead of the slowest consumer, as the demo's camera does, or connect consumers that only need
the newest map, such as HoloViz, with `QueuePolicy::kDropOldest`.

The kernels need no Holoscan graph: `HeightMapper` in `<holoscan/height_map/height_mapper.hpp>`
and `compute_traversability()` in `<holoscan/height_map/traversability.hpp>` work on device
pointers and a CUDA stream. `<holoscan/height_map/views.hpp>` colors the elevation and the cost
into RGB8 images, and `<holoscan/height_map/synthetic_terrain.hpp>` is the synthetic terrain of
the tests and the demo: flat ground with ramps, steps, boxes and pits whose heights, slopes and
edges are known, and a pinhole depth camera that renders it with depth noise.

## Accuracy

All on clouds made at test time.

| What | Against | Result |
| --- | --- | --- |
| Elevation, mean and fused, of flat ground, a 20 degree ramp, 10 cm steps, a 35 cm box and a 25 cm pit; 5 cm cells, 25 points each, 5 mm of height noise | The analytic height at every cell center more than 4 cm from a height jump (2740 to 3364 cells each) | Median 0.67 mm, 95th percentile 1.9 mm, largest 4.5 mm, on every terrain |
| Highest and lowest point and count, the same terrains without noise | The analytic heights; 25 points per cell | Exact (within 1e-5 m); every count 25; every cell outside the cloud unknown in every layer |
| Elevation from a depth camera over the demo's course, one 320x240 frame, depth noise 1 mm + 0.2 % of depth squared | The analytic height, cells more than 8 cm from a jump (4624 cells) | Median 1.6 mm, 95th percentile 9.2 mm |
| Temporal fusion, ten frames of flat ground, 1 cm noise, 4 points per cell | The truth | Median variance 8.2e-5 to 7.4e-6 m^2, falling every frame; RMS error 5.0 to 1.8 mm |
| Moving window, moved 3 cells in x and -3 in y | The map before the move | Carried cells identical and in place in the map frame; entering cells unknown; a move beyond the window clears it |
| Slope of exact ramps from 2 to 40 degrees | The analytic slope | Within 0.0001 degrees; the cost rises with the slope (0.04 at 2 degrees to 0.62 at 29) and is 1 beyond 30 |
| Slope of a mapped 18 degree ramp, 3 mm noise | The analytic slope | Median error 0.06 degrees, 95th percentile 0.17 |
| Step height at steps of 4, 10, 15 and 25 cm, a box and a pit | The analytic height | Exact (1e-5 m) at every edge; 0 away from them; lethal from `max_step` on |
| Every layer of the map, three elevation modes, fixed and moving windows, rotated sensor frames, noise, invalid and out-of-limit points, ageing, growth and the gate, eight frames | A CPU reference of the mapper | Identical, bit for bit |
| Traversability, three window sizes, noisy terrain with holes | A CPU reference | Unknown mask and step heights and roughness exact; slopes within 1e-4 degrees, costs within 1e-5 |

The mapper, traversability and terrain kernels are compiled without fused multiply-adds, so they
compute what the CPU computes from the same expressions; host code that checks them exactly must
use `-ffp-contract=off`. Every result is the same on every run.

## Performance

A map update is `HeightMapper::update()` and `write_layers()` of the elevation and the variance, as
`HeightMapOp` runs them, on a depth camera's cloud of the demo's course, the window following a
sensor moving 1 cm a frame; mean milliseconds of GPU time over 200 runs (100 on the Orin);
`height_map_benchmark` measures it. Both maps are 10 m square.

| Step | Map | RTX 5090, 640x480 | RTX 5090, 1280x720 | Orin, 640x480 | Orin, 1280x720 |
| --- | --- | --- | --- | --- | --- |
| Map update (fused) | 200x200, 5 cm | 0.058 | 0.117 | 0.296 | 0.881 |
| Map update (fused) | 500x500, 2 cm | 0.054 | 0.080 | 0.414 | 0.968 |
| Map update (highest point) | 200x200 | 0.057 | 0.117 | 0.290 | 0.880 |
| Map update (highest point) | 500x500 | 0.054 | 0.079 | 0.406 | 0.962 |
| Traversability (5x5 plane fit) | 200x200 | 0.009 | 0.008 | 0.030 | 0.029 |
| Traversability (5x5 plane fit) | 500x500 | 0.012 | 0.010 | 0.104 | 0.096 |
| Elevation and cost images | 500x500 | 0.009 | 0.008 | 0.057 | 0.054 |

The x86_64 machine has an RTX 5090; the Orin is a Jetson AGX Orin developer kit (64 GB) with JetPack
7.2 in MAXN. The results, accuracy included, are the same on both. The update is the sort of the points by cell and one pass per
cell; it grows with the points, and with the most crowded cell, whose points one thread sums, which
is why 1280x720 into 5 cm cells costs more than into 2 cm cells. The traversability's time grows
with the known cells.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13 (with its CUB), CMake 4.0
  or later and a C++20 compiler.

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/height-map -B build/height-map -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/height-map
ctest --test-dir build/height-map --output-on-failure
cmake --install build/height-map --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_HEIGHT_MAP_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests, all on synthetic clouds made at test time:

| Test | Needs | Checks |
| --- | --- | --- |
| `height_map_mapper` | GPU | `HeightMapper`: flat ground, a ramp, steps, a box and a pit, with and without noise, per cell against the analytic heights; counts and unknown cells; height and range limits; the moving window; temporal fusion; clearing, ageing, variance growth and the gate (a box appearing, a pit appearing); the CPU reference, bit for bit; determinism with 750k points; a depth camera over the course; empty, all-NaN, infinite, out-of-grid and clipped clouds, a single point; rejected settings |
| `height_map_traversability` | GPU | `compute_traversability()`: flat ground, ramps of known slopes (the cost rising with the slope), steps of known heights, a box and a pit, the unknown mask (holes, a lone cell, collinear cells), a mapped noisy ramp, the CPU reference, determinism; the color views; rejected settings |
| `height_map_operators` | GPU | Both operators through the SDK's operator test harness: organized and unorganized clouds in host and device memory, a configured pose and poses as an input (float32 and float64, host and device), fixed and following windows, every output bit for bit against the kernels run directly; `map_info`; the images and their headers; capture time and frame id; an empty cloud; rejected inputs and settings |
| `height_map_benchmark` | GPU | Runs the benchmark briefly, so that it keeps working |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: host-memory clouds of a box on flat ground with a moving pose, through both
operators, checking the box's height, its lethal rim, the free ground and the moving window. Build
it with the install prefix on `CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/height-map package holoscan-height-map --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-height-map_1.0.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)`, so
`apt install ./holoscan-height-map_1.0.0_<arch>.deb` needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. See [NOTICE](NOTICE).
