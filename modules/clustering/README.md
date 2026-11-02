# holoscan-clustering

A Holoscan SDK 5 Module that splits point clouds into objects on the GPU. `EuclideanClusterOp`
finds the connected components of points within a tolerance of each other, the Euclidean cluster
extraction of PCL, for clouds from a depth camera, a stereo matcher or a lidar: after ground
removal, each cluster is one object for grasping, obstacle tracking or counting. It labels every
point with its cluster and summarizes each cluster with its point count, centroid, axis-aligned box
and, optionally, an oriented box. The [Clustering demo](../../applications/clustering/) runs it end
to end.

It takes clouds in [holoscan-point-cloud](../point-cloud/)'s layout, so `DepthToPointCloudOp`
connects straight in, but it does not depend on that module.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-clustering)` |
| Targets | `holoscan::clustering` |
| Headers | `<holoscan/clustering/...>` |
| C++ namespace | `holoscan::clustering` |
| Debian package | `holoscan-clustering` |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

## `EuclideanClusterOp`

Two valid points are linked when they are at most `tolerance` apart, and a cluster is a connected
component of these links: every point of a cluster can be reached from every other in steps of at
most `tolerance`. Components with fewer than `min_cluster_size` or more than `max_cluster_size`
points are dropped; of the rest, the `max_clusters` largest are kept. Clusters are numbered 0, 1,
2, ... by decreasing size, ties going to the cluster whose lowest point index is lower; every
other point, invalid ones included, is labeled -1. A point is invalid when a coordinate is not
finite, as `DepthToPointCloudOp` writes NaN for pixels without depth. The result is deterministic:
the same cloud gives the same bits on every run, and the labels are exactly those of a CPU
breadth-first search with the same distances.

| Port | Type | |
| --- | --- | --- |
| `points` (input) | `holoscan::Tensor` | float32 `[N, 3]`, or organized `[H, W, 3]`; in host, pinned or device memory |
| `colors` (input, with `color`) | `holoscan::Tensor` | uint8 RGB of the points' shape |
| `labels` (output) | `holoscan::Tensor` | int32 cluster id per point, -1 for noise and invalid points: `[N]` or `[H, W]` with `OutputLayout::kKeep`; `[M]` with `kDense` |
| `clusters` (output) | `holoscan::Tensor` | float32 `[K, 26]`: one `ClusterSummary` per cluster, by id |
| `clustered_points` (output, with `emit_points`) | `holoscan::Tensor` | float32: the input's shape with unclustered points NaN (`kKeep`), or `[M, 3]` packed by cluster (`kDense`) |
| `clustered_colors` (output, with `color`) | `holoscan::Tensor` | uint8 RGB of the clustered points' shape, black where a point is not clustered |

All outputs are in device memory and carry the cloud's capture time and frame id; they go out in
the order of the table. With `kDense`, the clustered points come grouped by cluster in id order,
in input order within one, so cluster `k`'s rows start at the sum of the point counts of the
clusters before it. The optional ports exist only when configured; with colors, the operator runs
once per cloud when the colors with exactly the same capture time have arrived.

A row of `clusters` (`ClusterSummary` in `<holoscan/clustering/clustering.hpp>`):

| Columns | Field | |
| --- | --- | --- |
| 0 | `id` | The cluster's id, equal to the row |
| 1 | `point_count` | Exact: clouds are limited to 2^24 points |
| 2..4 | `centroid` | The mean of the cluster's points |
| 5..7, 8..10 | `min`, `max` | The axis-aligned bounding box of its points |
| 11..13 | `obb_center` | With `oriented_boxes`: the center of the oriented box |
| 14..22 | `obb_axes` | Its axes, three unit rows from PCA of the points, by decreasing variance, right-handed |
| 23..25 | `obb_half_extents` | Its half sizes along those axes |

The oriented-box fields are NaN without `oriented_boxes`. A cluster whose principal variances are
equal, such as a sphere, has a valid but arbitrary oriented box.

`EuclideanClusterConfig`, passed to the constructor:

| Setting | Default | Meaning |
| --- | --- | --- |
| `clustering.tolerance` | `0.02` | Largest distance between linked points, in the cloud's units (meters) |
| `clustering.min_cluster_size`, `max_cluster_size` | `1`, `2^24` | Clusters with fewer or more points are dropped |
| `clustering.max_clusters` | `1024` | At most this many of the largest clusters are kept, 1..65536 |
| `clustering.connectivity` | `Connectivity::kEuclidean` | `kEuclidean`, any two points within the tolerance; or `kImage`, organized clouds only, the points within the tolerance among a pixel's neighbors |
| `clustering.image_window` | `2` | `kImage`: the neighborhood's radius in pixels, 1..8 (a 5x5 window by default) |
| `oriented_boxes` | `false` | Fill the summaries' oriented boxes, at the cost of one more pass over the points |
| `emit_points` | `false` | Declare the `clustered_points` output |
| `color` | `false` | Declare the `colors` input and `clustered_colors` output; needs `emit_points` |
| `layout` | `OutputLayout::kKeep` | Per-point outputs in the input's shape, or packed by cluster (`kDense`) |
| `max_points` | `1920 * 1080` | Largest cloud accepted, up to 2^24 points; it sizes the outputs |

Inputs must be packed, and points in device memory aligned to 4 bytes and on the operator's GPU.
The operator rejects another element type or shape, colors that do not match the points, a cloud
above `max_points`, and `kImage` with an unorganized cloud, with an error that says so. Clouds in
host memory are copied to the GPU through a staging buffer that grows with the largest cloud seen.

The number of clusters is known only once the GPU has found them, so the operator waits for its GPU
work once per frame for the summaries' shape, once more with `kEuclidean` for the size of the grid,
and once more with `kDense` for the number of clustered points. A refused output cannot be sent
again, as its loan is gone: it fails the frame with an error that names the port. Connect a
consumer that only displays the results with `QueuePolicy::kDropOldest`, so that it never pushes
back.

### How it works

Both connectivities build a graph on the GPU and find its components with a lock-free union-find:
each link hooks the larger root under the smaller with `atomicCAS`, halving paths on the way, so a
component's root is its lowest node whatever the order of the links.

- **`kEuclidean`** sorts the points into a uniform grid of cells slightly smaller than
  `tolerance / sqrt(3)` (a stable CUB radix sort on 63-bit cell keys). Any two points of one cell
  are within the tolerance, so a cell is one node. A point within the tolerance of another lies in a
  cell at most two cells away on each axis, so one thread per pair of such cells (62 per cell)
  tests for a close pair, skipping cells already joined or whose bounding boxes are too far apart,
  and stops at the first pair found. Cells are found by binary search over the sorted keys. Points
  more than 2^20 cells from the origin (about ten kilometers at a 2 cm tolerance) are not
  clustered.
- **`kImage`** links each pixel to the valid pixels of its window that follow it and are within the
  tolerance: no sort and no grid, and every pixel is one node.

The components are counted, filtered by size and sorted by size and lowest point index with CUB,
and the points labeled and grouped by label with a stable radix sort. The summaries are sums over
fixed chunks of 2048 grouped points per block, in double, reduced in a fixed order; the oriented
box comes from Jacobi rotations of the covariance in double and a second pass for the extents.
Distances are computed without fused multiply-adds, exactly as a CPU computes them, so a pair of
points is linked exactly when the CPU reference links it.

### Which connectivity

`kEuclidean` is the default: it is exact for any cloud, organized or not, and on a desktop GPU it
costs about as much as `kImage` (and less on the wall, where every `kImage` link contends for one
root). `kImage` skips the sort and the grid, and on the Orin it is 2 to 3 times faster on the
demo's scene, so it is the choice for organized clouds on a Jetson when milliseconds count. It
links only points that are close in the image as well as in space, though: across an occlusion
edge, or where a grazing surface is sampled sparsely, two points within the tolerance but more than
`image_window` pixels apart stay apart. On the demo's scene the two agree to an adjusted Rand index
above 0.99.

## Accuracy

On clouds made at test time (`clustering_clusterer`), against the truth and a CPU breadth-first
search with the same distances:

- Boxes and spheres on a removed ground, sampled at half the tolerance: touching boxes are one
  cluster, spheres 1.6 tolerances apart are two, the labels equal the true objects exactly and the
  CPU reference's labels, ids included. With more jitter and 1000 isolated noise points the labels
  still equal the reference's, and agree with the truth to an adjusted Rand index above 0.999.
- Random clouds near the percolation density, lattices whose neighbors are exactly one tolerance
  apart, a scene 140 m from the origin, and organized clouds in both connectivities: labels equal
  to the reference's.
- Summaries: point counts and boxes exact, centroids within 2 micrometers of a double-precision
  sum; the oriented box of a rotated box within 0.2 degrees and 1 mm of the box.
- The synthetic depth camera at 640x480 and 1280x720 (the demo's scene): 8 clusters, the true
  objects but the one below the minimum size, adjusted Rand index above 0.997 in both
  connectivities.

## Timings

`clustering_benchmark`, organized clouds of the demo's scene (about 62 000 valid points of 307 200
at 640x480 and 190 000 of 921 600 at 1280x720, 8 objects, tolerance as listed, minimum size 300),
the clusterer as the operator runs it: cluster, write the labels and the summaries, host waits
included. Mean milliseconds of 100 runs (RTX 5090, x86_64) and 50 runs (Jetson AGX Orin, JetPack
7.2), in the CI image.

| Case | RTX 5090 640x480 | RTX 5090 1280x720 | Orin 640x480 | Orin 1280x720 |
| --- | --- | --- | --- | --- |
| Euclidean, 2.5 cm | 0.257 | 0.306 | 1.859 | 3.274 |
| Euclidean, 1 cm | 0.321 | 0.468 | 2.568 | 4.261 |
| Euclidean, 5 cm | 0.238 | 0.866 | 1.700 | 4.602 |
| Euclidean, 2.5 cm, oriented boxes | 0.299 | 0.343 | 1.580 | 3.230 |
| Euclidean, 2.5 cm, `kDense` | 0.257 | 0.310 | 1.598 | 3.651 |
| Image window 2, 2.5 cm | 0.170 | 0.286 | 0.656 | 1.490 |
| Image window 1, 2.5 cm | 0.107 | 0.187 | 0.700 | 1.314 |
| Image window 2, 2.5 cm, oriented boxes | 0.207 | 0.329 | 0.704 | 1.505 |
| Wall filling the image, Euclidean, oriented boxes | 0.337 | 0.712 | 2.179 | 6.191 |
| Wall filling the image, image window 2, oriented boxes | 0.486 | 1.358 | 1.543 | 4.789 |

The wall is the worst case: a tilted plane at 1 to 1.5 m with every pixel valid, one cluster of
all 307 200 or 921 600 points. At 5 cm the two nearly touching spheres merge, hence 7 clusters at
1280x720. Sorting the grid dominates `kEuclidean` on the Orin.

The x86 GPU was shared with other work during the runs.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-clustering 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::clustering)
```

```cpp
#include <holoscan/clustering/euclidean_cluster_op.hpp>

namespace hc = holoscan::clustering;

auto cluster = graph.op<hc::EuclideanClusterOp>(
    "cluster", hc::EuclideanClusterConfig{
                   .clustering = {.tolerance = 0.025F, .min_cluster_size = 100},
                   .oriented_boxes = true,
                   .max_points = 640 * 480});
graph.add_flow(deproject->point_cloud, cluster->points);   // after ground removal
graph.add_flow(cluster->labels, consumer->labels);        // int32 [H, W]
graph.add_flow(cluster->clusters, consumer->clusters);    // float32 [K, 26]
```

The outputs are in device memory, so bind their GPU when compiling:

```cpp
holoscan::CompileOptions options;
for (const char* port : {"labels", "clusters"}) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "cluster", .output_port = port, .device = holoscan::DeviceId{0}});
}
```

The clusterer needs no Holoscan graph: `holoscan::clustering::EuclideanClusterer` in
`<holoscan/clustering/clustering.hpp>`. `<holoscan/clustering/render.hpp>` draws labels in color
and the clusters' boxes into an image, as the demo shows them, and
`<holoscan/clustering/synthetic_scene.hpp>` is the synthetic depth camera of the tests and the
demo, which knows the object each pixel shows. Host code that compares against it exactly must be
compiled with `-ffp-contract=off`.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13 with CUB and libcu++ from
  its toolkit, CMake 4.0 or later and a C++20 compiler.

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/clustering -B build/clustering -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/clustering
ctest --test-dir build/clustering --output-on-failure
cmake --install build/clustering --prefix "$PWD/install"
build/clustering/tests/clustering_benchmark   # the timings above
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_CLUSTERING_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests, all on data made at test time:

| Test | Needs | Checks |
| --- | --- | --- |
| `clustering_clusterer` | GPU | The clusterer against the truth and the CPU reference: boxes and spheres on a removed ground, touching and nearly touching, of a range of sizes, with jitter and noise, near the origin and far from it; random clouds and lattices on the tolerance; organized clouds in both connectivities and the synthetic depth camera at 640x480 and 1280x720; every output against the labels, summaries against a CPU sum, oriented boxes against a rotated box; the size limits and `max_clusters`; determinism across runs, instances, streams and scratch reuse; empty, all-invalid, all-noise, single-point, million-point and out-of-range clouds; rejected settings; and the drawing |
| `clustering_operator` | GPU | `EuclideanClusterOp` through the SDK's operator test harness: organized and unorganized clouds in host and device memory, both layouts, both connectivities, with and without points, colors and oriented boxes, every output against the CPU reference, the capture time and frame id of every output, an empty cloud, a cloud above `max_points`, `kImage` with an unorganized cloud, and the rejection of unusable settings |
| `clustering_benchmark` | GPU | Runs the benchmark briefly, to keep it working |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a small graph that clusters synthetic clouds with colors from host memory,
packed by cluster with oriented boxes, and checks the number of clusters against the truth and
that the labels and summaries agree. Build it with the install prefix on `CMAKE_PREFIX_PATH` and
run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/clustering package holoscan-clustering --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-clustering_1.0.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)`, so
`apt install ./holoscan-clustering_1.0.0_<arch>.deb` needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. CUB and libcu++ (NVIDIA CCCL, Apache-2.0 with LLVM exceptions) come header-only from
the CUDA Toolkit. See [NOTICE](NOTICE).
