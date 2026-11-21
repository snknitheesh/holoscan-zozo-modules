# holoscan-clustering

A Holoscan SDK 5 Module that splits point clouds into objects on the GPU. `EuclideanClusterOp`
finds the connected components of points within a tolerance of each other, the Euclidean cluster
extraction of PCL, for clouds from a depth camera, a stereo matcher or a lidar: after ground
removal, each cluster is one object for grasping, obstacle tracking or counting. It labels every
point with its cluster and summarizes each cluster with its point count, centroid, axis-aligned box
and, optionally, an oriented box. The [Clustering demo](../../applications/clustering/) runs it end
to end.

It takes and sends the `PointCloud` message of [holoscan-messages](../messages/), so
holoscan-point-cloud's `DepthToPointCloudOp`, or any other producer of that message, connects
straight in.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-clustering)` |
| Targets | `holoscan::clustering` |
| Headers | `<holoscan/clustering/...>` |
| C++ namespace | `holoscan::clustering` |
| Debian package | `holoscan-clustering` |
| Module dependencies | [holoscan-messages](../messages/) 1.x |
| Platforms | x86_64; aarch64 (Jetson Thor and Jetson Orin, JetPack 7.2) |

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
| `point_cloud` (input) | `messages::schema::PointCloudT` | Points float32 `[N, 3]`, or organized `[H, W, 3]`, with optional colors, normals and intensities; in host, pinned host or device memory |
| `labels` (output) | `holoscan::Tensor` | int32 cluster id per point, -1 for noise and invalid points: `[N]` or `[H, W]` with `OutputLayout::kKeep`; `[M]` with `kDense` |
| `clusters` (output) | `holoscan::Tensor` | float32 `[K, 26]`: one `ClusterSummary` per cluster, by id |
| `clustered_point_cloud` (output, with `emit_points`) | `messages::schema::PointCloudT` | The input's shape with unclustered points NaN (`kKeep`), or an unorganized cloud of the `M` clustered points, packed by cluster (`kDense`); the input's colors, normals and intensities that `fields` names, when the input has them |

The operator runs once per cloud. All outputs are in device memory and carry the cloud's capture
time and frame id, and the clustered cloud its header; they go out in the order of the table. With
`kDense`, the clustered points come grouped by cluster in id order, in input order within one, so
cluster `k`'s points start at the sum of the point counts of the clusters before it, and the cloud's
`valid_count` is `M`; with `kKeep` it is -1 (not counted). Where a point is not clustered, `kKeep`
gives it NaN coordinates, normal and intensity and a black color. The optional output exists only
when configured.

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
| `emit_points` | `false` | Declare the `clustered_point_cloud` output |
| `fields` | none | The optional fields (`colors`, `normals`, `intensity`) the clustered cloud carries when the input has them; they size its pools. Needs `emit_points` |
| `layout` | `OutputLayout::kKeep` | Per-point outputs in the input's shape, or packed by cluster (`kDense`) |
| `max_points` | `1920 * 1080` | Largest cloud accepted, up to 2^24 points; it sizes the outputs |

The operator checks each cloud with the messages module's `check_point_cloud()` (every tensor
there, packed, of the right shapes and element types, its fields in the points' memory), and
rejects points in device memory not aligned to 4 bytes or on another GPU, a cloud above
`max_points`, and `kImage` with an unorganized cloud, with an error that says so. Clouds in host
memory are copied to the GPU through staging buffers that grow with the largest cloud seen; the
operator logs that once.

A refused output cannot be sent again, as its loan is gone: it fails the frame with an error that
names the port. Keep a lossless chain on one worker (see the messages module's
[README](../messages/README.md#run-a-graph-on-one-worker)), and connect a consumer that only
displays the results with `QueuePolicy::kDropOldest` and depth 1, so that it never pushes back.
The input's two slots (`queue_depth(2)`, one cloud in `compute()` while the next arrives) are the
operator's default; an edge's `ConnectionOptions::queue_depth` replaces them.

### CPU waits and copies per frame

| Where | Waits | Copies |
| --- | --- | --- |
| `EuclideanClusterOp`, device input | 1: the number of clusters, which shapes the summaries, read with the number of their points | 1 device-to-device: the labels from the clusterer's scratch into the output (`kKeep`) |
| ... with `kDense` | None more, unless more components pass the size filter than `max_clusters` keeps: then 1 more, for the points of those kept | None more |
| ... with host or pinned host input | 1 more for pageable host memory: the host waits for the producer before copying | 1 host-to-device per field read (points, and colors, normals and intensities when carried) |
| The [demo](../../applications/clustering/)'s reporter | 3: one per copy | 3 device-to-host: the cloud, the labels and the summaries (none for the summaries when there is no cluster) |

Scratch memory grows in stream order (`cudaMallocAsync` on the operator's stream, from its GPU's
memory pool, by at least half each time), so a growing cloud costs no wait either.

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

`kEuclidean` is the default: it is exact for any cloud, organized or not, and on an RTX 5090 it
costs about as much as `kImage` (and less on the wall, where every `kImage` link contends for one
root). `kImage` skips the sort and the grid, and on the Jetson Orin it is about twice as fast on
the demo's scene, so it is the choice for organized clouds on a Jetson Orin when milliseconds count. It
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

On the local recordings (`clustering_real_data`, depth images of a static indoor scene with real
noise, holes and flying pixels, at 848x480 and 1280x720): labels equal to the CPU reference's in
both connectivities.

## Timings

`clustering_benchmark` times the clusterer as the operator runs it: cluster, write the labels and
the summaries, the host's wait included. That leaves out the operator's input staging (none for a
cloud in device memory) and its output loans, which the table
[above](#cpu-waits-and-copies-per-frame) counts. Its clouds are organized clouds of the demo's
scene: about 62 000 valid points of 307 200 at 640x480 and 190 000 of 921 600 at 1280x720, 8
objects, the tolerance as listed, minimum size 300. Mean milliseconds of 100 runs on an RTX 5090
(x86_64, driver 595.99.02; idle at the start of the run, though another process held memory on it)
and 50 runs on a Jetson Orin (JetPack 7.2, L4T R39.2.1, power mode MAXN), in the CI image.

| Case | RTX 5090 640x480 | RTX 5090 1280x720 | Jetson Orin 640x480 | Jetson Orin 1280x720 |
| --- | --- | --- | --- | --- |
| Euclidean, 2.5 cm | 0.243 | 0.316 | 1.467 | 3.128 |
| Euclidean, 1 cm | 0.291 | 0.447 | 2.124 | 4.260 |
| Euclidean, 5 cm | 0.238 | 0.827 | 1.386 | 4.467 |
| Euclidean, 2.5 cm, oriented boxes | 0.286 | 0.354 | 1.616 | 3.370 |
| Euclidean, 2.5 cm, `kDense` | 0.243 | 0.312 | 1.470 | 3.240 |
| Image window 2, 2.5 cm | 0.172 | 0.266 | 0.695 | 1.563 |
| Image window 1, 2.5 cm | 0.115 | 0.183 | 0.621 | 1.443 |
| Image window 2, 2.5 cm, oriented boxes | 0.213 | 0.308 | 0.690 | 1.557 |
| Wall filling the image, Euclidean, oriented boxes | 0.347 | 0.699 | 2.072 | 5.679 |
| Wall filling the image, image window 2, oriented boxes | 0.476 | 1.453 | 1.503 | 4.781 |

The wall is the worst case: a tilted plane at 1 to 1.5 m with every pixel valid, one cluster of
all 307 200 or 921 600 points. At 5 cm the two nearly touching spheres merge, hence 7 clusters at
1280x720. Sorting the grid dominates `kEuclidean` on the Jetson Orin. Every case waits for the GPU
once per frame, `kDense` included.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-clustering 1.0 REQUIRED CONFIG)  # finds holoscan-messages too
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
graph.add_flow(deproject->point_cloud, cluster->point_cloud);  // after ground removal
graph.add_flow(cluster->labels, consumer->labels);            // int32 [H, W]
graph.add_flow(cluster->clusters, consumer->clusters);        // float32 [K, 26]
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
- [holoscan-messages](../messages/) 1.x, installed, or added before this module by the repository
  root's build.

The module downloads nothing and needs no models or data. Its real-data test reads the local
recordings when `HOLOSCAN_TEST_DATA` names them.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/clustering -B build/clustering -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH="$PWD/install"   # where holoscan-messages is installed
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

The tests, on data made at test time but for `clustering_real_data`:

| Test | Needs | Checks |
| --- | --- | --- |
| `clustering_clusterer` | GPU | The clusterer against the truth and the CPU reference: boxes and spheres on a removed ground, touching and nearly touching, of a range of sizes, with jitter and noise, near the origin and far from it; random clouds and lattices on the tolerance; organized clouds in both connectivities and the synthetic depth camera at 640x480 and 1280x720; every output against the labels, summaries against a CPU sum, oriented boxes against a rotated box; the size limits and `max_clusters`; determinism across runs, instances, streams and scratch reuse; empty, all-invalid, all-noise, single-point, million-point and out-of-range clouds; rejected settings; and the drawing |
| `clustering_operator` | GPU | `EuclideanClusterOp` through the SDK's operator test harness, on `PointCloud` messages: organized and unorganized clouds in host and device memory, both layouts, both connectivities, with and without the clustered cloud, its colors, normals and intensities (carried only when configured and present) and oriented boxes, every output against the CPU reference, the capture time, frame id and header of every output, an empty cloud, a cloud without any cluster, a cloud above `max_points`, a malformed cloud, `kImage` with an unorganized cloud, and the rejection of unusable settings; through a graph on one worker, clouds from a producer's host, pinned host and device pools |
| `clustering_benchmark` | GPU | Runs the benchmark briefly, to keep it working |
| `clustering_real_data` | GPU, `HOLOSCAN_TEST_DATA` | The local recordings' first two depth images per clip, deprojected through their `depth.yaml`, clustered by the operator in both connectivities: labels equal to the CPU reference's, summaries consistent with them. Skipped without `HOLOSCAN_TEST_DATA` |
| `clustering_clusterer_memcheck`, `_racecheck`, `_initcheck`, `_synccheck` | GPU, `compute-sanitizer` | `clustering_clusterer` under each compute-sanitizer tool (memcheck with a leak check), failing on any report; label `sanitizer` |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a small graph on one worker (which it checks) that clusters synthetic
`PointCloud` messages with colors from host memory, packed by cluster with oriented boxes, and
checks the number of clusters against the truth and that the labels, summaries and clustered cloud
agree. Build it with the install prefix on `CMAKE_PREFIX_PATH` and
run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/clustering package holoscan-clustering --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-clustering_1.0.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)` and
`holoscan-messages` 1.x, so `apt install ./holoscan-messages_1.0.0_<arch>.deb
./holoscan-clustering_1.0.0_<arch>.deb` needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. From NVIDIA CCCL, header-only from the CUDA Toolkit and compiled into the library: CUB
(BSD-3-Clause; its copyright, conditions and disclaimer are in the NOTICE, which the Debian package
installs) and libcu++ (Apache-2.0 with LLVM exceptions). See [NOTICE](NOTICE).
