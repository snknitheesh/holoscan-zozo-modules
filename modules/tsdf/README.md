# holoscan-tsdf

A Holoscan SDK 5 Module for dense 3D reconstruction on the GPU: depth images from a camera whose
pose is known (from a robot's kinematics, a tracker or odometry) are fused into a truncated signed
distance (TSDF) volume, which gives back a clean triangle mesh of the scene and depth images and
point clouds from any viewpoint. The camera comes as `CameraInfo` and the ray cast goes out as a
`PointCloud`, the messages of [holoscan-messages](../messages/), so that the module sits in a
chain with the collection's other modules.

- `TsdfIntegrateOp` integrates each depth image (uint16 millimeters or float32 meters), with its
  aligned color when configured, into the volume from the camera's pose, through the camera its
  `CameraInfo` describes.
- `TsdfMeshOp` ray casts the volume from each pose it receives, through the camera of a
  `CameraInfo`, into a depth image and an organized `PointCloud` with normals and colors, and every
  N poses extracts the volume's mesh by marching cubes: vertices, normals, colors and triangle
  indices, with their counts, into fixed-size outputs.
- `TsdfVolume`, shared by the two operators, needs no graph: it integrates, ray casts, resets and
  saves its mesh as a PLY file from plain C++. `MeshExtractor` extracts the mesh into your own
  device buffers.

The [TSDF Reconstruction demo](../../applications/tsdf_reconstruction/) runs both end to end on a
synthetic RGB-D camera circling a known scene, its calibration written to a camera_info file and
sent by `CameraInfoSourceOp`, checks every result against the scene's exact surfaces, and shows the
input depth beside the shaded ray cast in HoloViz.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-tsdf)` |
| Targets | `holoscan::tsdf` |
| Headers | `<holoscan/tsdf/...>` |
| C++ namespace | `holoscan::tsdf` |
| Debian package | `holoscan-tsdf` |
| Uses | holoscan-messages 1.x: `CameraInfo` in, `PointCloud` out |
| Platforms | x86_64; aarch64 (Jetson Thor and Jetson Orin, JetPack 7.2) |

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

Every output carries the capture time and frame id of the frame it came from. An output that
exists must be connected, and an optional port exists only when configured. A refused emit fails
the frame, with an error that names the operator, its instance and the port: an operator cannot
send a refused output again. The two operators may run on different streams and threads: the
volume orders their GPU work with a CUDA event, so neither waits on the host for the other. Each
operator's configuration is fixed when it is constructed; it has no run-time parameters.

The camera info comes on a mandatory `camera_info` input, with the messages module's contract
([Take camera info in an operator](../messages/README.md#take-camera-info-in-an-operator)): at any
rate (`CameraInfoSourceOp` sends it periodically; connect it with
`latest_camera_info_connection()`), or with every frame (`CameraInfoTiming::kEveryFrame`). Each
frame uses the newest camera info received before it; frames before the first one are skipped.
The camera must be undistorted or rectified and without skew: a raw camera info (a distortion
coefficient that is not zero) fails the frame, as does a skew; undistort or rectify the images
first.

### `TsdfIntegrateOp`

| Port | Type | |
| --- | --- | --- |
| `depth` (input) | `schema::ImageT` | MONO16 (raw units, times `depth_scale`) or MONO32F, `[H, W]` or `[H, W, 1]`, packed, host or device |
| `color` (input, with `color`) | `schema::ImageT` | RGB8, BGR8, RGBA8 or BGRA8 of the depth image's size, aligned with it |
| `pose` (input) | `Tensor` | float32 or float64 `[4, 4]` world_from_camera, host memory (device memory is copied to the host, which waits) |
| `camera_info` (input) | `messages::schema::CameraInfoT` | The depth camera, for images of the depth image's size; mandatory |
| `integrated_pose` (with `pose_output`) | `Tensor` | float32 `[4, 4]`, host: the pose just integrated, to drive `TsdfMeshOp` |

It runs once per depth image. The depth image, its color and, by default, its pose carry one
capture time and are aligned exactly, as a tracked camera sends them. A pose from a source on its
own clock (a tracker or the robot's state at its own rate) needs `pose_max_age`: each depth image
then takes the newest pose captured at or before it and at most that long before, and a depth
image without one is skipped, with a warning the first time. Only the voxels inside the camera's
view, out to `depth_max` plus the truncation, are visited, one CUDA thread per voxel; each voxel
projects to its nearest pixel. Depths outside `[depth_min, depth_max]`, zero and non-finite depths
are no measurement.

| `TsdfIntegrateConfig` | Default | |
| --- | --- | --- |
| `volume` | | The shared `TsdfVolume`; required |
| `depth_scale` | `0.001` | Raw depth to meters: 0.001 for uint16 millimeters, 1 for float32 meters |
| `depth_min`, `depth_max` | `0.1`, `4.0` | The depth range in meters |
| `color` | `false` | Declare `color`; needs a volume with color |
| `pose_output` | `true` | Declare `integrated_pose` |
| `camera_info` | `kLatest`, 10 s | How the camera info arrives (`CameraInfoInputOptions`) |
| `pose_max_age` | `0` | 0: the pose carries the depth image's capture time. Above 0: poses on their own clock, at most this old |
| `max_width`, `max_height` | `1920`, `1080` | Largest image accepted; it sizes the staging of host input |

### `TsdfMeshOp`

| Port | Type | |
| --- | --- | --- |
| `pose` (input) | `Tensor` | float32 or float64 `[4, 4]` world_from_camera, host memory: `integrated_pose`, or any pose to look from |
| `camera_info` (input, with `raycast`) | `messages::schema::CameraInfoT` | The camera to ray cast through: its size and intrinsics; mandatory |
| `vertices` (with `mesh`) | `Tensor` | float32 `[max_vertices, 3]`, world meters |
| `normals` (with `mesh`) | `Tensor` | float32 `[max_vertices, 3]`, unit, pointing out of the surface |
| `colors` (with `mesh`, color volume) | `Tensor` | uint8 `[max_vertices, 3]` |
| `triangles` (with `mesh`) | `Tensor` | uint32 `[max_triangles, 3]`, counterclockwise seen from outside |
| `mesh_counts` (with `mesh`) | `Tensor` | uint32 `[4]`: vertices and triangles written, vertices and triangles the surface needs |
| `raycast_depth` (with `raycast`) | `schema::ImageT` | MONO32F `[H, W]` meters along the camera's z axis, 0 where a ray hits nothing; `H, W` from the camera info |
| `raycast_cloud` (with `raycast`) | `messages::schema::PointCloudT` | Organized `[H, W, 3]`: the hits in the camera's optical frame (NaN where a ray hits nothing), their unit normals facing the camera (NaN likewise), and with a color volume their RGB8 colors (black likewise); `valid_count` -1. Its header has the camera info's frame id and the pose's capture time |

The ray cast goes out on every pose; the mesh on the `mesh_every`-th, the twice `mesh_every`-th,
and so on. With `raycast`, poses before the first camera info are skipped, and not counted. To
mesh on demand, send a pose when a mesh is wanted, with `mesh_every` 1, or call `MeshExtractor`
yourself. `raycast_depth` and the camera info that went in describe a depth image, which
`DepthToPointCloudOp` of [holoscan-point-cloud](../point-cloud/) takes as it is.

The mesh outputs are this module's own tensors: the messages module has no mesh message, and a
mesh is not a point cloud (its vertex count is known only on the GPU, and its triangles index the
vertices). They have a fixed size, so the operator never waits for the GPU to learn the mesh's:
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
| `max_vertices`, `max_triangles` | 2^20, 2^21 | The mesh's capacity; it sizes the outputs and the scratch. The defaults take about 51 MiB per pool slot of the outputs plus 27 MiB of scratch: size them to the scene, on Jetson Orin especially |
| `raycast` | `false` | Declare the ray-cast outputs and `camera_info` |
| `raycast_depth_min`, `raycast_depth_max` | `0.1`, `4.0` | Rays start and end this far along the camera's z axis, meters |
| `max_width`, `max_height` | `1920`, `1080` | The largest ray cast; it sizes the outputs, and a camera info for larger images fails the pose |
| `camera_info` | `kLatest`, 10 s | How the camera info arrives (`CameraInfoInputOptions`) |

### Waits and copies

As the demo wires them (device depth and color, host poses), per frame: `TsdfIntegrateOp` and
`TsdfMeshOp` wait for nothing and copy nothing; the mesh scratch is allocated in `TsdfMeshOp`'s
`start()`. Otherwise:

| Input | Per frame |
| --- | --- |
| Depth or color in pinned host memory | One host-to-device copy each, on the stream (logged once) |
| Depth or color in pageable host memory | One host-to-device copy each, which returns once the data is staged: a wait (logged once) |
| A pose in device memory | One device-to-host copy and a wait (logged once) |
| Device input when the runtime gives the operator no stream (every port in host memory) | A wait for the integration, on the operator's own stream |

The staging buffer for host input is allocated once, on the first host frame, for the largest
image (`max_width` x `max_height`), so it never grows. `TsdfVolume::download()` and `export_ply()`
wait for the GPU.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-tsdf 1.0 REQUIRED CONFIG)  # Finds holoscan-messages too.
target_link_libraries(my_app PRIVATE holoscan::tsdf)
```

From `add_tsdf()` in [tests/installed_consumer/main.cpp](tests/installed_consumer/main.cpp), which
compiles both forms and runs the lossless one:

```cpp
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/tsdf/tsdf_integrate_op.hpp>
#include <holoscan/tsdf/tsdf_mesh_op.hpp>

namespace hm = holoscan::messages;
namespace ht = holoscan::tsdf;

ht::TsdfVolumeConfig config;  // 256^3 voxels of 1 cm around the origin.
auto volume = std::make_shared<ht::TsdfVolume>(config);

// The depth camera's ROS camera_info file.
auto info = graph.op<hm::CameraInfoSourceOp>(
    "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = "depth.yaml"});
auto integrate = graph.op<ht::TsdfIntegrateOp>(
    "integrate", ht::TsdfIntegrateConfig{.volume = volume, .depth_scale = 0.001F});
auto mesh = graph.op<ht::TsdfMeshOp>(
    "mesh", ht::TsdfMeshConfig{.volume = volume, .mesh_every = 30, .raycast = true});

// A live camera: keep its newest frames, never hold it back.
const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                         .queue_depth = 1U};
graph.add_flow(camera->depth, integrate->depth, latest);
graph.add_flow(camera->pose, integrate->pose, latest);
graph.add_flow(info->camera_info, integrate->camera_info, hm::latest_camera_info_connection());
graph.add_flow(info->camera_info, mesh->camera_info, hm::latest_camera_info_connection());
graph.add_flow(integrate->integrated_pose, mesh->pose, latest);
graph.add_flow(mesh->raycast_cloud, planner->cloud);  // And every other output of mesh.

// Later, from any thread:
volume->export_ply("scene.ply");
```

The mesh and ray-cast outputs are in device memory, so bind their GPU when compiling:

```cpp
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "mesh", .output_port = "raycast_cloud", .device = holoscan::DeviceId{0}});
```

When every frame must arrive (recorded data, tests), make the edges lossless (`kBlocking`) and run
the whole graph on one worker, as the demo and the installed consumer do:

```cpp
graph.partition("my_app").include_all();  // plan.scheduler_plan().worker_count is then 1.
```

Split into several workers, as the automatic partitioning does with a second source such as
`CameraInfoSourceOp`, an operator can run while its consumer's lossless queue is full, and the
refused emit fails the frame (see the messages module's
[Run a graph on one worker](../messages/README.md#run-a-graph-on-one-worker)). A source should
send a frame again when its allocation or emit is refused.

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
noise, camera orbits around it, its camera as a `CameraInfo` (`synthetic_camera_info()`), and its
exact distance and depth functions.

### API and ABI

Version 1.0.0. The headers under `<holoscan/tsdf/>` are the API, kept compatible within 1.x
(`SameMajorVersion`), except `detail::VolumeOp` in `volume_op.hpp`, the operators' shared base,
which is installed only because they derive from it and may change in any release. The library's
soname is `libholoscan_tsdf.so.1`. There are no Python bindings.

## Accuracy

All on frames rendered at test time, with depth noise of 1 mm plus 2 mm per square meter of depth
(about 3 mm at 1 m), on x86_64; the same tests pass on Jetson Orin.

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

On the local recordings of a static camera (real depth, not in the repository; see the tests
below), 20 images integrated with the identity pose into a volume of 512 voxels on its longest
side: the ray cast sees 85 to 86 % of the surfaces inside the volume, 97 to 98 % of them within two
voxels of the median input depth. The rest are thin or broken structures several pixels narrower
than a voxel. The results are the same on x86_64 and Jetson Orin.

## Performance

Milliseconds of GPU time, the median of 50 runs (`tsdf_benchmark --iterations 50`), on the
tabletop scene after a quarter orbit, with Holoscan SDK 5.0.0 and CUDA 13.2. Integration visits
only the voxels inside the camera's view; the mesh extraction reads the whole volume twice; the ray
cast marches half a voxel at a time near surfaces and writes depth, normals and points, as
`TsdfMeshOp` does.

| Step | RTX 5090, 640x480 | RTX 5090, 1280x720 | Jetson Orin, 640x480 | Jetson Orin, 1280x720 |
| --- | --- | --- | --- | --- |
| Integrate, 256^3 | 0.062 | 0.067 | 1.23 | 0.97 |
| Ray cast, 256^3 | 0.21 | 0.36 | 4.39 | 14.3 |
| Mesh, 256^3 (about 120k vertices, 240k triangles) | 0.40 | 0.39 | 10.4 | 7.94 |
| Integrate, 512^3 | 0.50 | 0.42 | 8.93 | 7.19 |
| Ray cast, 512^3 | 1.51 | 0.75 | 13.0 | 15.4 |
| Mesh, 512^3 (about 740k vertices, 1.5M triangles) | 2.95 | 2.88 | 64.2 | 62.6 |

GPU memory: the 256^3 volume 128 MiB, the 512^3 volume 1 GiB (with color 192 MiB and 1.5 GiB); the
benchmark's mesh capacity of 4M vertices and 8M triangles adds 110 to 124 MiB of scratch and 208 MiB
of outputs.

- The RTX 5090 (driver 595.99, CUDA 13.2) was shared with another process holding most of its
  memory, though nearly idle (2 % utilization) during the run.
- Jetson Orin ran JetPack 7.2 in power mode MAXN, with other containers running, which makes its
  times vary from run to run.
- 640x480 can take longer than 1280x720: both have the same horizontal field of view, and 4:3 sees
  more of the volume than 16:9, which integration and the ray cast pay for; the mesh does not
  depend on the image.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13 (with its CUB, BSD-3-Clause,
  header-only), CMake 4.0 or later and a C++20 compiler.
- [holoscan-messages](../messages/) 1.x, installed or built in the same tree.
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

The tests; all but `tsdf_real_data` on synthetic frames made at test time:

| Test | Needs | Checks |
| --- | --- | --- |
| `tsdf_volume` | GPU | `TsdfVolume` and `MeshExtractor`: the zero crossings, mesh vertices, normals, winding and colors against the exact surfaces; the triangle count settling and repeat extraction; the closed sphere's mesh watertight; the ray cast against the exact depth from poses on and off the orbit, and its points on their rays; the weight cap and truncation; the CPU reference bit for bit; determinism; empty depth, depths out of range, the camera inside the volume and refused arguments; the capacity and reset; the PLY file; rejected settings; the staging buffer's growth |
| `tsdf_operators` | GPU | Both operators through the SDK's operator test harness, with `CameraInfo` messages: uint16 and float32 depth in host and device memory, RGB8 and BGRA8 color, float32, float64 and device poses, camera info at its own rate and with every frame, a frame before the first camera info, a change of calibration, poses on their own clock (one too old, skipped), bit for bit against `TsdfVolume`; the mesh every second pose and the ray cast every pose (depth image and organized point cloud), bit for bit against `MeshExtractor` and `TsdfVolume::raycast()`; a pose before the first camera info; the capacity; capture times and frame ids; raw, skewed and wrongly sized cameras, rejected inputs and settings |
| `tsdf_real_data` | GPU, `HOLOSCAN_TEST_DATA` | The local recordings of a static camera (`stereo-camera/<clip>/` with `depth.yaml` and 16-bit PGM depth): 20 images with the identity pose and their camera_info file through `CameraInfoSourceOp`, both operators on one worker; the ray cast against the median input depth (see [Accuracy](#accuracy)). Skipped (77) when `HOLOSCAN_TEST_DATA` is not set, as in CI; a clip with depth on fewer than 5 % of its pixels is left out |
| `tsdf_benchmark` | GPU | Runs the benchmark briefly on 256^3, so that it keeps working; labelled `benchmark` too. It fails when every volume is skipped for lack of memory |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: the graph of [Use it in your graph](#use-it-in-your-graph), compiled live and
run lossless on one worker (it fails on any other worker count), from host-memory depth images of
the synthetic scene and a camera_info file it writes to a temporary folder; it checks the mesh
against the scene, every ray-cast point cloud, and saves a PLY file. Build it with the install
prefix on `CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/tsdf package holoscan-tsdf --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-tsdf_1.0.0_<arch>.deb` to the current
directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)` and on
`holoscan-messages` 1.x, so `apt install ./holoscan-messages_1.0.0_<arch>.deb
./holoscan-tsdf_1.0.0_<arch>.deb` needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. CUB, compiled into the library from the SDK's or the CUDA Toolkit's CCCL, is
BSD-3-Clause; the marching cubes triangle table, from "Polygonising a scalar field", is in the
public domain. See [NOTICE](NOTICE).
