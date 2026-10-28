# holoscan-point-cloud

A Holoscan SDK 5 Module for point clouds on the GPU, in both directions:

- `DepthToPointCloudOp` turns a depth image from a depth camera, a stereo matcher or a monocular
  depth network, with pinhole intrinsics, into an organized point cloud, optionally colored from an
  aligned color image. It is the step from 2D perception to 3D: mapping, ground and obstacle
  estimation, registration and grasp planning all start from it. The
  [Depth to Point Cloud demo](../../applications/depth_to_point_cloud/) runs it end to end.
- `PointCloudToDepthOp` projects a point cloud from a lidar, a depth camera or a map into a pinhole
  camera at a given pose, and makes the depth image that camera would see, optionally with the
  points' colors. It is the step from 3D back to 2D: lidar depth for image networks, depth
  completion and fusion with camera images, and virtual cameras over a map. The
  [Point Cloud to Depth demo](../../applications/point_cloud_to_depth/) runs it end to end.

Deprojecting a depth image and projecting the points back with the same intrinsics gives the image
back exactly.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-point-cloud)` |
| Targets | `holoscan::point_cloud` |
| Headers | `<holoscan/point_cloud/...>` |
| C++ namespace | `holoscan::point_cloud` |
| Debian package | `holoscan-point-cloud` |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

## Operators

### `DepthToPointCloudOp`

For each pixel `(u, v)` with metric depth `Z = raw_depth * depth_scale`, the point in the camera
optical frame (x right, y down, z forward) is:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z
```

Pixel centers are at integer coordinates, as in OpenCV and ROS `CameraInfo`. Zero, negative and
non-finite depths, and depths outside `[depth_min, depth_max]`, become `(NaN, NaN, NaN)` (or another
chosen invalid value). The cloud stays organized, `H x W x 3`, so pixel neighbors stay neighbors for
normals and segmentation. One CUDA thread handles one pixel. For a body frame such as ROS's
(x forward, y left, z up), rotate the cloud downstream.

| Port | Type | |
| --- | --- | --- |
| `depth` (input) | `holoscan::schema::ImageT` | `MONO16` (raw units, scaled by `depth_scale`) or `MONO32F`, `[H, W]` or `[H, W, 1]`, in host or CUDA device memory |
| `color` (input, with `color`) | `holoscan::schema::ImageT` | `RGB8`, `BGR8`, `RGBA8` or `BGRA8`, the depth image's size |
| `intrinsics` (input, with `intrinsics_input`) | `holoscan::Tensor` | float32 `[fx, fy, cx, cy]` for the depth image with the same capture time, in host or device memory |
| `point_cloud` (output) | `holoscan::Tensor` | float32 `[H, W, 3]`, meters, in device memory |
| `colors` (output, with `color`) | `holoscan::Tensor` | uint8 `[H, W, 3]` RGB per point, in device memory |

The optional ports exist only when configured. With them, the operator runs once per frame when the
depth, color and intrinsics with exactly the same capture time have all arrived; a depth image
without them gives no cloud. Intrinsics must therefore come with every depth image, stamped like it:
a camera that sends them at their own rate never completes a frame, and the graph produces nothing.
If the intrinsics do not change, configure them instead. Outputs carry the depth image's capture
time and frame id.

`DepthToPointCloudConfig`, passed to the constructor:

| Setting | Default | Meaning |
| --- | --- | --- |
| `intrinsics` | none | `{fx, fy, cx, cy}` in pixels; give these or set `intrinsics_input`, not both |
| `depth_scale` | `0.001` | Raw depth to meters: `0.001` for uint16 millimeters, `1.0` for float32 meters |
| `depth_min`, `depth_max` | `0`, `100` | Valid depth range in meters |
| `invalid_value` | `NaN` | Written to X, Y and Z of invalid pixels |
| `color` | `false` | Declare the `color` input and `colors` output |
| `intrinsics_input` | `false` | Declare the `intrinsics` input, for intrinsics that change from frame to frame |
| `max_width`, `max_height` | `1920`, `1080` | Largest image the outputs are sized for |

Rows must be packed, and data in device memory aligned to its element size (4 bytes for
four-channel color). The operator rejects an unsupported encoding, a color image of another size, an
image larger than its maximum, or unusable intrinsics in host memory, with an error that says so.
Intrinsics in device memory are read by the kernel, so the host never waits for them; unusable ones
(a focal length that is not positive and finite, or a center that is not finite) make every point of
that frame invalid instead.

### `PointCloudToDepthOp`

Each point `p` is first moved into the camera optical frame (x right, y down, z forward) by the
configured extrinsics, a 3x4 row-major `[R | t]`: `(x, y, z) = R * p + t`. Then it projects to

```text
u = fx * x / z + cx
v = fy * y / z + cy
```

and lands on the pixel nearest `(u, v)`, `(lroundf(u), lroundf(v))`, with pixel centers at integer
coordinates. Each pixel keeps the nearest point: a z-buffer of one 64-bit key per pixel, the point's
depth above its index, written with `atomicMin`. Among equally near points the lower index wins, so
the image does not depend on thread scheduling. A second kernel turns the winning keys into depth
and color.

Points are dropped when a coordinate is not finite, when `z` is not positive (behind the camera) or
lies outside `[depth_min, depth_max]`, and when their pixel is outside the image. With `point_size`
3 or 5, a point covers the square of that many pixels a side around its pixel, clipped at the image
border; a point whose own pixel is outside the image draws nothing. Pixels no point reaches are 0,
and black in the color image.

The depth image is float32 meters, or uint16 counts of `depth_scale` meters, `round(z /
depth_scale)` rounded half away from zero. A uint16 depth cannot hold a point whose count would be
0, which marks an empty pixel, or above 65535: such a point is dropped rather than saturated, so a
pixel never shows a depth no point has, and the next point behind it can show through instead.

| Port | Type | |
| --- | --- | --- |
| `points` (input) | `holoscan::Tensor` | float32 `[N, 3]`, or an organized `[H, W, 3]` taken as `H * W` points; in host, pinned or device memory |
| `colors` (input, with `color`) | `holoscan::Tensor` | uint8 RGB, the points' shape |
| `intrinsics` (input, with `intrinsics_input`) | `holoscan::Tensor` | float32 `[fx, fy, cx, cy]` for the cloud with the same capture time, in host or device memory |
| `depth` (output) | `holoscan::schema::ImageT` | `MONO32F` meters or `MONO16` counts, `[height, width]`, in device memory |
| `color` (output, with `color`) | `holoscan::schema::ImageT` | `RGB8`, `[height, width, 3]`, in device memory |

As for `DepthToPointCloudOp`, the optional ports exist only when configured, and with them the
operator runs once per cloud when the colors and intrinsics with exactly the same capture time have
all arrived. Outputs carry the cloud's capture time and frame id, and each image's header holds the
capture time and the configured `frame_id`.

The depth image of a frame goes out before its color image. Should the color then fail to go out,
the depth image has been delivered alone and the frame fails. Pair the two by capture time, as the
demo's reporter does, and connect a consumer that only displays them with
`QueuePolicy::kDropOldest`, so that it never pushes back.

`PointCloudToDepthConfig`, passed to the constructor:

| Setting | Default | Meaning |
| --- | --- | --- |
| `width`, `height` | `640`, `480` | Depth image size in pixels, 1..16384 |
| `intrinsics` | none | `{fx, fy, cx, cy}` in pixels; give these or set `intrinsics_input`, not both |
| `extrinsics` | none (identity) | 3x4 row-major `[R \| t]` from the points' frame into the camera optical frame |
| `depth_min`, `depth_max` | `0`, `100` | Valid depth range in meters |
| `depth_format` | `DepthFormat::kFloat32` | `kFloat32` meters, or `kUint16` counts of `depth_scale` meters |
| `depth_scale` | `0.001` | Meters per uint16 count: millimeters by default |
| `point_size` | `1` | Pixels a side each point covers: 1, 3 or 5 |
| `color` | `false` | Declare the `colors` input and `color` output |
| `frame_id` | empty | The camera's coordinate frame, for the images' headers |
| `intrinsics_input` | `false` | Declare the `intrinsics` input, for intrinsics that change from frame to frame |
| `max_points` | `16777216` | Largest cloud accepted, up to 2^30 points; a larger one fails the frame |

Inputs must be packed, and points and intrinsics in device memory aligned to 4 bytes. Inputs in
device memory must be on the operator's GPU. The operator rejects another element type or shape,
colors that do not match the points, a cloud above `max_points`, or unusable intrinsics in host
memory, with an error that says so. Intrinsics in device memory are read by the kernel, so the host
never waits for them; unusable ones leave that frame's image empty instead. Clouds in host memory
are copied to the GPU through a staging buffer that grows with the largest cloud seen.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-point-cloud 1.1 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::point_cloud)
```

```cpp
#include <holoscan/point_cloud/depth_to_point_cloud_op.hpp>

namespace pc = holoscan::point_cloud;

auto cloud = graph.op<pc::DepthToPointCloudOp>(
    "deproject", pc::DepthToPointCloudConfig{
                     .intrinsics = pc::CameraIntrinsics{615.0F, 615.0F, 319.5F, 239.5F},
                     .depth_scale = 0.001F,  // uint16 millimeters
                     .depth_min = 0.1F,
                     .depth_max = 10.0F,
                     .max_width = 640,
                     .max_height = 480});
graph.add_flow(camera->depth, cloud->depth);            // Output<holoscan::schema::ImageT>
graph.add_flow(cloud->point_cloud, consumer->points);  // Input<holoscan::Tensor>
```

```cpp
#include <holoscan/point_cloud/point_cloud_to_depth_op.hpp>

auto project = graph.op<pc::PointCloudToDepthOp>(
    "project", pc::PointCloudToDepthConfig{
                   .width = 640,
                   .height = 480,
                   .intrinsics = pc::CameraIntrinsics{615.0F, 615.0F, 319.5F, 239.5F},
                   // Lidar x forward, y left, z up into the camera optical frame,
                   // the lidar 10 cm above the camera.
                   .extrinsics = pc::Extrinsics{{0.0F, -1.0F, 0.0F, 0.0F,    //
                                                 0.0F, 0.0F, -1.0F, -0.1F,   //
                                                 1.0F, 0.0F, 0.0F, 0.0F}},
                   .depth_min = 0.3F,
                   .depth_max = 50.0F,
                   .depth_format = pc::DepthFormat::kUint16,  // millimeters
                   .point_size = 3});
graph.add_flow(lidar->points, project->points);   // Output<holoscan::Tensor>, float32 [N, 3]
graph.add_flow(project->depth, consumer->depth);  // Input<holoscan::schema::ImageT>
```

The outputs of both operators are in device memory, so bind their GPU when compiling:

```cpp
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "deproject", .output_port = "point_cloud", .device = holoscan::DeviceId{0}});
const auto plan = holoscan::compile(graph, std::move(options));
```

The kernels need no Holoscan graph: `holoscan::point_cloud::launch_deproject()` in
`<holoscan/point_cloud/deproject.hpp>` and `launch_project()` in
`<holoscan/point_cloud/project.hpp>`. For tests, `<holoscan/point_cloud/synthetic_depth.hpp>` is a
synthetic depth camera that renders frames on the GPU and evaluates the same depths on the host,
`<holoscan/point_cloud/synthetic_scene.hpp>` makes shuffled clouds in a body frame from it, with
points mixed in that must not reach the image, and `<holoscan/point_cloud/depth_truth.hpp>` checks
a depth image made from such a cloud against the camera. Host code that compares against them
exactly must be compiled with `-ffp-contract=off`.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler.

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/point-cloud -B build/point-cloud -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/point-cloud
ctest --test-dir build/point-cloud --output-on-failure
cmake --install build/point-cloud --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_POINT_CLOUD_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests:

| Test | Needs | Checks |
| --- | --- | --- |
| `point_cloud_deproject` | GPU | The kernel against a CPU reference: float32 and uint16 depth, holes, the valid range, non-finite depth, every color layout and the channel swap, image sizes that are not a multiple of the block size, intrinsics in device memory, usable or not, rejected color layouts, and launch error reporting |
| `point_cloud_depth_to_point_cloud_op` | GPU | `DepthToPointCloudOp` through the SDK's operator test harness and in small graphs: host and device input, color in every encoding and per-frame intrinsics, every point and color against the synthetic camera, the capture time and frame id of the outputs, frames whose intrinsics carry another capture time, how host input is held until its copy is done, and the rejection of misaligned data and unusable inputs and settings |
| `point_cloud_project` | GPU | The projection kernels against a CPU reference, pixel for pixel: random clouds through a rigid transform, every splat size, float32 and uint16 depth, colors, ties and the z-buffer's determinism, the valid range, non-finite points, points behind the camera, the image border, uint16 rounding and its limits, odd image sizes, intrinsics in device memory, and the round trip of a depth image through its points; the synthetic cloud's GPU and host versions, the rigid inverse, and launch error reporting |
| `point_cloud_point_cloud_to_depth_op` | GPU | `PointCloudToDepthOp` through the SDK's operator test harness and in small graphs: organized clouds that give their image back exactly, unorganized synthetic clouds with colors in a body frame against ground truth, configured and per-frame intrinsics, the capture time and frame id of the outputs, clouds whose intrinsics carry another capture time, an empty cloud, the image headers, how host input is held, and the rejection of misaligned, foreign-device and unusable inputs and settings |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a small graph that deprojects synthetic uint16 depth images from host memory,
checks every point, and projects the clouds back into the camera, which must give every depth back
exactly. Build it with the install prefix on `CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/point-cloud package holoscan-point-cloud --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-point-cloud_1.1.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)`, so
`apt install ./holoscan-point-cloud_1.1.0_<arch>.deb` needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. See [NOTICE](NOTICE).
